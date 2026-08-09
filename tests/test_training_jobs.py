"""Focused concurrency tests for the local training job manager."""

import json
import math
import sys
import threading
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from backend.services.training_jobs import (  # noqa: E402
    ActiveTrainingJobError,
    TrainingJobCancelled,
    TrainingJobManager,
    TrainingJobManagerShutdownError,
    TrainingJobNotFoundError,
)


class TrainingJobManagerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.manager = TrainingJobManager(
            max_workers=2,
            max_history=3,
            max_outstanding_jobs=4,
        )

    def tearDown(self) -> None:
        self.manager.shutdown(wait=True, cancel_futures=True)

    def test_success_progress_result_and_json_snapshot(self) -> None:
        def runner(progress, is_cancelled):
            self.assertFalse(is_cancelled())
            progress(25, "preparing", "Preparing data")
            progress(70, "fitting", "Fitting models")
            progress(60, "ignored-regression", "Progress stays monotonic")
            return {
                "score": float("nan"),
                "created": datetime(2026, 1, 1, tzinfo=timezone.utc),
                "path": Path("model.joblib"),
                "_pipeline": object(),
            }

        created = self.manager.submit(
            "project-one",
            runner,
            metadata={"target": "churn", "non_finite": math.inf},
        )
        settled = self.manager.wait(created["job_id"], timeout=2)

        self.assertEqual(settled["state"], "succeeded")
        self.assertEqual(settled["progress"], 100.0)
        self.assertEqual(settled["stage"], "completed")
        self.assertIsNotNone(settled["started_at"])
        self.assertIsNotNone(settled["finished_at"])
        self.assertIsNone(settled["result"]["score"])
        self.assertEqual(settled["result"]["path"], "model.joblib")
        self.assertNotIn("_pipeline", settled["result"])
        self.assertIsNone(settled["metadata"]["non_finite"])
        json.dumps(settled)
        self.assertEqual(
            self.manager.get_result(created["job_id"])["path"],
            Path("model.joblib"),
        )

    def test_only_one_active_job_per_project(self) -> None:
        started = threading.Event()
        release = threading.Event()

        def blocking_runner(progress, is_cancelled):
            started.set()
            release.wait(timeout=2)
            return "done"

        first = self.manager.submit("same-project", blocking_runner)
        self.assertTrue(started.wait(timeout=1))
        with self.assertRaises(ActiveTrainingJobError) as captured:
            self.manager.submit("same-project", blocking_runner)
        self.assertEqual(captured.exception.active_job_id, first["job_id"])
        self.assertEqual(
            self.manager.get_active_job("same-project")["job_id"],
            first["job_id"],
        )

        release.set()
        self.assertEqual(
            self.manager.wait(first["job_id"], timeout=2)["state"],
            "succeeded",
        )
        second = self.manager.submit("same-project", lambda _p, _c: "again")
        self.assertEqual(
            self.manager.wait(second["job_id"], timeout=2)["state"],
            "succeeded",
        )

    def test_running_job_cancels_cooperatively(self) -> None:
        started = threading.Event()

        def cancellable_runner(progress, is_cancelled):
            started.set()
            while not is_cancelled():
                progress(40, "fitting", "Working")
                time.sleep(0.005)
            raise TrainingJobCancelled("cancelled by test")

        created = self.manager.submit("cancel-project", cancellable_runner)
        self.assertTrue(started.wait(timeout=1))
        requested = self.manager.request_cancel(created["job_id"])
        self.assertIn(requested["state"], {"cancel_requested", "cancelled"})
        settled = self.manager.wait(created["job_id"], timeout=2)
        self.assertEqual(settled["state"], "cancelled")
        self.assertTrue(settled["cancellation_requested"])
        self.assertIsNone(settled["result"])
        self.assertIsNone(self.manager.get_active_job("cancel-project"))

    def test_queued_job_is_cancelled_before_runner_starts(self) -> None:
        manager = TrainingJobManager(
            max_workers=1,
            max_history=3,
            max_outstanding_jobs=3,
        )
        blocker_started = threading.Event()
        release_blocker = threading.Event()
        queued_runner_started = threading.Event()

        def blocker(_progress, _is_cancelled):
            blocker_started.set()
            release_blocker.wait(timeout=2)

        def queued_runner(_progress, _is_cancelled):
            queued_runner_started.set()
            return "should not run"

        try:
            first = manager.submit("first-project", blocker)
            self.assertTrue(blocker_started.wait(timeout=1))
            queued = manager.submit("queued-project", queued_runner)
            cancelled = manager.request_cancel(queued["job_id"])
            self.assertEqual(cancelled["state"], "cancelled")
            self.assertFalse(queued_runner_started.is_set())
            self.assertIsNone(manager.get_active_job("queued-project"))
            release_blocker.set()
            manager.wait(first["job_id"], timeout=2)
        finally:
            release_blocker.set()
            manager.shutdown(wait=True, cancel_futures=True)

    def test_failure_is_normalized_without_traceback(self) -> None:
        def failing_runner(_progress, _is_cancelled):
            raise ValueError("invalid target")

        created = self.manager.submit("failure-project", failing_runner)
        settled = self.manager.wait(created["job_id"], timeout=2)
        self.assertEqual(settled["state"], "failed")
        self.assertEqual(settled["error"]["type"], "ValueError")
        self.assertEqual(settled["error"]["message"], "invalid target")
        self.assertNotIn("traceback", json.dumps(settled).lower())

    def test_history_is_bounded_and_newest_first(self) -> None:
        manager = TrainingJobManager(
            max_workers=1,
            max_history=2,
            max_outstanding_jobs=2,
        )
        try:
            job_ids = []
            for index in range(4):
                created = manager.submit(
                    f"project-{index}",
                    lambda _progress, _cancelled, value=index: value,
                )
                job_ids.append(created["job_id"])
                manager.wait(created["job_id"], timeout=2)

            history = manager.list_jobs()
            self.assertEqual(len(history), 2)
            self.assertEqual(
                [item["job_id"] for item in history],
                list(reversed(job_ids[-2:])),
            )
            with self.assertRaises(TrainingJobNotFoundError):
                manager.get_job(job_ids[0])
        finally:
            manager.shutdown(wait=True, cancel_futures=True)

    def test_shutdown_is_idempotent_and_rejects_submission(self) -> None:
        self.manager.shutdown(wait=True)
        self.manager.shutdown(wait=True)
        with self.assertRaises(TrainingJobManagerShutdownError):
            self.manager.submit("late-project", lambda _p, _c: None)


if __name__ == "__main__":
    unittest.main()
