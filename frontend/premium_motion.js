(() => {
  "use strict";

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  const finePointer = window.matchMedia("(pointer: fine)");
  const canvas = document.getElementById("ambientCanvas");
  const isProductApp = document.body.classList.contains("product-app");
  const logicalProcessors = navigator.hardwareConcurrency || 4;
  const deviceMemory = navigator.deviceMemory || 4;
  const lowPowerDevice = logicalProcessors <= 4 || deviceMemory <= 4;

  function initializeCanvas() {
    if (!canvas || reducedMotion.matches) return;
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) return;

    let width = 0;
    let height = 0;
    let frame = 0;
    let lastFrame = 0;
    let particles = [];
    let visible = !document.hidden;
    const pointer = { x: -1000, y: -1000, active: false };
    const targetFps = lowPowerDevice ? 20 : 30;
    const frameInterval = 1000 / targetFps;

    function particleCount() {
      const area = width * height;
      const maximum = lowPowerDevice ? 16 : isProductApp ? 22 : 32;
      const minimum = lowPowerDevice ? 10 : 14;
      return Math.max(minimum, Math.min(maximum, Math.round(area / 62000)));
    }

    function createParticle(index) {
      const angle = (index * 2.399963229728653) % (Math.PI * 2);
      const speed = 0.055 + (index % 7) * 0.012;
      return {
        x: Math.random() * width,
        y: Math.random() * height,
        vx: Math.cos(angle) * speed,
        vy: Math.sin(angle) * speed,
        radius: 0.55 + (index % 4) * 0.28,
        alpha: 0.16 + (index % 5) * 0.035,
      };
    }

    function resize() {
      const pixelRatio = 1;
      width = window.innerWidth;
      height = window.innerHeight;
      canvas.width = Math.round(width * pixelRatio);
      canvas.height = Math.round(height * pixelRatio);
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
      particles = Array.from({ length: particleCount() }, (_, index) => createParticle(index));
    }

    function updateParticle(particle) {
      if (pointer.active) {
        const dx = particle.x - pointer.x;
        const dy = particle.y - pointer.y;
        const distanceSquared = dx * dx + dy * dy;
        if (distanceSquared < 18000 && distanceSquared > 1) {
          const force = (18000 - distanceSquared) / 18000;
          const distance = Math.sqrt(distanceSquared);
          particle.vx += (dx / distance) * force * 0.018;
          particle.vy += (dy / distance) * force * 0.018;
        }
      }

      particle.vx *= 0.992;
      particle.vy *= 0.992;
      particle.x += particle.vx;
      particle.y += particle.vy;

      if (particle.x < -12) particle.x = width + 12;
      if (particle.x > width + 12) particle.x = -12;
      if (particle.y < -12) particle.y = height + 12;
      if (particle.y > height + 12) particle.y = -12;
    }

    function draw(timestamp = 0) {
      if (!visible) return;
      if (timestamp - lastFrame < frameInterval) {
        frame = requestAnimationFrame(draw);
        return;
      }
      lastFrame = timestamp;
      context.clearRect(0, 0, width, height);

      for (let first = 0; first < particles.length; first += 1) {
        const particle = particles[first];
        updateParticle(particle);
        context.beginPath();
        context.arc(particle.x, particle.y, particle.radius, 0, Math.PI * 2);
        context.fillStyle = `rgba(210, 214, 235, ${particle.alpha})`;
        context.fill();

        const connectionStep = isProductApp ? 2 : 1;
        for (let second = first + 1; second < particles.length; second += connectionStep) {
          const neighbor = particles[second];
          const dx = particle.x - neighbor.x;
          const dy = particle.y - neighbor.y;
          const distanceSquared = dx * dx + dy * dy;
          if (distanceSquared > 10500) continue;
          const opacity = (1 - distanceSquared / 10500) * 0.045;
          context.beginPath();
          context.moveTo(particle.x, particle.y);
          context.lineTo(neighbor.x, neighbor.y);
          context.strokeStyle = `rgba(184, 191, 226, ${opacity})`;
          context.lineWidth = 0.7;
          context.stroke();
        }
      }

      frame = requestAnimationFrame(draw);
    }

    window.addEventListener("resize", resize, { passive: true });
    window.addEventListener("pointermove", (event) => {
      pointer.x = event.clientX;
      pointer.y = event.clientY;
      pointer.active = true;
    }, { passive: true });
    document.addEventListener("pointerleave", () => {
      pointer.active = false;
    });
    document.addEventListener("visibilitychange", () => {
      visible = !document.hidden;
      if (visible) {
        cancelAnimationFrame(frame);
        draw();
      } else {
        cancelAnimationFrame(frame);
      }
    });

    resize();
    draw();
  }

  function initializeReveal() {
    const selectors = [
      "body.marketing-page main > section",
      "body.marketing-page .workflow-card",
      "body.marketing-page .feature-card",
      "body.marketing-page .privacy-card",
      "body.marketing-page .price-card",
      "body.marketing-page .enterprise-point",
      "body.marketing-page .faq-item",
    ];
    const targets = [...document.querySelectorAll(selectors.join(","))];
    if (!targets.length) return;

    targets.forEach((target, index) => {
      target.classList.add("motion-target");
      target.style.setProperty("--reveal-index", String(index % 6));
    });

    if (reducedMotion.matches || !("IntersectionObserver" in window)) {
      targets.forEach((target) => target.classList.add("is-visible"));
      return;
    }

    const observer = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        entry.target.classList.add("is-visible");
        observer.unobserve(entry.target);
      });
    }, { threshold: 0.12, rootMargin: "0px 0px -7% 0px" });

    targets.forEach((target) => observer.observe(target));
  }

  function initializePointerPhysics() {
    if (reducedMotion.matches || !finePointer.matches) return;
    const tiltSelector = [
      ".workspace-preview",
      ".model-constellation",
    ].join(",");

    document.addEventListener("pointermove", (event) => {
      const surface = event.target.closest(tiltSelector);
      if (!surface) return;
      const bounds = surface.getBoundingClientRect();
      const horizontal = (event.clientX - bounds.left) / bounds.width - 0.5;
      const vertical = (event.clientY - bounds.top) / bounds.height - 0.5;
      const strength = surface.classList.contains("workspace-preview") ? 3.2 : 1.8;
      surface.classList.add("tilt-surface");
      surface.style.setProperty("--tilt-x", `${(-vertical * strength).toFixed(2)}deg`);
      surface.style.setProperty("--tilt-y", `${(horizontal * strength).toFixed(2)}deg`);
      surface.style.setProperty("--lift", surface.classList.contains("workspace-preview") ? "-4px" : "-2px");
      surface.style.setProperty("--pointer-x", `${((event.clientX - bounds.left) / bounds.width) * 100}%`);
      surface.style.setProperty("--pointer-y", `${((event.clientY - bounds.top) / bounds.height) * 100}%`);
    }, { passive: true });

    document.addEventListener("pointerout", (event) => {
      const surface = event.target.closest(tiltSelector);
      if (!surface || surface.contains(event.relatedTarget)) return;
      surface.style.setProperty("--tilt-x", "0deg");
      surface.style.setProperty("--tilt-y", "0deg");
      surface.style.setProperty("--lift", "0px");
    });
  }

  function initializeModelStage() {
    const stage = document.querySelector("[data-model-constellation]");
    const buttons = [...document.querySelectorAll("[data-model-stage]")];
    if (!stage || !buttons.length) return;

    const content = {
      profile: {
        title: "Profile the evidence",
        copy: "Map types, missingness, cardinality, and target candidates before model choices enter the picture.",
      },
      compare: {
        title: "Compare on the same ground",
        copy: "Evaluate compatible candidates against a shared validation plan and keep the champion selection legible.",
      },
      explain: {
        title: "Trace the final decision",
        copy: "Inspect performance, feature signals, and exportable artifacts without losing the original project context.",
      },
    };

    const applyStage = (name) => {
      const selected = content[name] || content.profile;
      stage.dataset.activeStage = name;
      buttons.forEach((button) => {
        const active = button.dataset.modelStage === name;
        button.classList.toggle("active", active);
        button.setAttribute("aria-selected", String(active));
      });
      document.querySelectorAll("[data-model-card]").forEach((card) => {
        card.classList.toggle("active", card.dataset.modelCard === name);
      });
      const title = document.getElementById("modelStageTitle");
      const copy = document.getElementById("modelStageCopy");
      if (title) title.textContent = selected.title;
      if (copy) copy.textContent = selected.copy;
    };

    buttons.forEach((button) => {
      button.addEventListener("click", () => applyStage(button.dataset.modelStage));
    });
  }

  function initializeMagneticActions() {
    if (reducedMotion.matches || !finePointer.matches) return;

    document.addEventListener("pointermove", (event) => {
      const button = event.target.closest(".button-primary");
      if (!button) return;
      const bounds = button.getBoundingClientRect();
      const x = (event.clientX - bounds.left - bounds.width / 2) * 0.045;
      const y = (event.clientY - bounds.top - bounds.height / 2) * 0.07;
      button.classList.add("magnetic");
      button.style.setProperty("--magnetic-x", `${x.toFixed(2)}px`);
      button.style.setProperty("--magnetic-y", `${y.toFixed(2)}px`);
    }, { passive: true });

    document.addEventListener("pointerout", (event) => {
      const button = event.target.closest(".button-primary");
      if (!button || button.contains(event.relatedTarget)) return;
      button.style.setProperty("--magnetic-x", "0px");
      button.style.setProperty("--magnetic-y", "0px");
    });
  }

  initializeCanvas();
  initializeReveal();
  initializePointerPhysics();
  initializeMagneticActions();
  initializeModelStage();
})();
