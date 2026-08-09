"""AI-powered ML advisor using Mistral AI - Full autonomous ML pipeline generation."""

import os
import json
import re
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
import numpy as np
from io import StringIO
import traceback

try:
    from mistralai import Mistral
except ImportError:
    Mistral = None


class AIAdvisor:
    """Mistral AI-powered advisor for intelligent ML recommendations."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize Mistral AI client."""
        self.api_key = api_key or os.getenv("MISTRAL_API_KEY")
        self.model = os.getenv("MISTRAL_MODEL", "mistral-large-latest")
        
        if not self.api_key:
            raise ValueError("Mistral API key not found. Set MISTRAL_API_KEY environment variable.")
        
        if Mistral is None:
            raise ImportError("mistralai package not installed. Run: pip install mistralai")
        
        self.client = Mistral(api_key=self.api_key)
        self.enabled = True
    
    def _call_mistral(self, system_prompt: str, user_prompt: str) -> str:
        """Make a call to Mistral AI API."""
        try:
            response = self.client.chat.complete(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Mistral AI API error: {e}")
            return None
    
    def get_dataset_insights(self, profile: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered insights about the dataset quality and characteristics."""
        
        system_prompt = """You are an expert data scientist and ML engineer. Analyze dataset profiles and provide actionable insights about data quality, potential issues, and recommendations for preprocessing.

Your analysis should be:
- Concise and actionable
- Focused on practical ML implications
- Based on statistical evidence
- Formatted as JSON with specific recommendations"""

        user_prompt = f"""Analyze this dataset profile and provide insights:

Dataset: {profile.get('filename', 'Unknown')}
Rows: {profile.get('n_rows', 0):,}
Columns: {profile.get('n_cols', 0)}
Missing Values: {profile.get('total_missing', 0):,}
Duplicates: {profile.get('n_duplicates', 0)}

Column Details:
{json.dumps([{
    'name': c['name'],
    'type': c['semantic_type'],
    'missing_pct': c['missing_pct'],
    'n_unique': c['n_unique']
} for c in profile.get('columns', [])[:15]], indent=2)}

Data Quality Warnings:
{json.dumps([w['message'] for w in profile.get('warnings', [])], indent=2)}

Provide a JSON response with this structure:
{{
    "overall_quality": "excellent|good|fair|poor",
    "quality_score": 0-100,
    "key_insights": ["insight1", "insight2", "insight3"],
    "preprocessing_recommendations": ["rec1", "rec2"],
    "potential_issues": ["issue1", "issue2"],
    "suggested_approaches": ["approach1", "approach2"]
}}"""

        response = self._call_mistral(system_prompt, user_prompt)
        
        if response:
            try:
                # Extract JSON from response (handle markdown code blocks)
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                insights = json.loads(response)
                return {
                    "ai_powered": True,
                    "provider": "Mistral AI",
                    "model": self.model,
                    **insights
                }
            except json.JSONDecodeError:
                # Fallback: return as raw text
                return {
                    "ai_powered": True,
                    "provider": "Mistral AI",
                    "model": self.model,
                    "raw_insights": response
                }
        
        return {"ai_powered": False, "error": "Failed to get AI insights"}
    
    def recommend_models(self, 
                        profile: Dict[str, Any],
                        target_column: str,
                        task_type: str) -> Dict[str, Any]:
        """Get AI-powered model recommendations based on dataset characteristics."""
        
        system_prompt = """You are an expert ML engineer specializing in model selection and architecture design. Based on dataset characteristics, recommend the most appropriate machine learning models and explain why.

Consider:
- Dataset size and dimensionality
- Feature types (numeric, categorical)
- Class balance for classification
- Data quality and missing values
- Computational efficiency
- Interpretability vs performance tradeoffs

Provide specific, actionable model recommendations with justifications."""

        # Gather dataset statistics
        n_rows = profile.get('n_rows', 0)
        n_cols = profile.get('n_cols', 0)
        columns = profile.get('columns', [])
        
        numeric_cols = [c for c in columns if c.get('semantic_type') == 'numeric']
        categorical_cols = [c for c in columns if c.get('semantic_type') == 'categorical']
        
        target_info = next((c for c in columns if c['name'] == target_column), {})
        
        user_prompt = f"""Recommend ML models for this dataset:

**Dataset Characteristics:**
- Task Type: {task_type}
- Target Column: {target_column}
- Total Rows: {n_rows:,}
- Total Features: {n_cols}
- Numeric Features: {len(numeric_cols)}
- Categorical Features: {len(categorical_cols)}

**Target Column Info:**
- Type: {target_info.get('semantic_type', 'unknown')}
- Unique Values: {target_info.get('n_unique', 'N/A')}
- Missing: {target_info.get('missing_pct', 0)}%

**Data Quality:**
- Total Missing Values: {profile.get('total_missing', 0):,}
- Duplicates: {profile.get('n_duplicates', 0)}

Provide recommendations in JSON format:
{{
    "recommended_models": [
        {{
            "name": "Model Name",
            "priority": "high|medium|low",
            "rationale": "Why this model is suitable",
            "expected_performance": "Performance expectations",
            "hyperparameters": {{"key": "suggested values"}},
            "pros": ["pro1", "pro2"],
            "cons": ["con1", "con2"]
        }}
    ],
    "feature_engineering_suggestions": ["suggestion1", "suggestion2"],
    "training_recommendations": ["rec1", "rec2"],
    "expected_challenges": ["challenge1", "challenge2"]
}}

Recommend 3-5 models suitable for this {task_type} task."""

        response = self._call_mistral(system_prompt, user_prompt)
        
        if response:
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                recommendations = json.loads(response)
                return {
                    "ai_powered": True,
                    "provider": "Mistral AI",
                    "model": self.model,
                    **recommendations
                }
            except json.JSONDecodeError:
                return {
                    "ai_powered": True,
                    "provider": "Mistral AI",
                    "raw_recommendations": response
                }
        
        return {"ai_powered": False, "error": "Failed to get model recommendations"}
    
    def explain_results(self, workflow_result: Dict[str, Any]) -> Dict[str, Any]:
        """Get AI-powered explanations of model results and performance."""
        
        system_prompt = """You are an expert ML engineer who explains model results to both technical and non-technical audiences. Provide clear, insightful interpretations of model performance, highlighting strengths, weaknesses, and actionable next steps.

Focus on:
- What the metrics mean in practical terms
- Whether the results are good for this use case
- Specific improvements that could be made
- Potential deployment considerations"""

        user_prompt = f"""Explain these ML workflow results:

**Model:** {workflow_result.get('best_model', 'Unknown')}
**Task:** {workflow_result.get('task_type', 'Unknown')}
**Target:** {workflow_result.get('target_column', 'Unknown')}

**Metrics:**
{json.dumps(workflow_result.get('metrics', {}), indent=2)}

**Training Info:**
{json.dumps(workflow_result.get('split_info', {}), indent=2)}

**Models Compared:**
{json.dumps(workflow_result.get('models_compared', []), indent=2)}

**Tuning Result:**
{json.dumps(workflow_result.get('tuning_result', {}), indent=2) if workflow_result.get('tuning_result') else 'Not performed'}

**Top Features:**
{json.dumps(workflow_result.get('feature_importance', [])[:10], indent=2)}

**Warnings:**
{json.dumps(workflow_result.get('warnings', []), indent=2)}

Provide analysis in JSON format:
{{
    "performance_summary": "Overall assessment of model performance",
    "performance_rating": "excellent|good|fair|poor",
    "key_findings": ["finding1", "finding2", "finding3"],
    "strengths": ["strength1", "strength2"],
    "weaknesses": ["weakness1", "weakness2"],
    "improvement_suggestions": ["suggestion1", "suggestion2"],
    "deployment_readiness": "production-ready|needs-improvement|not-ready",
    "next_steps": ["step1", "step2", "step3"]
}}"""

        response = self._call_mistral(system_prompt, user_prompt)
        
        if response:
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                explanation = json.loads(response)
                return {
                    "ai_powered": True,
                    "provider": "Mistral AI",
                    "model": self.model,
                    **explanation
                }
            except json.JSONDecodeError:
                return {
                    "ai_powered": True,
                    "provider": "Mistral AI",
                    "raw_explanation": response
                }
        
        return {"ai_powered": False, "error": "Failed to get result explanation"}
    
    def suggest_feature_engineering(self, 
                                   df: pd.DataFrame,
                                   target_column: str,
                                   task_type: str) -> Dict[str, Any]:
        """Get AI-powered feature engineering suggestions."""
        
        system_prompt = """You are an expert feature engineer. Analyze dataset structure and suggest creative, effective feature engineering techniques that could improve model performance.

Suggest:
- Interaction features
- Polynomial features
- Domain-specific transformations
- Feature selection strategies
- Dimensionality reduction approaches"""

        # Sample data statistics
        columns_info = []
        for col in df.columns[:20]:  # Limit to first 20 columns
            if col == target_column:
                continue
            
            col_info = {
                "name": col,
                "dtype": str(df[col].dtype),
                "unique": int(df[col].nunique()),
                "missing_pct": round(float(df[col].isnull().mean() * 100), 2)
            }
            
            if pd.api.types.is_numeric_dtype(df[col]):
                col_info["stats"] = {
                    "mean": round(float(df[col].mean()), 4),
                    "std": round(float(df[col].std()), 4),
                    "min": round(float(df[col].min()), 4),
                    "max": round(float(df[col].max()), 4)
                }
            
            columns_info.append(col_info)

        user_prompt = f"""Suggest feature engineering strategies for this dataset:

**Task:** {task_type}
**Target:** {target_column}
**Dataset Size:** {len(df)} rows × {len(df.columns)} columns

**Feature Details:**
{json.dumps(columns_info, indent=2)}

Provide suggestions in JSON format:
{{
    "interaction_features": [
        {{"features": ["col1", "col2"], "operation": "multiply|add|divide", "rationale": "why"}}
    ],
    "transformations": [
        {{"feature": "col_name", "transform": "log|sqrt|polynomial", "rationale": "why"}}
    ],
    "domain_specific": ["suggestion1", "suggestion2"],
    "feature_selection": ["strategy1", "strategy2"],
    "priority_actions": ["high_impact_action1", "high_impact_action2"]
}}"""

        response = self._call_mistral(system_prompt, user_prompt)
        
        if response:
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                suggestions = json.loads(response)
                return {
                    "ai_powered": True,
                    "provider": "Mistral AI",
                    "model": self.model,
                    **suggestions
                }
            except json.JSONDecodeError:
                return {
                    "ai_powered": True,
                    "provider": "Mistral AI",
                    "raw_suggestions": response
                }
        
        return {"ai_powered": False, "error": "Failed to get feature engineering suggestions"}
    
    def generate_complete_ml_pipeline(self,
                                     df: pd.DataFrame,
                                     target_column: str,
                                     approach: str = "best") -> Dict[str, Any]:
        """Generate a complete, executable ML pipeline using Mistral AI.
        
        This is the CORE function - AI generates everything:
        - Data preprocessing code
        - Feature engineering code
        - Model selection with hyperparameters
        - Training code
        - Evaluation code
        """
        
        system_prompt = """You are an expert ML engineer who writes production-ready Python code.

Your task: Generate a COMPLETE, EXECUTABLE machine learning pipeline based on dataset analysis.

Requirements:
1. Analyze the dataset characteristics
2. Design optimal preprocessing steps
3. Select best algorithms with proper hyperparameters
4. Write clean, production-ready Python code
5. Include proper error handling
6. Add evaluation metrics

Return a JSON with this EXACT structure:
{
    "analysis": {
        "data_quality": "assessment of data quality",
        "complexity": "simple|moderate|complex",
        "recommended_approach": "description",
        "expected_performance": "performance expectations"
    },
    "preprocessing": {
        "steps": ["step1", "step2"],
        "rationale": "why these steps"
    },
    "models": [
        {
            "name": "Model Name",
            "algorithm": "sklearn class name",
            "hyperparameters": {"param": "value"},
            "priority": "high|medium|low",
            "expected_metric": 0.85,
            "rationale": "why this model"
        }
    ],
    "feature_engineering": {
        "transformations": ["transformation1", "transformation2"],
        "new_features": ["feature1", "feature2"]
    },
    "code": {
        "imports": "import statements",
        "preprocessing_code": "preprocessing function code",
        "training_code": "model training code",
        "evaluation_code": "evaluation code"
    },
    "optimization": {
        "tuning_strategy": "strategy description",
        "cv_folds": 5,
        "search_space": {"param": ["val1", "val2"]}
    }
}

Write REAL, EXECUTABLE code that can be run directly."""

        # Prepare dataset statistics
        stats = self._prepare_dataset_stats(df, target_column)
        
        user_prompt = f"""Create a complete ML pipeline for this dataset:

**Dataset Characteristics:**
- Shape: {len(df)} rows × {len(df.columns)} columns
- Target: {target_column}
- Target Type: {df[target_column].dtype}
- Target Unique Values: {df[target_column].nunique()}
- Missing Values: {df.isnull().sum().sum()} total

**Column Summary:**
{json.dumps(stats['columns'], indent=2)}

**Data Quality Issues:**
{json.dumps(stats['quality_issues'], indent=2)}

**Correlation with Target (top 10):**
{json.dumps(stats['correlations'], indent=2)}

**User Requirement:** {approach}

Generate the COMPLETE pipeline with executable code. Make it production-ready and optimized for best performance."""

        response = self._call_mistral(system_prompt, user_prompt)
        
        if response:
            try:
                # Clean and parse JSON
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                pipeline = json.loads(response)
                pipeline["ai_powered"] = True
                pipeline["provider"] = "Mistral AI"
                pipeline["model"] = self.model
                
                return pipeline
            except json.JSONDecodeError as e:
                # Return structured error with raw response
                return {
                    "ai_powered": True,
                    "provider": "Mistral AI",
                    "error": f"JSON parse error: {str(e)}",
                    "raw_response": response[:1000],  # First 1000 chars
                    "fallback": self._generate_fallback_pipeline(df, target_column)
                }
        
        return {
            "ai_powered": False,
            "error": "Failed to generate pipeline",
            "fallback": self._generate_fallback_pipeline(df, target_column)
        }
    
    def _prepare_dataset_stats(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Prepare detailed dataset statistics for AI analysis."""
        
        stats = {
            "columns": [],
            "quality_issues": [],
            "correlations": {}
        }
        
        try:
            # Column info
            for col in df.columns:
                if col == target_column:
                    continue  # Skip target column in features
                    
                col_info = {
                    "name": col,
                    "dtype": str(df[col].dtype),
                    "unique": int(df[col].nunique()),
                    "missing": int(df[col].isnull().sum()),
                    "missing_pct": round(float(df[col].isnull().sum() / len(df) * 100), 2) if len(df) > 0 else 0
                }
                
                if pd.api.types.is_numeric_dtype(df[col]):
                    try:
                        if not df[col].isnull().all():
                            col_info["stats"] = {
                                "mean": round(float(df[col].mean()), 4),
                                "std": round(float(df[col].std()), 4),
                                "min": round(float(df[col].min()), 4),
                                "max": round(float(df[col].max()), 4)
                            }
                    except Exception:
                        pass  # Skip if stats fail
                
                stats["columns"].append(col_info)
            
            # Quality issues
            if df.duplicated().sum() > 0:
                stats["quality_issues"].append(f"{df.duplicated().sum()} duplicate rows")
            
            high_missing = [col for col in df.columns if df[col].isnull().sum() / len(df) > 0.5]
            if high_missing:
                stats["quality_issues"].append(f"High missing values in: {', '.join(high_missing)}")
            
            # Correlations with target (if numeric)
            try:
                if pd.api.types.is_numeric_dtype(df[target_column]):
                    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
                    if target_column in numeric_cols and len(numeric_cols) > 1:
                        corr = df[numeric_cols].corr()[target_column].drop(target_column, errors='ignore')
                        top_corr = corr.abs().nlargest(10)
                        stats["correlations"] = {str(k): round(float(v), 4) for k, v in top_corr.items()}
            except Exception:
                pass  # Skip correlations if it fails
                
        except Exception as e:
            print(f"Error preparing dataset stats: {e}")
            # Return minimal stats
            pass
        
        return stats
    
    def _generate_fallback_pipeline(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Generate a basic fallback pipeline if AI fails."""
        
        is_classification = df[target_column].nunique() <= 20
        
        return {
            "analysis": {
                "data_quality": "automated assessment",
                "complexity": "moderate",
                "recommended_approach": "Standard sklearn pipeline",
                "expected_performance": "Measured after cross-validation"
            },
            "models": [
                {
                    "name": "Random Forest",
                    "algorithm": "RandomForestClassifier" if is_classification else "RandomForestRegressor",
                    "hyperparameters": {"n_estimators": 100, "max_depth": 10},
                    "priority": "high"
                }
            ],
            "preprocessing": {
                "steps": ["Handle missing values", "Scale features", "Encode categoricals"],
                "rationale": "Standard preprocessing pipeline"
            }
        }
    
    def generate_multiple_approaches(self,
                                    df: pd.DataFrame,
                                    target_column: str) -> Dict[str, Any]:
        """Generate multiple different ML approaches for user to choose from.
        
        Returns 3-5 different strategies with pros/cons.
        """
        
        system_prompt = """You are an expert ML consultant presenting multiple solution approaches.

Generate 3-5 DIFFERENT ML strategies, each with distinct trade-offs.

Examples:
- Fast & Simple: Logistic Regression, quick training
- Balanced: Random Forest, good performance vs speed
- High Performance: XGBoost + extensive tuning, best accuracy
- Interpretable: Linear models + feature selection
- Ensemble: Multiple models combined

For each approach, provide:
- Clear name and description
- Relative compute level
- Metrics that must be measured during validation
- Pros and cons
- Best use case

Do not invent expected accuracy, scores, or training times. Performance is
unknown until the approach runs on cross-validation and a final holdout.

Return JSON:
{
    "approaches": [
        {
            "id": "fast",
            "name": "Fast & Simple",
            "description": "Quick baseline with logistic regression",
            "models": ["LogisticRegression", "SGDClassifier"],
            "compute_level": "low",
            "expected_accuracy": null,
            "complexity": "low",
            "pros": ["Very fast", "Interpretable", "Low resource usage"],
            "cons": ["May underfit complex patterns", "Limited feature interactions"],
            "best_for": "Quick prototyping, simple datasets",
            "recommended": false
        }
    ],
    "recommendation": "ID of recommended approach",
    "rationale": "Why this is recommended"
}"""

        stats = self._prepare_dataset_stats(df, target_column)
        
        user_prompt = f"""Design multiple ML approaches for this dataset:

Dataset: {len(df)} rows × {len(df.columns)} columns
Target: {target_column} ({df[target_column].nunique()} unique values)
Task: {"Classification" if df[target_column].nunique() <= 20 else "Regression"}

Column Summary: {json.dumps(stats['columns'][:10], indent=2)}

Generate 3-5 diverse approaches with clear trade-offs."""

        response = self._call_mistral(system_prompt, user_prompt)
        
        if response:
            try:
                if "```json" in response:
                    response = response.split("```json")[1].split("```")[0].strip()
                elif "```" in response:
                    response = response.split("```")[1].split("```")[0].strip()
                
                approaches = json.loads(response)
                approaches["ai_powered"] = True
                approaches["provider"] = "Mistral AI"
                return approaches
            except Exception as e:
                print(f"Failed to parse AI response: {e}")
                # Fall through to fallback
        
        # Always return fallback if AI fails
        return self._generate_fallback_approaches(df, target_column)
    
    def _generate_fallback_approaches(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Generate fallback approaches if AI fails."""
        
        is_classification = df[target_column].nunique() <= 20
        
        return {
            "ai_powered": False,
            "approaches": [
                {
                    "id": "fast",
                    "name": "Fast & Simple",
                    "description": "Quick baseline model",
                    "models": ["LogisticRegression"] if is_classification else ["Ridge"],
                    "compute_level": "low",
                    "expected_accuracy": None,
                    "complexity": "low",
                    "pros": ["Very fast", "Interpretable"],
                    "cons": ["May underfit"],
                    "best_for": "Quick prototyping",
                    "recommended": False
                },
                {
                    "id": "balanced",
                    "name": "Balanced Approach",
                    "description": "Good performance with Random Forest",
                    "models": ["RandomForestClassifier"] if is_classification else ["RandomForestRegressor"],
                    "compute_level": "moderate",
                    "expected_accuracy": None,
                    "complexity": "medium",
                    "pros": ["Handles complex patterns", "Feature importance"],
                    "cons": ["Slower than linear models", "Less interpretable"],
                    "best_for": "Most use cases",
                    "recommended": True
                },
                {
                    "id": "high_performance",
                    "name": "High Performance",
                    "description": "Broader search with randomized tuning",
                    "models": ["RandomForestClassifier"] if is_classification else ["RandomForestRegressor"],
                    "compute_level": "high",
                    "expected_accuracy": None,
                    "complexity": "high",
                    "pros": ["Hyperparameter tuning included", "Broader search"],
                    "cons": ["Longer training time", "More computational resources"],
                    "best_for": "Deeper experimentation when compute is available",
                    "recommended": False
                }
            ],
            "recommendation": "balanced",
            "rationale": "Best balance of performance, speed, and resource usage for most datasets"
        }


# Singleton instance
_advisor_instance = None

def get_ai_advisor() -> Optional[AIAdvisor]:
    """Get the remote advisor only after an explicit environment opt-in."""
    global _advisor_instance

    enabled = os.getenv(
        "AIVORAML_ENABLE_REMOTE_AI",
        os.getenv("REDIMENT_ENABLE_REMOTE_AI", "false"),
    ).strip().lower() in {"1", "true", "yes"}
    if not enabled:
        return None
    
    if _advisor_instance is None:
        try:
            _advisor_instance = AIAdvisor()
        except Exception as e:
            print(f"Failed to initialize AI Advisor: {e}")
            return None
    
    return _advisor_instance
