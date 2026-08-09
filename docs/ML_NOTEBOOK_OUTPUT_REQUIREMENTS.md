ML NOTEBOOK AND OUTPUT QUALITY REQUIREMENT

The machine-learning part of this project must not produce only raw model metrics, charts, or predictions.

For every dataset and every generated machine-learning experiment, the system must create a complete, professional, educational, and presentation-ready notebook/report that explains the entire workflow step by step.

The final output should resemble a high-quality Kaggle notebook, portfolio project, or industry ML report.

The notebook must not look like automatically generated disconnected code cells.

It must look like a human data scientist carefully studied the dataset, made decisions based on evidence, implemented the solution, and explained every decision clearly.

PRIMARY OUTPUT FORMAT

For every project, generate a notebook such as:

notebooks/generated/<project_name>_complete_ml_workflow.ipynb

Also generate:

reports/<project_name>_ml_report.html
reports/<project_name>_ml_report.pdf
reports/<project_name>_model_summary.json
reports/<project_name>_metrics.csv

The notebook must contain:

- Markdown explanation cells
- Properly organised code cells
- Tables
- Charts
- Model outputs
- Result interpretations
- Conclusions
- Recommendations
- Reproducible code

NOTEBOOK STYLE

Use the following presentation style:

# Project Title

A clear title such as:

🎓 Predicting Student Mental Health Score from Social Media Usage

Then explain:

- Problem statement
- What is being predicted
- Why the problem matters
- Target variable
- Whether the problem is classification, regression, clustering, forecasting, anomaly detection, or another task
- Why that task type was selected
- Dataset name
- Number of rows and columns
- Important feature groups
- Expected business or practical outcome

Example writing style:

“Problem: We are predicting Mental_Health_Score, which is a continuous numeric value. Therefore, this is a regression problem rather than classification.”

The explanation must be adapted to the actual dataset and must never be copied blindly.

REQUIRED NOTEBOOK STRUCTURE

1. PROJECT OVERVIEW

Include:

- Project title
- Problem statement
- Objective
- Target variable
- ML task type
- Dataset overview
- Practical use case
- Success criteria

Explain the task in simple but technically correct language.

2. IMPORT LIBRARIES

Group imports according to purpose:

- Data handling
- Visualisation
- Statistical analysis
- Preprocessing
- Feature engineering
- Model selection
- Models
- Evaluation
- Explainability
- Model saving
- Utilities

Before the code, explain why imports are grouped.

Do not import unnecessary libraries.

3. LOAD THE DATASET

Show:

- File name
- Successful loading message
- Shape
- First five rows
- Last five rows where useful
- Random sample
- Column names

Explain what each initial command tells us.

4. INITIAL DATA UNDERSTANDING

Include:

- df.shape
- df.head()
- df.sample()
- df.info()
- df.describe()
- df.describe(include="object")
- Column data types
- Unique-value counts
- Target summary

After every important output, add a “Reading this” or “Interpretation” section.

Example:

“📌 Reading this: The dataset contains 5,000 rows and 13 columns. Seven columns are numeric and six are categorical.”

The interpretation must come from the actual output.

5. DATA-QUALITY ASSESSMENT

Check and report:

- Missing values
- Duplicate rows
- Constant columns
- Near-constant columns
- Invalid values
- Impossible values
- Negative values where impossible
- Infinite values
- Incorrect data types
- Inconsistent categories
- Leading and trailing spaces
- High-cardinality columns
- Identifier-like columns
- Date-column issues
- Target leakage risks
- Class imbalance for classification
- Target distribution for regression

Use tables where possible.

For each issue, explain:

- What was found
- Why it matters
- Whether it will be fixed
- How it will be fixed
- Why that method was selected

Do not claim “zero duplicates” when duplicates exist.

All written interpretations must match the actual calculated output.

6. EXPLORATORY DATA ANALYSIS

Do not generate a wall of irrelevant charts.

Each chart must answer a specific question.

Before every plot, include:

- The question being investigated
- Why the plot was selected
- What relationship is expected

After every plot, include:

- What the chart shows
- Important patterns
- Possible outliers
- Whether the observation may influence modelling
- A warning that correlation does not prove causation where relevant

For regression projects, include relevant plots such as:

- Target histogram
- Target KDE
- Target box plot
- Numeric-feature distributions
- Correlation heatmap
- Scatter plots against the target
- Regression trend plots
- Categorical feature vs target box plots
- Categorical feature vs target violin plots
- Pair plots only when the number of features is manageable
- Missing-value visualisation where applicable
- Outlier visualisation
- Feature cardinality charts

For classification projects, include:

- Class-distribution chart
- Class percentage table
- Numeric distributions by class
- Categorical distributions by class
- Box plots by class
- Correlation analysis
- Crosstabs
- Target-rate charts
- Imbalance discussion

Only generate plots relevant to the current dataset.

Use readable:

- Titles
- Axis labels
- Rotated tick labels
- Legends
- Figure sizes
- Tight layout

7. OUTLIER ANALYSIS

Use appropriate methods such as:

- IQR
- Z-score
- Domain constraints
- Isolation Forest only where justified

Create an outlier summary table containing:

| Feature | Lower Bound | Upper Bound | Outlier Count | Outlier Percentage | Recommended Action |

Do not automatically remove all statistical outliers.

For every affected feature, determine whether the values are:

- Valid rare values
- Measurement errors
- Data-entry errors
- Impossible values
- Influential but legitimate observations

Explain why each treatment is chosen:

- Keep
- Clip
- Winsorise
- Transform
- Replace
- Remove rows
- Use a robust model

8. DATA CLEANING

Show the cleaning operations clearly.

Possible operations:

- Remove exact duplicates
- Fix column names
- Strip whitespace
- Standardise category labels
- Correct data types
- Parse dates
- Handle missing values
- Replace impossible values
- Clip invalid ranges
- Remove leakage columns
- Handle identifiers

For every cleaning action, include an explanation such as:

“Negative physical-activity hours are impossible. We clip them to zero rather than deleting the entire row because the remaining student information is still valid.”

Do not manufacture cleaning steps when the dataset does not need them.

Show before-and-after comparisons.

9. SKEWNESS AND DISTRIBUTION ANALYSIS

For numeric columns, calculate:

- Skewness
- Distribution summary
- Transformation recommendation

Create a table:

| Feature | Skewness | Distribution Type | Recommended Transformation |

Interpret skewness:

- Approximately symmetric
- Moderately skewed
- Highly skewed

Use transformations only where justified:

- log1p
- square root
- Yeo-Johnson
- Box-Cox when valid
- No transformation

Do not transform the target or features unnecessarily.

10. FEATURE ENGINEERING

Feature engineering must be based on dataset meaning.

For every engineered feature, explain:

- Original columns used
- Formula or transformation
- Domain reason
- Expected benefit
- Leakage check

Examples:

- Grouping rare categories
- Date-derived features
- Ratios
- Differences
- Interaction features
- Binning
- Aggregations
- Behavioural indicators
- Frequency encoding
- Target-independent category grouping

For high-cardinality columns, evaluate:

- Grouping rare categories
- Frequency encoding
- Hashing
- Target encoding with leakage-safe cross-validation
- Dropping the column if it is identifier-like

Do not generate meaningless polynomial features blindly.

11. ENCODING STRATEGY

Create a clear encoding plan before implementing it.

Create a table:

| Feature | Feature Type | Encoding Method | Reason |

Examples:

- Ordered categories → OrdinalEncoder
- Unordered categories → OneHotEncoder
- High-cardinality categories → grouping, frequency encoding, or leakage-safe target encoding
- Binary categories → binary encoding where appropriate

Explicitly explain why nominal categories should not be given artificial numeric order.

12. TRAIN, VALIDATION, AND TEST SPLIT

Explain:

- Why unseen data is needed
- Selected split ratio
- Why the ratio is suitable
- Random seed
- Whether stratification is used
- Whether a time-based split is required
- Leakage prevention

Default behaviour:

- Classification: stratified split when appropriate
- Regression: random split unless temporal or grouped structure requires another strategy
- Time series: chronological split
- Grouped data: GroupShuffleSplit or GroupKFold

All preprocessing must be fitted only on training data.

13. PREPROCESSING WITH COLUMNTRANSFORMER

Build separate pipelines based on column type.

Possible branches:

- Skewed numeric features
- Regular numeric features
- Ordinal categorical features
- Nominal categorical features
- Text features
- Date-derived features

Include, where appropriate:

- SimpleImputer
- StandardScaler
- RobustScaler
- MinMaxScaler
- FunctionTransformer
- PowerTransformer
- OrdinalEncoder
- OneHotEncoder
- TF-IDF

Explain why every transformation was chosen.

Do not scale features unnecessarily for tree-only models unless a shared comparison pipeline requires it.

14. BUILD REUSABLE PIPELINES

Use sklearn Pipeline to combine:

- Preprocessing
- Feature selection where needed
- Model

Explain that the pipeline:

- Prevents inconsistent preprocessing
- Reduces leakage
- Simplifies cross-validation
- Makes deployment safer
- Allows raw inputs during inference
- Can be serialised as one object

15. BASELINE MODEL

Always train a simple baseline first.

Regression baselines may include:

- DummyRegressor
- LinearRegression
- Ridge
- Lasso

Classification baselines may include:

- DummyClassifier
- LogisticRegression

Explain:

- Why the baseline was chosen
- What performance level it establishes
- Whether advanced models meaningfully outperform it

16. MULTIPLE-MODEL COMPARISON

Train several appropriate models based on:

- Task type
- Dataset size
- Feature types
- Interpretability needs
- Computational limits

Regression candidates may include:

- Linear Regression
- Ridge
- Lasso
- Elastic Net
- Decision Tree
- Random Forest
- Gradient Boosting
- HistGradientBoosting
- XGBoost if installed
- LightGBM if installed
- CatBoost if installed

Classification candidates may include:

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting
- HistGradientBoosting
- XGBoost
- LightGBM
- CatBoost
- SVM where appropriate
- KNN where appropriate

Do not train every possible model without reason.

Create a model comparison table.

Regression table:

| Model | Train R² | Validation R² | Test R² | MAE | RMSE | CV Mean | CV Std | Training Time |

Classification table:

| Model | Train Accuracy | Test Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | CV Mean | CV Std | Training Time |

Clearly identify:

- Best-performing model
- Most stable model
- Most interpretable model
- Fastest model
- Generalisation gap
- Signs of overfitting

17. CROSS-VALIDATION

Use appropriate cross-validation:

- KFold for regression
- StratifiedKFold for classification
- TimeSeriesSplit for time series
- GroupKFold for grouped data

Show:

- Fold scores
- Mean
- Standard deviation
- Min
- Max

Explain what the variation across folds says about model stability.

18. HYPERPARAMETER TUNING

Tune only promising models.

Use:

- RandomizedSearchCV
- GridSearchCV for small search spaces
- Optuna when available and justified

Explain pipeline parameter naming such as:

model_step__parameter_name

Report:

- Search space
- Number of combinations
- Cross-validation strategy
- Scoring metric
- Best parameters
- Best cross-validation score
- Test-set performance after tuning
- Whether tuning actually improved the model

Never claim tuning improved the model unless the metrics support it.

19. MODEL EVALUATION

For regression, report:

- R²
- Adjusted R² where appropriate
- MAE
- MSE
- RMSE
- MAPE only when valid
- Median absolute error
- Explained variance
- Train-test performance gap

Include:

- Actual vs predicted plot
- Residual plot
- Residual distribution
- Error distribution
- Worst-prediction table
- Prediction-interval discussion where applicable

For classification, report:

- Accuracy
- Precision
- Recall
- F1
- Specificity where useful
- ROC-AUC
- PR-AUC
- Confusion matrix
- Classification report
- Log loss
- Brier score where probabilities are available

Include:

- ROC curve
- Precision-recall curve
- Confusion-matrix heatmap
- Probability distribution
- Calibration curve
- Threshold analysis

Explain every metric in relation to the actual problem.

Do not call R² “accuracy”.

Use:

“R² score” or “coefficient of determination”

instead of:

“accuracy of regression”.

20. THRESHOLD OPTIMISATION FOR CLASSIFICATION

When classification probabilities are available:

- Do not optimise the threshold using the final test set.
- Use validation or out-of-fold predictions.
- Compare thresholds.
- Report precision, recall, F1, specificity, and business cost.
- Lock the chosen threshold.
- Evaluate it once on the untouched test set.

Explain the trade-off clearly.

21. ERROR ANALYSIS

Create an error-analysis section.

For regression:

- Largest absolute errors
- Underpredictions
- Overpredictions
- Error by category
- Error by target range
- Error by important feature ranges

For classification:

- False positives
- False negatives
- Error rates by subgroup
- Confidence of wrong predictions
- Common patterns among mistakes

Explain what the model struggles with.

22. FEATURE IMPORTANCE AND EXPLAINABILITY

Use appropriate techniques:

- Coefficients for linear models
- Native feature importance for tree models
- Permutation importance
- SHAP where supported
- Partial dependence where useful

Include:

- Global importance
- Local explanation for sample predictions
- Direction of impact where possible
- Limitations of feature importance

Do not present correlation or feature importance as proof of causation.

23. MODEL SELECTION

Select the champion model using more than one metric.

Consider:

- Predictive performance
- Cross-validation stability
- Generalisation gap
- Interpretability
- Inference speed
- Model size
- Deployment complexity
- Probability quality
- Business objective

Create a final selection summary:

| Criterion | Champion Model Result | Reason |

Clearly explain why the model was selected.

24. MODEL SAVING

Save:

- Complete preprocessing and model pipeline
- Metadata
- Feature schema
- Expected input data types
- Target name
- Task type
- Class labels if classification
- Decision threshold if classification
- Metrics
- Training timestamp
- Random seed
- Library versions
- Dataset fingerprint
- Model version

Suggested files:

models/<project_name>_pipeline.joblib
models/<project_name>_metadata.json
models/<project_name>_feature_schema.json
models/<project_name>_metrics.json

Verify model loading after saving.

Run a prediction using the reloaded model and confirm that it matches the original prediction.

25. SAMPLE PREDICTION

Show at least one realistic sample input.

Display:

- Raw input
- Prediction
- Probability or confidence where applicable
- Human-readable interpretation
- Input validation warnings where needed

For regression, example:

“Predicted Mental Health Score: 6.72”

For classification, example:

“Predicted Churn Risk: Yes, with probability 72.4%, using decision threshold 0.34.”

Do not present probability as certainty.

26. FINAL CONCLUSION

Include:

- What problem was solved
- Dataset summary
- Most important EDA findings
- Cleaning decisions
- Important engineered features
- Models compared
- Best model
- Final metrics
- Generalisation assessment
- Limitations
- Ethical considerations
- Recommended next steps
- Deployment readiness

The conclusion should be understandable to both technical and non-technical readers.

27. PROJECT QUALITY SCORECARD

Create a final scorecard:

| Area | Status | Notes |

Include:

- Data quality
- EDA quality
- Leakage prevention
- Preprocessing
- Model comparison
- Cross-validation
- Tuning
- Evaluation
- Explainability
- Reproducibility
- Deployment readiness
- Documentation

28. AUTOMATIC WRITING REQUIREMENTS

All generated explanations must be based on actual computed values.

Never generate text such as:

- “There are no duplicates” unless duplicate count is zero
- “The data is balanced” without calculating class proportions
- “The model performs excellently” without supporting metrics
- “Feature X is the most important” without computed evidence
- “Tuning improved performance” without comparing before and after

The report-generation logic must insert real values into the narrative.

The notebook must remain technically honest.

29. OUTPUT PRESENTATION REQUIREMENTS

Use:

- Clear numbered headings
- Concise paragraphs
- “📌 Reading this” sections
- “Why this matters” sections
- Tables instead of long unstructured print outputs
- Proper chart titles
- Proper axis labels
- Rounded metrics
- Consistent decimal formatting
- Highlighted best-model rows
- Clear warning callouts
- Clean notebook flow

Avoid:

- Repeated df.columns calls
- Repeated df.describe calls without reason
- Unexplained code
- Huge raw outputs
- Random charts
- Duplicate plots
- Misleading metric names
- Broken cells
- Hardcoded results
- Outputs that contradict the text

30. GENERATED NOTEBOOK EXECUTION

After generating the notebook:

1. Execute every cell from top to bottom.
2. Fix all errors.
3. Ensure outputs are embedded.
4. Confirm charts render correctly.
5. Confirm saved artefacts exist.
6. Confirm the final pipeline reloads.
7. Confirm predictions work.
8. Confirm text interpretations match outputs.
9. Export the executed notebook to HTML.
10. Export to PDF where the environment supports it.

Do not mark notebook generation complete unless the notebook executes successfully from a clean kernel.

31. DATASET-SPECIFIC ADAPTATION

The system must adapt the workflow to the dataset.

Examples:

- Do not perform ordinal encoding unless categories have a real order.
- Do not use log transformation just because skewness is positive.
- Do not use stratification for ordinary regression without a justified binning strategy.
- Do not apply SMOTE before splitting.
- Do not use classification metrics for regression.
- Do not use random splitting for time series.
- Do not drop all outliers automatically.
- Do not one-hot encode identifier columns.
- Do not retain target-derived features.
- Do not use the final test set repeatedly during model development.

The output must reflect intelligent decisions rather than a fixed template.

32. CODE QUALITY

Generated notebook code must be:

- Reusable
- Modular
- Deterministic
- Readable
- PEP 8 compliant where practical
- Free of unnecessary repetition
- Divided into useful helper functions
- Safe against missing optional libraries
- Compatible with the selected Python environment

Create helper functions for:

- Dataset overview
- Missing-value summary
- Duplicate summary
- Outlier summary
- Plot generation
- Metric calculation
- Cross-validation summary
- Model comparison
- Error analysis
- Metadata saving

33. IMPORTANT CORRECTIONS TO THE EXAMPLE STYLE

Follow the quality and presentation style of the provided example, but correct technical mistakes automatically.

For example:

- If an output reports two duplicate rows, do not write “zero duplicate rows”.
- A regression R² score must not be labelled as accuracy.
- If the explanation says 80/20, the code must not use test_size=0.30.
- Imputation mentioned in the explanation must actually appear in the pipeline.
- High cardinality must be measured rather than assumed.
- Data cleaning must happen before feature calculations that depend on the cleaned values.
- Feature engineering performed outside the pipeline must be reproducible during API inference.
- Rare-category grouping should be implemented inside a fitted transformer or preserved in model metadata.
- Outliers should not be removed merely because they cross the IQR limits.
- Test-set metrics must not influence hyperparameter or threshold selection.

34. PLATFORM USER EXPERIENCE

Within the application, provide a “Generated ML Notebook” page.

It should show:

- Notebook generation progress
- Current section being processed
- Completed sections
- Warnings found
- Models trained
- Best model
- Main metrics
- Download notebook
- View HTML report
- Download PDF report
- Download trained pipeline
- Download metrics
- Download feature schema

The generated result must feel like a complete data-science project, not merely an API response.

FINAL ACCEPTANCE CRITERIA

This feature is complete only when:

- A user can upload a valid dataset.
- The system analyses the dataset.
- The system identifies the task type or asks the user to confirm it.
- The user selects the target.
- A complete notebook is generated.
- The notebook contains explanations, code, outputs, charts, and interpretations.
- The notebook executes without errors.
- Multiple relevant models are compared.
- The champion model is selected transparently.
- The fitted pipeline is saved.
- A sample prediction works.
- The report exports successfully.
- No fake values are shown.
- No narrative contradicts computed outputs.
- The result is professional enough to include in a portfolio.