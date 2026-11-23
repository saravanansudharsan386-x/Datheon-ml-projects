# Datheon ML Projects

Machine-learning evaluation notebooks built for the Datheon internship. Each notebook follows the same pattern: load data, split without leakage, train models, and evaluate with clear metrics and plots.

## Projects

| # | Notebook | Problem | Models |
|---|----------|---------|--------|
| 1 | `task1_iris_classification.ipynb` | Iris species classification | Logistic Regression, Decision Tree |
| 2 | `task2_housing_regression.ipynb` | California Housing price prediction (coming soon) | OLS, Ridge, Lasso |
| 3 | `task3_churn_classification.ipynb` | Telco customer churn (coming soon) | Random Forest, XGBoost |

## Task 1: Iris Classification

Results on a stratified 20% hold-out set (30 flowers):

| Model | Correct | Accuracy |
|-------|---------|----------|
| Logistic Regression | 28 / 30 | ~93.3% |
| Decision Tree (max_depth=3) | 29 / 30 | ~96.7% |

Key observations:
- Setosa is perfectly separable; both models classify it 100% correctly.
- Versicolor and virginica overlap slightly, so all errors happen between those two classes.
- Logistic Regression draws straight (linear) boundaries, while the Decision Tree draws axis-aligned boxes.
- 5-fold cross-validation is included to check the result is not a lucky split.

## Installation

Requires Python 3.9 or newer.

```bash
# 1. Clone the repo
git clone https://github.com/saravanansudharsan386-x/datheon-ml-projects.git
cd datheon-ml-projects

# 2. (Optional) create a virtual environment
python -m venv .venv
.venv\Scripts\activate        # Windows
# source .venv/bin/activate   # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the notebook
jupyter notebook
```

You can also open the `.ipynb` files directly in VS Code with the Jupyter extension and choose **Run All**.

## Architecture

Every notebook uses the same pipeline:

```
Raw data (CSV / sklearn dataset)
        |
        v
Exploration (pairplot, summary stats)
        |
        v
Stratified train/test split (80/20, random_state=42)
        |
        v
StandardScaler (fit on TRAIN only -> no data leakage)
        |
        v
Model training (Logistic Regression / Decision Tree / ...)
        |
        v
Evaluation
  - Accuracy, macro precision, recall, F1
  - Confusion matrix
  - 5-fold stratified cross-validation (scaler inside a Pipeline)
  - Decision boundary visualisation
        |
        v
Conclusions
```

Design choices:
- **Stratified split** keeps class proportions the same in train and test.
- **Scaler fitted on train only** prevents information from the test set leaking into training.
- **Pipeline in cross-validation** re-fits the scaler inside each fold for an honest estimate.
- **Fixed random_state** makes every result reproducible.

## Repository structure

```
datheon-ml-projects/
├── task1_iris_classification.ipynb
├── requirements.txt
└── README.md
```

## Author

Built by Sudharsan as part of the Datheon internship.
