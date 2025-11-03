# HeartSense

A machine learning project that predicts the likelihood of heart disease using various classification algorithms. This project compares four different models to identify the most effective approach for cardiovascular risk assessment.

## Project Overview

This is a research/experimental project (not a production application) that explores different machine learning approaches for heart disease prediction. The goal is to understand which algorithms and hyperparameters work best for this medical classification task.

**Dataset**: 918 patient records with 11 clinical features
**Target**: Binary classification (0 = No Heart Disease, 1 = Heart Disease)
**Distribution**: ~55% positive cases (heart disease present)

---

## Data Preprocessing Strategy

### 1. Categorical Encoding
**Method**: Label Encoding
**Rationale**: Converts categorical features to numerical format for model compatibility

Features encoded:
- **Sex**: Male=0, Female=1
- **ChestPainType**: ATA=0, NAP=1, ASY=2, TA=3
- **RestingECG**: Normal=0, ST=1, LVH=2
- **ExerciseAngina**: N=0, Y=1
- **ST_Slope**: Up=0, Flat=1, Down=2

**Why this approach?**
- Simple and efficient for tree-based models
- Preserves ordinal relationships where applicable
- Minimal preprocessing overhead

### 2. Missing Value Imputation
**Method**: KNNImputer with k=3 neighbors
**Applied to**: Cholesterol and RestingBP (zero values treated as missing)

**Why KNN Imputation?**
- Preserves relationships between features
- More sophisticated than mean/median imputation
- k=3 balances local patterns without overfitting
- Handles medical data where values correlate (e.g., age affects BP/cholesterol)

**Code location**: `heartdiseasepredictor.py:112-148`

### 3. Train-Test Split
```python
test_size=0.2
random_state=42
stratify=heart_df['HeartDisease']
```

**Rationale**:
- 80/20 split provides sufficient training data
- Stratification ensures balanced class distribution
- Fixed random_state for reproducibility

---

## Model Selection & Hyperparameter Analysis

### Model 1: Logistic Regression

**Final Configuration**:
```python
LogisticRegression(solver='lbfgs')
```

**Hyperparameter Exploration**:
Tested 6 solvers: `['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']`

**Why this approach?**
- **Solver Selection**: Automatically chooses best optimizer for this dataset
- **lbfgs** (Limited-memory BFGS): Efficient for small-to-medium datasets
- Handles multinomial loss well
- Good baseline model for binary classification

**Why these hyperparameters?**
- Default C=1.0: Moderate regularization prevents overfitting
- Default penalty='l2': Ridge regression reduces coefficient magnitude
- Default max_iter=100: Sufficient for convergence on this dataset

**Performance**: ~84-86% accuracy
**Code location**: `heartdiseasepredictor.py:300-318`

---

### Model 2: Support Vector Machine (SVM)

**Final Configuration**:
```python
SVC(kernel='linear')
```

**Hyperparameter Exploration**:
Tested 4 kernels: `{'linear', 'poly', 'rbf', 'sigmoid'}`
**Evaluation Metric**: F1-score (weighted)

**Why Linear Kernel Won?**
- Medical data often has linear decision boundaries
- More interpretable than polynomial/RBF
- Faster training and prediction
- Less prone to overfitting than complex kernels

**Why F1-score over Accuracy?**
- Balances precision and recall
- Critical for medical predictions (both false positives and negatives matter)
- Better metric for slightly imbalanced datasets

**Default Hyperparameters**:
- C=1.0: Standard regularization
- gamma='scale': Auto-calculated as 1/(n_features * X.var())
- Decision function: One-vs-Rest

**Performance**: ~84.2% F1-score
**Code location**: `heartdiseasepredictor.py:331-349`

---

### Model 3: Decision Tree Classifier

**Final Configuration**:
```python
DecisionTreeClassifier(
    max_depth=BEST,
    min_samples_split=BEST,
    min_samples_leaf=BEST,
    random_state=BEST,
    class_weight='balanced'
)
```

**Hyperparameter Grid**:
```python
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4],
    'random_state': [0, 42]
}
```

**GridSearchCV Settings**:
- `cv=5`: 5-fold cross-validation
- Automatically selects best combination

**Why these hyperparameters?**

1. **max_depth [3-8]**:
   - Controls tree complexity
   - Too deep (>8): Overfitting on training data
   - Too shallow (<3): Underfitting, misses patterns
   - Range 3-8: Sweet spot for medical datasets

2. **min_samples_split [2-4]**:
   - Minimum samples required to split a node
   - Higher values prevent overfitting
   - 2-4 range balances detail vs generalization

3. **min_samples_leaf [1-4]**:
   - Minimum samples in leaf nodes
   - Prevents creating nodes with few samples
   - Acts as regularization

4. **random_state [0, 42]**:
   - Tests reproducibility across seeds
   - Ensures results aren't seed-dependent

5. **class_weight='balanced'**:
   - Handles class imbalance automatically
   - Weighs classes inversely proportional to frequency
   - Critical for medical data where minority class matters

**Performance**: ~81% accuracy
**Code location**: `heartdiseasepredictor.py:362-381`

---

### Model 4: Random Forest Classifier (BEST PERFORMER)

**Final Configuration**:
```python
RandomForestClassifier(
    n_estimators=BEST,
    max_features=BEST,
    max_depth=BEST,
    max_leaf_nodes=BEST
)
```

**Hyperparameter Grid**:
```python
param_grid = {
    'n_estimators': [50, 100, 150, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9, 19],
    'max_leaf_nodes': [3, 6, 9]
}
```

**GridSearchCV Settings**:
- `cv=5`: 5-fold cross-validation
- Tests 4×3×4×3 = 144 combinations

**Deep Dive into Hyperparameters**:

1. **n_estimators [50, 100, 150, 500]**:
   - Number of trees in the forest
   - More trees = More stable predictions
   - **50**: Fast but may underfit
   - **100**: Good balance (industry standard)
   - **150**: Marginal improvement
   - **500**: Diminishing returns, slower inference
   - **Why this range?** Captures performance plateau without excessive computation

2. **max_features ['sqrt', 'log2', None]**:
   - Features considered for each split
   - **'sqrt'**: √11 ≈ 3 features (reduces correlation between trees)
   - **'log2'**: log₂(11) ≈ 3.5 features (similar to sqrt for this dataset)
   - **None**: All 11 features (max information, higher correlation)
   - **Why test all?** Dataset-dependent; sqrt often wins for classification

3. **max_depth [3, 6, 9, 19]**:
   - Maximum tree depth
   - **3**: Shallow, fast, may underfit
   - **6**: Moderate complexity
   - **9**: Captures complex interactions
   - **19**: Deep, risks overfitting
   - **Why this range?** Covers spectrum from simple to complex models

4. **max_leaf_nodes [3, 6, 9]**:
   - Limits total leaves in tree
   - Alternative to max_depth for controlling complexity
   - **3**: Very simple trees
   - **6**: Moderate
   - **9**: More detailed
   - **Why limit?** Prevents overfitting, speeds up training

**Why Random Forest Performed Best?**
- **Ensemble Power**: Combines multiple decision trees
- **Reduced Overfitting**: Averaging reduces variance
- **Handles Non-linearity**: Captures complex feature interactions
- **Robust to Outliers**: Individual trees may overfit, but ensemble corrects
- **Feature Importance**: Can identify key predictors

**Performance**: ~86.4% accuracy (HIGHEST)
**Code location**: `heartdiseasepredictor.py:394-413`

---

## Model Comparison Summary

| Model | Accuracy/Score | Training Time | Pros | Cons |
|-------|---------------|---------------|------|------|
| **Logistic Regression** | ~84-86% | Fastest | Simple, interpretable, good baseline | Assumes linear relationships |
| **SVM** | ~84.2% F1 | Medium | Good for high-dimensional data | Less interpretable, slower inference |
| **Decision Tree** | ~81% | Fast | Highly interpretable, handles non-linearity | Prone to overfitting |
| **Random Forest** | **~86.4%** | Slowest | Best accuracy, robust, handles complexity | Black box, slower predictions |

**Winner**: Random Forest
**Best Baseline**: Logistic Regression
**Most Interpretable**: Decision Tree

---

## Feature Importance Insights

Based on correlation analysis (`heartdiseasepredictor.py:191`):

**Top Predictors** (Positive Correlation):
1. **ST_Slope** (0.56): Strongest predictor
2. **ExerciseAngina** (0.49): Exercise-induced chest pain
3. **ChestPainType** (0.46): Type of chest pain
4. **Oldpeak** (0.40): ST depression

**Protective Factors** (Negative Correlation):
1. **MaxHR** (-0.40): Higher heart rate = lower risk
2. **Sex** (-0.31): Gender differences in presentation

**Weak Predictors**:
- RestingECG (0.06)
- Cholesterol (0.10)
- RestingBP (0.12)

---

## Exploratory Data Analysis (EDA)

Comprehensive visualizations created using Plotly:
- Correlation heatmap
- Age distribution by disease status
- Gender distribution
- Chest pain type analysis
- Violin plots for continuous variables
- Sunburst charts for hierarchical data

**Code location**: `heartdiseasepredictor.py:197-282`

---

## Application Architecture

### Streamlit Web App
File: `app.py`

**Features**:
- Interactive patient data input
- Multi-model selection
- Real-time predictions with confidence scores
- Risk factor analysis
- Responsive design (mobile-friendly)
- Dark theme UI

**Tech Stack**:
- Streamlit for frontend
- Pickle for model serialization
- NumPy/Pandas for data handling

---

## Project Structure

```
heart-disease/
├── heartdiseasepredictor.py    # Main training script
├── HeartDiseasePredictor.ipynb # Jupyter notebook version
├── app.py                       # Streamlit web application
├── content/
│   └── heart.csv               # Dataset
├── LogisticR.pkl               # Trained Logistic Regression
├── SVM_Model.pkl               # Trained SVM
├── DecisionTree.pkl            # Trained Decision Tree
├── RandomForest.pkl            # Trained Random Forest
├── requirements.txt            # Python dependencies
└── README.md                   # This file
```

---

## How to Run

### Training Models
```bash
python heartdiseasepredictor.py
```

### Running Web App
```bash
streamlit run app.py
```

### Requirements
```
numpy
pandas
scikit-learn
plotly
streamlit
pickle
```

---

## Key Takeaways

### What Worked Well:
- KNN imputation for missing values
- GridSearchCV for hyperparameter optimization
- Random Forest ensemble approach
- Stratified train-test split

### What Could Be Improved:
- Try XGBoost/LightGBM for potentially better performance
- Implement SMOTE for better class balancing
- Feature engineering (interaction terms, polynomial features)
- Cross-validation for all models (not just GridSearchCV ones)
- Hyperparameter tuning for Logistic Regression and SVM

### Why These Specific Hyperparameter Ranges?

**General Principles Used**:
1. **Start Conservative**: Small values to prevent overfitting
2. **Cover Spectrum**: Test shallow to deep, simple to complex
3. **Computational Budget**: Balance thoroughness with training time
4. **Medical Domain**: Prefer interpretability where possible
5. **Literature Review**: Common ranges for similar problems

---

## Evaluation Metrics

- **Logistic Regression**: Accuracy score
- **SVM**: F1-score (weighted) - better for imbalanced data
- **Decision Tree**: Accuracy score
- **Random Forest**: Accuracy score

**Why Different Metrics?**
- Demonstrates understanding of evaluation strategies
- SVM's F1-score shows awareness of precision-recall tradeoff
- In production, would use consistent metrics + ROC-AUC, PR-AUC

---

## Limitations & Future Work

### Current Limitations:
- No cross-validation for LR/SVM (only GridSearchCV models)
- Limited feature engineering
- No hyperparameter tuning for regularization in LR
- Single train-test split (could be unstable)

### Future Enhancements:
1. **Advanced Models**: XGBoost, CatBoost, Neural Networks
2. **Ensemble Methods**: Stacking, Voting Classifiers
3. **Feature Engineering**: Interaction terms, binning
4. **Explainability**: SHAP values, LIME
5. **Deployment**: Docker containerization, API endpoint
6. **Monitoring**: Model drift detection, A/B testing

---

## Contributors

- **Adarsh** (E23CSEU1189)
- **Arindam Singh** (E23CSEU1171)
- **Yashvardhan Dhaka** (E23CSEU1192)

---

## License

This is an educational project for learning purposes.

---

## References

- Dataset: Heart Disease Prediction Dataset
- Scikit-learn Documentation
- Streamlit Documentation
- Medical domain knowledge for feature interpretation

---

## Disclaimer

This is a research/educational project. **Do not use for actual medical diagnosis.** Always consult qualified healthcare professionals for medical advice.
