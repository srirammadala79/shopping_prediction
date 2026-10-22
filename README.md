# Machine Learning Capstone Project: Online Shopping Clickstream Prediction & Customer Segmentation

**Course:** 23CSE301 Machine Learning  
**Degree:** B.Tech. Computer Science and Engineering (III Year) — Academic Year 2026-27  
**Team:** Team 16  
**Tracks Covered:** 
1. **Regression Track (Review 1):** Continuous Product Price Prediction (`Price`)
2. **Classification Track (Review 2):** 4-Class Product Navigation Category Prediction (`Main_Category`)
3. **Clustering Track (Review 2):** Unsupervised Customer Behavioral Segmentation & Persona Profiling

---

## 1. Repository Directory Structure

This repository strictly conforms to the required directory structure:

```text
/ (root)
│
├── README.md                          # Comprehensive project overview, results, & setup instructions
├── requirements.txt                   # All Python dependencies with explicit versions
│
├── data/                              # Raw dataset file(s) and data verification script
│   ├── shopping2.csv                  # Raw clickstream dataset (165,474 interaction records)
│   └── download_data.py               # Dataset verification & integrity inspection utility
│
├── notebooks/                         # Core Jupyter Notebooks for all ML tracks
│   ├── regression.ipynb               # Review 1: 10 Regression algorithms, tuning, & residual analysis
│   ├── classification.ipynb           # Review 2: Session-grouped 10-model benchmark, tuning, & ROC curves
│   └── clustering.ipynb               # Review 2: K-Means & Hierarchical clustering, PCA/t-SNE, personas
│
├── models/                            # Serialized Joblib (.pkl) trained model artifacts
│   ├── best_regressor.pkl             # Champion Random Forest Regressor
│   ├── reg_preprocessor.pkl           # ColumnTransformer preprocessing pipeline for regression
│   ├── best_classifier.pkl            # Champion Tuned Bagging Classifier (with Decision Trees)
│   ├── best_classifier_rf.pkl         # Backward-compatible alias for champion classifier
│   ├── preprocessor.pkl               # Leakage-free ColumnTransformer for classification
│   ├── label_encoder.pkl              # Fitted LabelEncoder for category targets
│   ├── kmeans_model.pkl               # Fitted K-Means Clustering model (k=4)
│   └── clust_preprocessor.pkl         # StandardScaler & Median Imputer pipeline for clustering
│
├── figures/                           # Visualizations, diagnostic plots, and benchmark tables
│   ├── regression_model_comparison.csv
│   ├── classification_10_models_comparison.csv
│   ├── baseline_vs_tuned_champion_comparison.csv
│   ├── clustering_evaluation_metrics.csv
│   └── cluster_profiles_summary.csv
│
└── app/                               # Interactive Streamlit Web Application
    └── app.py                         # Live multi-track prediction and segmentation interface
```

---

## 2. Project Overview & Problem Statement

E-commerce platforms capture fine-grained clickstream trajectories as users browse product listings. This project harnesses clickstream data (`shopping2.csv`) spanning **165,474 interactions** across diverse customer sessions to tackle three interrelated machine learning problems:

1. **Regression (Continuous Price Prediction):**  
   Predict the monetary value (`Price`) of products viewed by users given contextual click attributes (page order, location code, photography style, session depth, and time of week) to dynamically evaluate user price sensitivity.
2. **Classification (Customer Intent & Category Navigation):**  
   Predict the target multi-class product department (`Main_Category`: *Trousers*, *Skirts*, *Blouses*, or *Sale*) that a shopper navigates to. To ensure real-world validity and zero data leakage, sessions are split as grouped units (`StratifiedGroupKFold`) so interactions from the same session never cross train/test boundaries.
3. **Clustering (Unsupervised Customer Segmentation):**  
   Group customer sessions into distinct behavioral segments without supervision or label guidance, identifying actionable business personas to enable personalized marketing strategies.

---

## 3. Results Summary & Model Benchmarks

### Track 1: Regression Track Results (Review 1)

All 10 regression algorithms were evaluated on the held-out test split using $R^2$, Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE):

| Rank | Algorithm | $R^2$ Score (↑) | RMSE (↓) | MAE (↓) | Status |
|:---:|:---|:---:|:---:|:---:|:---|
| 1 | **Random Forest Regressor** | **0.6335** | **7.19** | **4.20** | **Champion Model** |
| 2 | **Decision Tree Regressor** | 0.5855 | 7.65 | 4.97 | Runner-Up |
| 3 | **Gradient Boosting Regressor** | 0.5097 | 8.32 | 6.23 | Ensemble Baseline |
| 4 | **K-Nearest Neighbors (KNN)** | 0.2340 | 10.40 | 7.67 | Non-Parametric |
| 5 | **Support Vector Regressor (SVR)** | 0.1310 | 11.07 | 7.90 | Kernel RBF |
| 6 | **Polynomial Regression (Degree 2)** | 0.1192 | 11.15 | 9.13 | Non-linear Expansion |
| 7 | **ElasticNet Regression** | 0.0510 | 11.57 | 9.47 | Regularized Linear |
| 8 | **Lasso Regression (L1)** | 0.0509 | 11.57 | 9.46 | Feature Selection |
| 9 | **Ridge Regression (L2)** | 0.0509 | 11.57 | 9.46 | L2 Regularized |
| 10 | **Linear Regression (OLS)** | 0.0509 | 11.57 | 9.46 | Baseline Linear |

- **Hyperparameter Tuning:** 5-Fold Cross Validation on Random Forest optimized `n_estimators=100`, `max_depth=12`, reducing test RMSE to 7.19.
- **Diagnostics:** Residual distributions demonstrated zero-centered normality, confirming homoscedastic variance.

---

### Track 2: Classification Track Results (Review 2 — Section A)

#### 10-Algorithm Consolidated Leaderboard (Ranked by Weighted F1-Score)

| Rank | Algorithm | Track Part | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 (Primary) | ROC-AUC (OvR) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Bagging Classifier** | Part B | 0.8797 | 0.8819 | 0.8797 | **0.8802** | **0.9859** |
| 2 | **Random Forest Classifier** | Part B | 0.8719 | 0.8732 | 0.8719 | 0.8720 | 0.9785 |
| 3 | **MLP Classifier (Neural Net)** | Part B | 0.8672 | 0.8682 | 0.8672 | 0.8675 | 0.9805 |
| 4 | **Gradient Boosting** | Part B | 0.8597 | 0.8609 | 0.8597 | 0.8598 | 0.9761 |
| 5 | **Decision Tree Classifier** | Part A | 0.7118 | 0.7370 | 0.7118 | 0.7086 | 0.9241 |
| 6 | **K-Nearest Neighbors** | Part A | 0.6362 | 0.6363 | 0.6362 | 0.6350 | 0.8405 |
| 7 | **Support Vector Machine (SVC)**| Part A | 0.5892 | 0.5943 | 0.5892 | 0.5752 | 0.8180 |
| 8 | **Naive Bayes (Gaussian)** | Part A | 0.4594 | 0.4489 | 0.4594 | 0.4409 | 0.7218 |
| 9 | **Logistic Regression** | Part A | 0.4422 | 0.4199 | 0.4422 | 0.4173 | 0.7420 |
| 10 | **AdaBoost Classifier** | Part B | 0.4422 | 0.4190 | 0.4422 | 0.4123 | 0.7928 |

#### Champion Model Selection & Justification (Rubric A3)
- **Champion Selected:** **Bagging Classifier** (utilizing parallel deep `DecisionTreeClassifier` estimators).
- **Justification over Random Forest:** Bagging preserves full feature subspaces across bootstrap samples, achieving higher ensemble diversity and outperforming Random Forest on both test F1-score (+0.82 pp) and ROC-AUC (+0.74 pp).
- **Primary Metric Justification:** **Weighted F1-Score** accounts for multi-class category imbalance across product departments, avoiding raw accuracy distortion.

#### GridSearchCV Hyperparameter Tuning Performance

Optimal Hyperparameters: `n_estimators=50`, `estimator__max_depth=12`, `estimator__min_samples_split=2`, `max_samples=0.85` via 3-Fold session-grouped CV.

| Metric | Baseline Bagging | Tuned Bagging | Absolute Change (pp) | Relative Improvement (%) |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 0.8797 | **0.8869** | **+0.72 pp** | **+0.82%** |
| **Weighted Precision** | 0.8819 | **0.8871** | **+0.52 pp** | **+0.59%** |
| **Weighted Recall** | 0.8797 | **0.8869** | **+0.72 pp** | **+0.82%** |
| **Weighted F1-Score** | 0.8802 | **0.8869** | **+0.67 pp** | **+0.76%** |
| **ROC-AUC (OvR)** | 0.9859 | **0.9883** | **+0.24 pp** | **+0.24%** |

---

### Track 3: Clustering Track Results (Review 2 — Section B)

Both **K-Means** and **Agglomerative Hierarchical Clustering** were fitted on standardized clickstream attributes (`Page_Order`, `Location_Code`, `Photography_Type`, `session_total_clicks`, `click_progress`, `is_weekend`) at optimal $k=4$:

| Algorithm | Clusters ($k$) | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) | Recommended Use Case |
|:---|:---:|:---:|:---:|:---:|:---|
| **K-Means Clustering** | **4** | **0.2981** | **1.2294** | **3156.9** | **Production & Live User Scoring** |
| **Agglomerative Hierarchical** | 4 | 0.2858 | 1.2562 | 2919.4 | Exploratory Taxonomy & Tree Analysis |

#### Discovered Customer Personas:
1. **Cluster 0 — Quick Bouncers:** Single-item visitors (`session_total_clicks` $\approx 1.8$, low page depth). Strategy: High-impact homepage hooks and instant discount modals.
2. **Cluster 1 — Weekend Explorers:** High weekend activity (`is_weekend` = 1.0) with varied photography exploration. Strategy: Weekend flash sales and bundle recommendations.
3. **Cluster 2 — Deeply Engaged High-Intent Shoppers:** Long browsing journeys (`session_total_clicks` > 15, high `Page_Order`). Strategy: Cart abandonment reminders and loyalty promotions.
4. **Cluster 3 — Category-Specific Focused Buyers:** Direct navigators focused on specific catalog coordinates. Strategy: Targeted cross-selling within category.

---

## 4. Setup & Installation Instructions

Follow these steps to configure your Python environment and run the project locally.

### Step 1: Clone the Repository & Open Directory
```bash
git clone https://github.com/srirammadala79/shopping_prediction.git
cd team16_ml_casestudy
```

### Step 2: Create and Activate Virtual Environment
```bash
# Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Required Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 4: Verify the Dataset
Ensure the raw dataset is properly placed and validated:
```bash
python data/download_data.py
```

### Step 5: Run the Jupyter Notebooks
Open VS Code, JupyterLab, or start Jupyter Notebook:
```bash
jupyter notebook
```
Navigate to `notebooks/` and run:
- `notebooks/regression.ipynb`
- `notebooks/classification.ipynb`
- `notebooks/clustering.ipynb`

### Step 6: Launch the Interactive Web Application (Bonus)
Launch the Streamlit user interface to interact with the trained models and visualize live predictions:
```bash
streamlit run app/app.py
```

---

## 5. Team 16 Members & Contributions
- **Team:** Team 16 (B.Tech. CSE, 3rd Year)
- **Course:** 23CSE301 Machine Learning
- **Institution:** Amrita Vishwa Vidyapeetham
