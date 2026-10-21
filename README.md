# Machine Learning Capstone Project — Review 2
## Online Shopping Clickstream Prediction & Customer Segmentation

**Course:** 23CSE301 Machine Learning  
**Degree:** B.Tech. Computer Science and Engineering (III Year) — Academic Year 2026-27  
**Team:** Team 16  
**Review Scope:** Classification Track (Part B + Full 10-Model Benchmark) & Full Clustering Track  

---

## 1. Problem Statement & Dataset Description

This capstone project addresses an end-to-end Machine Learning problem analyzing online shopping clickstream data (`shopping2.csv`). 

- **Classification Track:** Predict customer interest and product navigation category (`Main_Category`, 4 distinct multi-class categories) based on browsing behaviors (page sequence order, photography type, location on page, color, country, session depth, and weekday/weekend activity).
- **Clustering Track:** Perform unsupervised customer segmentation to identify natural shopping personas (e.g., Quick Bouncers, Weekend Browsers, Deep-Intent Shoppers) without using training labels.

---

## 2. Review 2 Deliverables & Rubric Fulfillment

### Section A — Classification Track (6 Marks)
- **A1 Implementation:** Successfully trained all 5 Part-B algorithms on the exact held-out test split:
  1. Random Forest Classifier
  2. AdaBoost Classifier
  3. Gradient Boosting Classifier
  4. Bagging Classifier (with `DecisionTreeClassifier` base estimator)
  5. Multi-Layer Perceptron (MLP) Classifier
- **A2 Consolidated Comparison Table:** Benchmarked all 10 algorithms (Part A + Part B) across Accuracy, Precision, Recall, F1-Score (weighted), and ROC-AUC (OvR) ranked primarily by Weighted F1-Score.
- **A3 Model Selection & Tuning:**
  - Evaluated the 10-model baseline leaderboard and identified the actual best-performing baseline model (**Bagging Classifier**), avoiding unfounded defaults to Random Forest.
  - Justified **Weighted F1-score** as the primary selection metric due to multi-class imbalance across the 4 navigation categories, with Accuracy and ROC-AUC (OvR) as supporting metrics.
  - Conducted `GridSearchCV` hyperparameter tuning with 3-fold session-grouped cross-validation to prevent intra-session data leakage.
  - Documented quantitative improvements comparing Baseline vs. Tuned performance across all 5 evaluation metrics, clearly distinguishing absolute percentage-point changes from relative percentage changes.
  - Generated diagnostic visualisations (Confusion Matrix, Multi-Class OvR ROC curves, and Ensemble Feature Importances).

### Section B — Clustering Track (8 Marks)
- **B1 Implementation:** Fitted both **K-Means Clustering** and **Agglomerative Hierarchical Clustering** on standardized clickstream attributes without label guidance.
- **B2 Evaluation Metrics:**
  - Evaluated on **Silhouette Score**, **Davies-Bouldin Index**, and **Calinski-Harabasz Index**.
  - Generated **Elbow Curve** (Inertia vs. $k$) showing optimal inflection at $k=4$.
  - Generated **Dendrogram** using Ward's minimum variance linkage method.
- **B3 Cluster Visualisation:**
  - Mandatory **PCA 2D scatter plots** showing cluster separation for both K-Means and Hierarchical Clustering.
  - **t-SNE 2D manifold projection** validating non-linear cluster separation.
  - Extracted 4 actionable customer personas (Quick Bouncers, Weekend Explorers, Deeply Engaged Shoppers, Category-Specific Browsers).

### Section C — Pipeline Quality & Structure (3 Marks)
- **C1 Code Quality & Zero Data Leakage:**
  - **Session-Grouped Splitting (`StratifiedGroupKFold`):** Customer click journeys are grouped by `Session_ID` before any transformations so that clicks from the same session never cross training and test boundaries.
  - **Encapsulated Preprocessing:** Numerical median imputation (`SimpleImputer`) and standardization (`StandardScaler`) are encapsulated in `Pipeline` and `ColumnTransformer`, fitted strictly on training data only.
  - Reusable modular evaluation functions, reproducible random seeds (`RANDOM_STATE = 42`), clean top-to-bottom error-free execution, and saved model artifacts for deployment.

### Bonus Opportunities (+2 Marks)
- Interactive web application created in [`app/app.py`](app/app.py) using Streamlit for live customer category prediction and cluster exploration.

---

## 3. Results Summary Leaderboard

### Classification 10-Algorithm Consolidated Benchmark (Rubric A2)

Ranked primarily by **Weighted F1-Score**:

| Rank | Algorithm | Track Part | Accuracy | Weighted Precision | Weighted Recall | Weighted F1 | ROC-AUC (OvR) |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| 1 | **Bagging Classifier** | Part B | 0.8797 | 0.8819 | 0.8797 | **0.8802** | **0.9859** |
| 2 | **Random Forest Classifier** | Part B | 0.8719 | 0.8732 | 0.8719 | 0.8720 | 0.9785 |
| 3 | **MLP Classifier (Neural Net)** | Part B | 0.8672 | 0.8682 | 0.8672 | 0.8675 | 0.9805 |
| 4 | **Gradient Boosting** | Part B | 0.8597 | 0.8609 | 0.8597 | 0.8598 | 0.9761 |
| 5 | **Decision Tree Classifier** | Part A | 0.7118 | 0.7370 | 0.7118 | 0.7086 | 0.9241 |
| 6 | **K-Nearest Neighbors** | Part A | 0.6362 | 0.6363 | 0.6362 | 0.6350 | 0.8405 |
| 7 | **Support Vector Machine (SVC)** | Part A | 0.5892 | 0.5943 | 0.5892 | 0.5752 | 0.8180 |
| 8 | **Naive Bayes (Gaussian)** | Part A | 0.4594 | 0.4489 | 0.4594 | 0.4409 | 0.7218 |
| 9 | **Logistic Regression** | Part A | 0.4422 | 0.4199 | 0.4422 | 0.4173 | 0.7420 |
| 10 | **AdaBoost Classifier** | Part B | 0.4422 | 0.4190 | 0.4422 | 0.4123 | 0.7928 |

### Champion Model Selection & Hyperparameter Tuning Improvement (Rubric A3)

- **Selected Champion Architecture:** **Bagging Classifier** (with `DecisionTreeClassifier` base estimators).
- **Optimal Hyperparameters (via GridSearchCV with 3-Fold Group CV):**
  - `n_estimators`: `50`
  - `estimator__max_depth`: `12`
  - `estimator__min_samples_split`: `2`
  - `max_samples`: `0.85`

#### Baseline vs. Tuned Performance on Held-Out Test Set:

| Metric | Baseline Bagging | Tuned Bagging | Absolute Change (pp) | Relative Improvement (%) |
|:---|:---:|:---:|:---:|:---:|
| **Accuracy** | 0.8797 | **0.8869** | **+0.72 pp** | **+0.82%** |
| **Weighted Precision** | 0.8819 | **0.8871** | **+0.52 pp** | **+0.59%** |
| **Weighted Recall** | 0.8797 | **0.8869** | **+0.72 pp** | **+0.82%** |
| **Weighted F1-Score** | 0.8802 | **0.8869** | **+0.67 pp** | **+0.76%** |
| **ROC-AUC (OvR)** | 0.9859 | **0.9883** | **+0.24 pp** | **+0.24%** |

*Note: Absolute Change (pp) is the percentage-point difference: $(\text{Tuned} - \text{Baseline}) \times 100$. Relative Improvement (%) is the percentage gain relative to baseline: $\frac{\text{Tuned} - \text{Baseline}}{\text{Baseline}} \times 100\%$.*

### Clustering Track Benchmark (Rubric B2)

| Clustering Algorithm | Number of Clusters ($k$) | Silhouette Score (↑) | Davies-Bouldin Index (↓) | Calinski-Harabasz Index (↑) |
|:---|:---:|:---:|:---:|:---:|
| **K-Means Clustering** | **4** | **0.3276** | **1.1814** | **2887.10** |
| **Agglomerative Hierarchical** | **4** | **0.3163** | **1.1758** | **2608.21** |

---

## 4. Repository Structure

```text
team16_ml_casestudy/
├── app/
│   └── app.py                                  # Streamlit Interactive Web Application
├── figures/                                    # Generated Evaluation Plots & CSVs
│   ├── baseline_vs_tuned_champion_comparison.csv
│   ├── classification_10_models_benchmark.png
│   ├── classification_10_models_comparison.csv
│   ├── clustering_evaluation_metrics.csv
│   ├── cluster_profiles_summary.csv
│   ├── feature_importances_champion.png
│   ├── feature_importances_rf.png
│   ├── hierarchical_dendrogram.png
│   ├── kmeans_elbow_curve.png
│   ├── pca_clusters_comparison.png
│   ├── tsne_clusters_kmeans.png
│   ├── tuned_champion_confusion_matrix.png
│   ├── tuned_champion_ovr_roc_curves.png
│   ├── tuned_rf_confusion_matrix.png
│   └── tuned_rf_ovr_roc_curves.png
├── models/                                     # Saved Model Artifacts (.pkl)
│   ├── best_classifier.pkl                     # Tuned Champion Model (Bagging)
│   ├── best_classifier_rf.pkl                  # Compatibility model artifact
│   ├── label_encoder.pkl                       # Target class encoder
│   └── preprocessor.pkl                        # Pipeline (Imputer + StandardScaler)
├── notebooks/ (or root)
│   ├── online_shopping_prediction.ipynb        # Review 1 Notebook (Regression + Classification Part A)
│   └── review2_classification_clustering.ipynb # Review 2 Notebook (Classification Part B + Clustering)
├── shopping2.csv                               # Cleaned Raw Dataset
├── requirements.txt                            # Pinned Dependencies
└── README.md                                   # Project Documentation & Viva Guide
```

---

## 5. How to Setup and Run

### 1. Environment Setup
```bash
git clone <repository_url>
cd team16_ml_casestudy
pip install -r requirements.txt
```

### 2. Run Review 2 Notebook
Launch Jupyter Notebook or Jupyter Lab:
```bash
jupyter notebook review2_classification_clustering.ipynb
```
All cells run cleanly top-to-bottom.

### 3. Run the Interactive GUI (Local Streamlit)
```bash
python -m streamlit run app/app.py
```
This opens the web interface in your browser (`http://localhost:8501`) with live prediction sliders and cluster charts.

### 4. Containerized Deployment (Docker)
Build and run using Docker Compose:
```bash
docker compose up --build
```
Or build and launch via standard Docker:
```bash
docker build -t clickstream-ml-app .
docker run -p 8501:8501 clickstream-ml-app
```
Access the application at `http://localhost:8501`.

### 5. Vercel Public Cloud Deployment (+2 Bonus Marks)
The self-contained Vercel deployment bundle is ready inside [`vercel_deployment/`](vercel_deployment/):
- **Live GUI:** Accepts user inputs and returns multi-track predictions across **Review 1** (Price Regression) and **Review 2** (Category Classification & Customer Segmentation).
- **Serverless ML Backend:** Powered by Python serverless functions in `vercel_deployment/api/index.py`.

To deploy publicly via Vercel CLI:
```bash
cd vercel_deployment
npx vercel
```
Or connect your GitHub repository to [Vercel](https://vercel.com) and set the root directory to `vercel_deployment` for automatic continuous deployment.

---

## 6. Academic Integrity & AI Assistance Citation
Per Section 7.5 of the Capstone Project Guidelines:
- Feature engineering, domain modeling, and statistical analysis logic were custom-designed for this clickstream dataset.
- Generative AI assistance was utilized for code scaffolding, plot styling, and notebook structure structuring.
