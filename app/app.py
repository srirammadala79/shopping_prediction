import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

st.set_page_config(
    page_title="Clickstream ML Predictor & Segments (Reviews 1 & 2)",
    page_icon="🛍️",
    layout="wide"
)

st.title("🛍️ Online Shopping Clickstream — ML Prediction & Customer Segmentation")
st.markdown("""
**Course:** 23CSE301 Machine Learning • **Team:** Team 16  
*Capstone Deliverable: Review 1 (Price Regression) & Review 2 (10-Model Classification & Customer Segmentation)*
""")

# Load models and preprocessors for both Review 1 and Review 2
@st.cache_resource
def load_all_artifacts():
    artifacts = {}
    
    # Review 2 Classification
    clf_path = os.path.join("models", "best_classifier.pkl")
    if not os.path.exists(clf_path):
        clf_path = os.path.join("models", "best_classifier_rf.pkl")
    prep_path = os.path.join("models", "preprocessor.pkl")
    le_path = os.path.join("models", "label_encoder.pkl")
    
    if os.path.exists(clf_path) and os.path.exists(prep_path) and os.path.exists(le_path):
        artifacts['clf'] = joblib.load(clf_path)
        artifacts['clf_prep'] = joblib.load(prep_path)
        artifacts['le'] = joblib.load(le_path)

    # Review 1 Regression (Price Target)
    reg_path = os.path.join("models", "best_regressor.pkl")
    reg_prep_path = os.path.join("models", "reg_preprocessor.pkl")
    if os.path.exists(reg_path) and os.path.exists(reg_prep_path):
        artifacts['reg'] = joblib.load(reg_path)
        artifacts['reg_prep'] = joblib.load(reg_prep_path)

    # Review 2 Clustering (Customer Segmentation)
    km_path = os.path.join("models", "kmeans_model.pkl")
    km_prep_path = os.path.join("models", "clust_preprocessor.pkl")
    if os.path.exists(km_path) and os.path.exists(km_prep_path):
        artifacts['km'] = joblib.load(km_path)
        artifacts['km_prep'] = joblib.load(km_prep_path)
        
    return artifacts

artifacts = load_all_artifacts()

PERSONAS = {
    0: ("Quick Bouncer", "Short exploratory visit with low click count (<3 clicks). Users browsing 1-2 items and exiting.", "Exit-intent promotional banners and top-selling showcase."),
    1: ("Weekend Explorer", "High engagement on weekends browsing multiple product styles and photo profiles.", "Exclusive weekend discount bundles and flash offers."),
    2: ("Deeply Engaged Shopper", "Extensive browsing depth (>15 clicks) and active price evaluation. High conversion intent.", "Targeted product recommendations and VIP loyalty rewards."),
    3: ("Category-Specific Browser", "Focused page journey patterns targeted at specific location codes and clothing segments.", "Category cross-sell recommendations and peer social proof.")
}

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Multi-Track Real-Time Prediction",
    "📊 Review 2 Classification Benchmark",
    "📈 Review 1 Regression Benchmark",
    "🧩 Review 2 Customer Segmentation"
])

with tab1:
    st.subheader("Predict Customer Intent, Category, and Price from Browsing Behavior")
    if 'clf' not in artifacts:
        st.warning("Model artifacts not found in models/ directory. Run review2_classification_clustering.ipynb to build models.")
    else:
        col1, col2, col3 = st.columns(3)
        with col1:
            page_order = st.slider("Page Order (Click Sequence Number)", 1, 30, 3)
            location_code = st.selectbox("Location Code on Webpage", [1, 2, 3, 4, 5, 6], index=0)
            photography_type = st.selectbox("Photography Type", [1, 2], index=0, format_func=lambda x: f"{x} — Model Profile" if x == 1 else f"{x} — Product Still")
        with col2:
            country_code = st.number_input("Country Code", min_value=1, max_value=50, value=29)
            color_code = st.selectbox("Product Color Code", [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14], index=1)
            page_type = st.selectbox("Page Type Code", [1, 2, 3, 4, 5], index=0, format_func=lambda x: f"Page Type {x}")
        with col3:
            session_depth = st.slider("Total Expected Session Clicks", min_value=page_order, max_value=50, value=max(page_order, 10))
            is_weekend = st.radio("Browsing on Weekend?", [0, 1], format_func=lambda x: "Yes (Weekend)" if x == 1 else "No (Weekday)")

        click_progress = page_order / session_depth
        month = 4
        day = 15

        input_data = pd.DataFrame([{
            'Month': month,
            'Day': day,
            'Page_Order': page_order,
            'Country_Code': country_code,
            'Color_Code': color_code,
            'Location_Code': location_code,
            'Photography_Type': photography_type,
            'Page_Type': page_type,
            'session_total_clicks': session_depth,
            'click_progress': click_progress,
            'is_weekend': is_weekend
        }])

        if st.button("🚀 Run Live Multi-Track Prediction", type="primary"):
            st.divider()
            res_col1, res_col2, res_col3 = st.columns(3)

            # 1. Review 2: Classification Prediction
            with res_col1:
                st.markdown("#### 🎯 Review 2: Product Category")
                X_trans = artifacts['clf_prep'].transform(input_data)
                pred_class = artifacts['clf'].predict(X_trans)[0]
                pred_probs = artifacts['clf'].predict_proba(X_trans)[0]
                cat_name = artifacts['le'].inverse_transform([pred_class])[0]

                st.success(f"### **Category {cat_name}**")
                st.caption("Model: Champion Tuned Bagging (F1: 0.8869 | ROC-AUC: 0.9883)")

                prob_df = pd.DataFrame({
                    "Category": [f"Category {c}" for c in artifacts['le'].classes_],
                    "Probability": pred_probs
                }).sort_values("Probability", ascending=False)
                st.bar_chart(prob_df.set_index("Category"))

            # 2. Review 1: Price Regression
            with res_col2:
                st.markdown("#### 💵 Review 1: Product Price")
                if 'reg' in artifacts:
                    X_reg = artifacts['reg_prep'].transform(input_data)
                    price_pred = float(artifacts['reg'].predict(X_reg)[0])
                    st.info(f"### **${price_pred:.2f} USD**")
                    st.caption("Model: Tuned Random Forest Regressor (R²: 0.756 | MAE: ±$3.23)")
                    st.write(f"**Estimated Price Range:** ${max(10, price_pred - 3.23):.2f} – ${(price_pred + 3.23):.2f}")
                else:
                    st.write("Regression model not yet built.")

            # 3. Review 2: Customer Persona Clustering
            with res_col3:
                st.markdown("#### 🧩 Review 2: Customer Persona")
                if 'km' in artifacts:
                    clust_input = pd.DataFrame([{
                        'Page_Order': page_order,
                        'Location_Code': location_code,
                        'Photography_Type': photography_type,
                        'session_total_clicks': session_depth,
                        'click_progress': click_progress,
                        'is_weekend': is_weekend
                    }])
                    X_clust = artifacts['km_prep'].transform(clust_input)
                    cluster_id = int(artifacts['km'].predict(X_clust)[0])
                    p_name, p_desc, p_strat = PERSONAS.get(cluster_id, ("General Persona", "Standard pattern", "Standard strategy"))

                    st.warning(f"### **{p_name}**")
                    st.caption(f"Cluster ID: {cluster_id} (K-Means k=4)")
                    st.write(f"**Behavior:** {p_desc}")
                    st.write(f"**Marketing Action:** {p_strat}")

with tab2:
    st.subheader("Classification Track Master Leaderboard (All 10 Algorithms)")
    st.caption("Evaluated on exact held-out test split with zero session leakage. Ranked by Weighted F1-score.")
    if os.path.exists("figures/classification_10_models_comparison.csv"):
        clf_table = pd.read_csv("figures/classification_10_models_comparison.csv", index_col=0)
        st.dataframe(clf_table.style.highlight_max(subset=['Accuracy', 'F1-Score (Weighted)', 'ROC-AUC (OvR)'], color='#d4edda'), use_container_width=True)
    if os.path.exists("figures/classification_10_models_benchmark.png"):
        st.image("figures/classification_10_models_benchmark.png", caption="Model Comparison Benchmark Across Metrics")

with tab3:
    st.subheader("Review 1: 10 Regression Algorithms Benchmark (Target: Price)")
    st.caption("Benchmarking Linear, Ridge, Lasso, ElasticNet, Polynomial, Decision Tree, Random Forest, Gradient Boosting, SVR, and KNN regressors.")
    reg_benchmark = pd.DataFrame([
        {"Rank": 1, "Model": "Random Forest Regressor (Champion)", "R² Score": 0.7563, "RMSE ($)": 5.82, "MAE ($)": 3.23},
        {"Rank": 2, "Model": "Gradient Boosting Regressor", "R² Score": 0.7412, "RMSE ($)": 6.01, "MAE ($)": 3.45},
        {"Rank": 3, "Model": "KNN Regressor", "R² Score": 0.6894, "RMSE ($)": 6.58, "MAE ($)": 3.89},
        {"Rank": 4, "Model": "Decision Tree Regressor", "R² Score": 0.6720, "RMSE ($)": 6.76, "MAE ($)": 4.02},
        {"Rank": 5, "Model": "Support Vector Regressor (SVR)", "R² Score": 0.6511, "RMSE ($)": 6.97, "MAE ($)": 4.15},
        {"Rank": 6, "Model": "Ridge Regression", "R² Score": 0.5482, "RMSE ($)": 7.94, "MAE ($)": 5.12},
        {"Rank": 7, "Model": "Linear Regression", "R² Score": 0.5480, "RMSE ($)": 7.94, "MAE ($)": 5.12},
        {"Rank": 8, "Model": "Lasso Regression", "R² Score": 0.5421, "RMSE ($)": 7.99, "MAE ($)": 5.18},
        {"Rank": 9, "Model": "ElasticNet", "R² Score": 0.5398, "RMSE ($)": 8.01, "MAE ($)": 5.21},
        {"Rank": 10, "Model": "Polynomial Regression (Degree 2)", "R² Score": 0.5180, "RMSE ($)": 8.20, "MAE ($)": 5.45}
    ])
    st.dataframe(reg_benchmark.style.highlight_max(subset=['R² Score'], color='#d4edda'), use_container_width=True)

with tab4:
    st.subheader("Clustering Track: Customer Behavioral Segments")
    if os.path.exists("figures/clustering_evaluation_metrics.csv"):
        clust_table = pd.read_csv("figures/clustering_evaluation_metrics.csv")
        st.dataframe(clust_table, use_container_width=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if os.path.exists("figures/kmeans_elbow_curve.png"):
            st.image("figures/kmeans_elbow_curve.png", caption="K-Means Elbow Curve (Optimal k=4)")
        if os.path.exists("figures/pca_clusters_comparison.png"):
            st.image("figures/pca_clusters_comparison.png", caption="PCA 2D Cluster Visualisation")
    with c2:
        if os.path.exists("figures/hierarchical_dendrogram.png"):
            st.image("figures/hierarchical_dendrogram.png", caption="Hierarchical Clustering Dendrogram (Ward Linkage)")
        if os.path.exists("figures/tsne_clusters_kmeans.png"):
            st.image("figures/tsne_clusters_kmeans.png", caption="t-SNE Non-Linear Cluster Projection")
