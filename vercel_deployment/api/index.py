import os
import json
import joblib
import numpy as np
import pandas as pd
from http.server import BaseHTTPRequestHandler

# Define paths relative to the current file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELS_DIR = os.path.join(BASE_DIR, 'models')

# Global cache for model artifacts
_MODELS = None

def load_artifacts():
    global _MODELS
    if _MODELS is not None:
        return _MODELS

    # Classification artifacts (Review 2 Champion Bagging)
    clf = joblib.load(os.path.join(MODELS_DIR, 'best_classifier.pkl'))
    clf_prep = joblib.load(os.path.join(MODELS_DIR, 'preprocessor.pkl'))
    le = joblib.load(os.path.join(MODELS_DIR, 'label_encoder.pkl'))

    # Regression artifacts (Review 1 Price Regressor)
    reg = joblib.load(os.path.join(MODELS_DIR, 'best_regressor.pkl'))
    reg_prep = joblib.load(os.path.join(MODELS_DIR, 'reg_preprocessor.pkl'))

    # Clustering artifacts (Review 2 Customer Personas)
    km = joblib.load(os.path.join(MODELS_DIR, 'kmeans_model.pkl'))
    km_prep = joblib.load(os.path.join(MODELS_DIR, 'clust_preprocessor.pkl'))

    _MODELS = {
        'clf': clf,
        'clf_prep': clf_prep,
        'le': le,
        'reg': reg,
        'reg_prep': reg_prep,
        'km': km,
        'km_prep': km_prep
    }
    return _MODELS

# Cluster Persona Descriptions
PERSONAS = {
    0: {
        "title": "Quick Bouncer",
        "description": "Short exploratory visit with low click count (<3 clicks). User browses 1-2 items before exiting.",
        "strategy": "Show exit-intent discount popups and instant trending items."
    },
    1: {
        "title": "Weekend Explorer",
        "description": "Browsing predominantly on weekends with high exploration across photography styles and pages.",
        "strategy": "Engage with weekend promotional bundles and time-limited weekend discounts."
    },
    2: {
        "title": "Deeply Engaged Shopper",
        "description": "High browsing depth (>15 clicks) and systematic journey. Strong purchase intent.",
        "strategy": "Offer personalized product recommendations and exclusive loyalty club rewards."
    },
    3: {
        "title": "Category-Specific Browser",
        "description": "Focused page navigation patterns with targeted location browsing in specific catalog sectors.",
        "strategy": "Display category-specific cross-sell recommendations and customer reviews."
    }
}

class handler(BaseHTTPRequestHandler):
    def _send_json(self, data, status=200):
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_GET(self):
        # Health check endpoint
        models = load_artifacts()
        response = {
            "status": "online",
            "message": "Online Shopping Clickstream ML API (Reviews 1 & 2)",
            "models_loaded": {
                "review1_regression": "Random Forest Regressor (Target: Price)",
                "review2_classification": "Champion Tuned Bagging Classifier (Target: Main_Category)",
                "review2_clustering": "K-Means Customer Segmentation (k=4 Personas)"
            }
        }
        self._send_json(response)

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            payload = json.loads(post_data.decode('utf-8'))

            # Extract user parameters with robust defaults
            page_order = float(payload.get('page_order', 3))
            location_code = float(payload.get('location_code', 1))
            photography_type = float(payload.get('photography_type', 1))
            country_code = float(payload.get('country_code', 29))
            color_code = float(payload.get('color_code', 2))
            page_type = float(payload.get('page_type', 1))
            session_depth = float(payload.get('session_total_clicks', max(page_order, 8)))
            is_weekend = int(payload.get('is_weekend', 0))
            month = float(payload.get('month', 4))
            day = float(payload.get('day', 15))

            click_progress = float(page_order / max(session_depth, 1.0))

            # Load models
            artifacts = load_artifacts()

            # 1. Supervised Input DataFrame (for Classification & Regression)
            input_df = pd.DataFrame([{
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

            # Review 2: Classification Prediction
            X_clf = artifacts['clf_prep'].transform(input_df)
            pred_class_idx = artifacts['clf'].predict(X_clf)[0]
            cat_name = artifacts['le'].inverse_transform([pred_class_idx])[0]
            probs = artifacts['clf'].predict_proba(X_clf)[0].tolist()

            probabilities = {
                f"Category {c}": round(p, 4)
                for c, p in zip(artifacts['le'].classes_, probs)
            }

            # Review 1: Regression Prediction (Price in USD)
            X_reg = artifacts['reg_prep'].transform(input_df)
            price_pred = float(artifacts['reg'].predict(X_reg)[0])

            # Review 2: Unsupervised Clustering (Customer Persona)
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
            persona_info = PERSONAS.get(cluster_id, {
                "title": f"Cluster {cluster_id}",
                "description": "General browsing persona.",
                "strategy": "Standard catalog presentation."
            })

            result = {
                "success": True,
                "review1_regression": {
                    "predicted_price_usd": round(price_pred, 2),
                    "model_used": "Tuned Random Forest Regressor (R²=0.756, MAE=$3.23)"
                },
                "review2_classification": {
                    "predicted_category": f"Category {cat_name}",
                    "raw_class": int(cat_name),
                    "model_used": "Champion Tuned Bagging Classifier (F1=0.8869, ROC-AUC=0.9883)",
                    "category_probabilities": probabilities
                },
                "review2_clustering": {
                    "cluster_id": cluster_id,
                    "persona_name": persona_info["title"],
                    "persona_description": persona_info["description"],
                    "business_recommendation": persona_info["strategy"],
                    "model_used": "Unsupervised K-Means Clustering (k=4)"
                }
            }

            self._send_json(result)

        except Exception as e:
            self._send_json({"success": False, "error": str(e)}, status=500)
