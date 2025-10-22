import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import logging
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

def explain_single_prediction(model, sample: pd.DataFrame, feature_names: list):
    try:
        sample = sample.iloc[[0]]  # one-row DataFrame

        # --- Model-specific SHAP explainer setup ---
        explainer = None

        # TreeExplainer for RandomForest or XGBoost
        if isinstance(model, (RandomForestClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(model, model_output="raw")
        else:
            return f"SHAP explanation not supported for model type: {type(model)}"

        # --- SHAP value extraction ---
        shap_values = explainer.shap_values(sample)
        shap_array = np.array(shap_values)
        logging.info(f"[SHAP] Raw SHAP array shape: {shap_array.shape}")

        pred_probs = model.predict_proba(sample)[0]
        pred_class = int(np.argmax(pred_probs))
        logging.info(f"[SHAP] Predicted class index: {pred_class}")

        if shap_array.ndim == 3:
            shap_array = shap_array[0].T
        elif shap_array.ndim == 2 and shap_array.shape[0] == len(feature_names):
            shap_array = np.expand_dims(shap_array, axis=0)
        elif shap_array.ndim == 2:
            shap_array = shap_array.T

        if pred_class >= shap_array.shape[0]:
            pred_class = shap_array.shape[0] - 1

        shap_vector = shap_array[pred_class]
        logging.info(f"SHAP vector values: {shap_vector}")
        contribs = pd.Series(shap_vector, index=feature_names)
        top_features = contribs.abs().nlargest(3).index.tolist()

        reason_map = {
            "xg_diff": "a significant expected goals (xG) advantage",
            "form_diff": "strong recent form",
            "home_shots_on_target": "high shot accuracy at home",
            "away_shots_on_target": "limiting away team shot threat",
            "possession": "control of possession",
            "corners": "territorial advantage",
            "h2h_home_wins": "strong head-to-head record at home",
            "h2h_away_wins": "historical away dominance"
        }
        reasons = [reason_map.get(f, f"notable {f.replace('_', ' ')}") for f in top_features]

        # --- Plotting SHAP impact ---
        plt.figure(figsize=(10, 6))
        try:
            if shap_array.shape[0] > 1:
                shap.summary_plot(shap_array, sample, feature_names=feature_names, plot_type="bar", show=False, max_display=10)
            else:
                shap.bar_plot(shap_array[0], feature_names=feature_names, show=False)

            filename = f"shap_plot_{type(model).__name__.lower()}.png"
            plt.tight_layout()
            plt.savefig(filename, bbox_inches='tight', dpi=300)
            plt.close()
        except Exception as e:
            return f"Visualization failed: {str(e)}"

        return f"The model based this decision on: {', '.join(reasons)}."

    except Exception as e:
        logging.error(f"Explanation failed: {str(e)}", exc_info=True)
        return f"Explanation failed: {str(e)}"
