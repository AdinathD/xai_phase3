import pandas as pd
import numpy as np
import shap
from xgboost import XGBClassifier

df = pd.read_csv('diabetes.csv')
X = df[['Glucose', 'BMI', 'Age', 'DiabetesPedigreeFunction', 'BloodPressure']]
y = df['Outcome']

model = XGBClassifier(random_state=42, eval_metric='logloss', verbosity=0)
model.fit(X, y)

explainer = shap.TreeExplainer(model)
shap_obj = explainer(X.iloc[[0]])
sv = shap_obj[0]
print("SV Base value:", sv.base_values)
print("SV type:", type(sv.base_values))

try:
    shap.plots.decision(sv.base_values, sv.values, X.iloc[[0]], feature_names=X.columns.tolist())
    print("shap.plots.decision with sv.base_values worked")
except Exception as e:
    print("shap.plots.decision with sv.base_values failed:", repr(e))

try:
    base_val = float(sv.base_values[0]) if isinstance(sv.base_values, (list, np.ndarray)) else float(sv.base_values)
    shap.plots.decision(base_val, sv.values, X.iloc[[0]], feature_names=X.columns.tolist())
    print("shap.plots.decision with float worked")
except Exception as e:
    print("shap.plots.decision with float failed:", repr(e))

try:
    base_val = float(sv.base_values[0]) if isinstance(sv.base_values, (list, np.ndarray)) else float(sv.base_values)
    shap.decision_plot(base_val, sv.values, X.iloc[0], feature_names=X.columns.tolist())
    print("shap.decision_plot with 1D worked")
except Exception as e:
    print("shap.decision_plot with 1D failed:", repr(e))

