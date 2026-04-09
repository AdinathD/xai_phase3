import nbformat as nbf
import textwrap

def create_nb():
    nb = nbf.v4.new_notebook()
    cells = []

    # ========================== OVERVIEW & SETUP ==========================
    cells.append(nbf.v4.new_markdown_cell("# Phase 3: Comprehensive Explainability Analysis (XGBoost vs LightGBM)\nFollowing Phase 2 comparisons, we now dive into per-patient explainability for the exact configurations requested:\n\n1. **PART A: Our Best Experimental Configuration (XGBoost + RFE ~84.62%)**\n2. **PART B: The Paper's Stated Best Model (LightGBM + Boruta ~85.16%)**\n\nThe notebook explicitly performs a full end-to-end SHAP and LIME audit natively separated into distinct blocks to prevent variable cross-contamination."))

    cells.append(nbf.v4.new_code_cell("""# 1. Global Setup & Imports
%pip install -q shap lime xgboost lightgbm imbalanced-learn seaborn

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import lime
import lime.lime_tabular
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from collections import Counter
import warnings
import zipfile
import glob
import os

warnings.filterwarnings('ignore')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

CSV_PATH = 'diabetes.csv' 
df = pd.read_csv(CSV_PATH)
col_lower = {c.lower(): c for c in df.columns}
if 'outcome' not in col_lower:
    for alias in ['class', 'target', 'label', 'diabetes']:
        if alias in col_lower:
            df.rename(columns={col_lower[alias]: 'Outcome'}, inplace=True)
            break
df['Outcome'] = df['Outcome'].astype(int)

# Core Preprocessing Shared by Both
zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_cols = [c for c in zero_cols if c in df.columns]
df[zero_cols] = df[zero_cols].replace(0, np.nan)
for col in zero_cols: df[col] = df[col].fillna(df[col].mean())
for col in df.select_dtypes(include=[np.number]).columns:
    if col != 'Outcome': df[col] = df[col].fillna(df[col].mean())

df_clean = df.copy()
for col in [c for c in df.columns if c != 'Outcome']:
    Q1, Q3 = df_clean[col].quantile(0.25), df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    df_clean = df_clean[(df_clean[col] >= Q1 - 1.5 * IQR) & (df_clean[col] <= Q3 + 1.5 * IQR)]

X = df_clean[[c for c in df.columns if c != 'Outcome']]
y = df_clean['Outcome']

smote = SMOTE(random_state=RANDOM_SEED)
X_res, y_res = smote.fit_resample(X, y)
"""))

    # =========================================================================
    # PART A: XGBOOST + RFE
    # =========================================================================
    
    cells.append(nbf.v4.new_markdown_cell("# ===========================================\n# PART A: Our Best Model (XGBoost + RFE)\n# ===========================================\nWe recreate the RFE feature extraction and train XGBoost, our empirically strongest standalone pipeline."))

    cells.append(nbf.v4.new_code_cell("""# A1. XGBoost Feature Mapping & Training
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
rfe_fold_features = []
for train_idx, _ in cv5.split(X_res, y_res):
    rfe = RFE(estimator=LGBMClassifier(random_state=RANDOM_SEED, verbose=-1), n_features_to_select=5)
    rfe.fit(X_res.iloc[train_idx], y_res.iloc[train_idx])
    rfe_fold_features.extend(X_res.columns[rfe.support_].tolist())

rfe_features = [f for f, cnt in Counter(rfe_fold_features).items() if cnt >= 3]
print(f"RFE dynamically selected {len(rfe_features)} features: {rfe_features}")

X_rfe_main = X_res[rfe_features]
X_train_xgb, X_test_xgb, y_train_xgb, y_test_xgb = train_test_split(
    X_rfe_main, y_res, test_size=0.3, random_state=RANDOM_SEED, stratify=y_res
)

xgb_model = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)
xgb_model.fit(X_train_xgb, y_train_xgb)

y_pred_xgb = xgb_model.predict(X_test_xgb)
y_proba_xgb = xgb_model.predict_proba(X_test_xgb)[:, 1]

print("\\n--- XGBOOST CLASSIFICATION REPORT ---")
print(classification_report(y_test_xgb, y_pred_xgb, target_names=['No Diabetes', 'Diabetes']))

plt.figure(figsize=(5,3))
sns.heatmap(confusion_matrix(y_test_xgb, y_pred_xgb), annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix: XGBoost + RFE")
plt.savefig('phase3_xgb_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# A2. XGBoost Global SHAP & Patient Isolation
X_te_xgb_df = X_test_xgb.reset_index(drop=True)
explainer_xgb = shap.TreeExplainer(xgb_model)
shap_obj_xgb = explainer_xgb(X_te_xgb_df)

# Adjust dimensions safely
sv_global_xgb = shap_obj_xgb[:, :, 1] if len(shap_obj_xgb.shape) == 3 else shap_obj_xgb

plt.figure(figsize=(7,5))
shap.plots.beeswarm(sv_global_xgb, show=False)
plt.title("XGBoost Global SHAP", pad=15)
plt.savefig('phase3_xgb_global_shap.png', dpi=150, bbox_inches='tight')
plt.show()

y_test_s = pd.Series(y_test_xgb).reset_index(drop=True)
y_pred_s = pd.Series(y_pred_xgb).reset_index(drop=True)

tp_mask = (y_test_s == 1) & (y_pred_s == 1)
fp_mask = (y_test_s == 0) & (y_pred_s == 1)
fn_mask = (y_test_s == 1) & (y_pred_s == 0)

y_proba_s = pd.Series(y_proba_xgb)
tp_idx = int(y_proba_s[tp_mask].idxmax()) if tp_mask.sum() > 0 else None
fp_idx = int(y_proba_s[fp_mask].idxmax()) if fp_mask.sum() > 0 else None
fn_idx = int(y_proba_s[fn_mask].idxmin()) if fn_mask.sum() > 0 else None

xgb_patients = {
    'True_Positive': tp_idx,
    'False_Positive': fp_idx,
    'False_Negative': fn_idx
}

print("Selected Patients (XGBoost):")
for p_type, idx in xgb_patients.items():
    if idx is not None:
        print(f"{p_type}: idx={idx}, actual={y_test_s[idx]}, pred={y_pred_s[idx]}, conf={y_proba_xgb[idx]:.4f}")

records_xgb = []
for p, idx in xgb_patients.items():
    if idx is None: continue
    rec = X_te_xgb_df.iloc[idx].to_dict()
    rec['Patient Type'] = p.replace('_', ' ')
    rec['Actual'] = y_test_s.iloc[idx]
    rec['Predicted'] = y_pred_s.iloc[idx]
    rec['Confidence'] = f"{y_proba_xgb[idx]*100:.1f}%"
    records_xgb.append(rec)

df_xgb_pat = pd.DataFrame(records_xgb).set_index('Patient Type')
print("\\nXGBoost Elected Patients:")
display(df_xgb_pat)

full_ref = {
    'Glucose': {'Range': '70–99 mg/dL', 'Indicator': '≥126 mg/dL (Diabetic)'},
    'BloodPressure': {'Range': '60-80 mmHg', 'Indicator': '≥90 mmHg (Hypertension)'},
    'BMI': {'Range': '18.5–24.9', 'Indicator': '≥30 (Obese)'},
    'DiabetesPedigreeFunction': {'Range': '<0.5 (low risk)', 'Indicator': '>1.0 (high family risk)'},
    'Age': {'Range': '<45 years', 'Indicator': 'increasing risk with age'},
    'SkinThickness': {'Range': '<35mm (F)', 'Indicator': 'elevated'},
    'Insulin': {'Range': '15-166 mIU/L', 'Indicator': 'abnormal if outside range'}
}

ref_data = []
for f in rfe_features:
    if f in full_ref:
        ref_data.append({'Feature': f, 'Normal range': full_ref[f]['Range'], 'Diabetic indicator': full_ref[f]['Indicator']})

ref_table = pd.DataFrame(ref_data)
print("\\nClinical Reference Ranges (RFE Features):")
display(ref_table)
"""))

    cells.append(nbf.v4.new_code_cell("""# A3. XGBoost Local SHAP (Waterfall, Force, Decision)
xgb_sv_dict = {}

for p, idx in xgb_patients.items():
    if idx is None: continue
    row = X_te_xgb_df.iloc[[idx]]
    obj = explainer_xgb(row)
    sv = obj[0, :, 1] if len(obj.shape) == 3 else obj[0]
    xgb_sv_dict[p] = sv
    
    # Waterfall
    plt.figure()
    shap.plots.waterfall(sv, show=False)
    plt.title(f"XGBoost Waterfall | {p.replace('_', ' ')}")
    plt.savefig(f'phase3_xgb_waterfall_{p}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Decision
    try:
        base_val = float(np.atleast_1d(sv.base_values)[0])
        plt.figure()
        shap.decision_plot(base_val, sv.values, row.iloc[0], feature_names=rfe_features, show=False)
        plt.title(f"XGBoost Decision | {p.replace('_', ' ')}")
        plt.savefig(f'phase3_xgb_decision_{p}.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print("Decision plot parsing bypass:", e)

    # Force
    try:
        base_val = float(np.atleast_1d(sv.base_values)[0])
        shap.force_plot(base_val, sv.values, row.iloc[0], feature_names=rfe_features, matplotlib=True, show=False)
        plt.savefig(f'phase3_xgb_force_{p}.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print("Force plot parsing bypass:", e)
"""))

    cells.append(nbf.v4.new_code_cell("""# A4. XGBoost LIME & Agreement Analysis
lime_xgb = lime.lime_tabular.LimeTabularExplainer(
    X_train_xgb.values, feature_names=rfe_features, class_names=['No', 'Yes'], mode='classification', random_state=RANDOM_SEED
)

lime_xgb_results = {}
plot_data_xgb = []
agreement_xgb = []

for p, idx in xgb_patients.items():
    if idx is None: continue
    row = X_te_xgb_df.iloc[idx]
    exp = lime_xgb.explain_instance(row.values, xgb_model.predict_proba, num_features=5)
    lime_xgb_results[p] = exp
    
    # Lime display
    exp.as_pyplot_figure(label=1)
    plt.title(f"XGBoost LIME | {p.replace('_', ' ')}")
    plt.xlabel("Feature weight → Diabetes")
    plt.savefig(f'phase3_xgb_lime_{p}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # Extraction
    lime_map = dict(exp.as_map()[1])
    lime_top_idx = sorted(lime_map, key=lambda i: abs(lime_map[i]), reverse=True)[:3]
    lime_top_names = [rfe_features[i] for i in lime_top_idx]
    
    sv = xgb_sv_dict[p]
    shap_top_idx = np.argsort(np.abs(sv.values))[-3:][::-1]
    shap_top_names = [rfe_features[i] for i in shap_top_idx]
    
    intersect = len(set(lime_top_names) & set(shap_top_names))
    agreement_xgb.append({'Patient': p.replace('_', ' '), 'SHAP Top 3': ", ".join(shap_top_names), 'LIME Top 3': ", ".join(lime_top_names), 'Match': intersect})
    
    for i, f in enumerate(rfe_features):
        plot_data_xgb.append({'Patient': p.replace('_', ' '), 'Feature': f, 'Weight': sv.values[i], 'Type': 'SHAP'})
        plot_data_xgb.append({'Patient': p.replace('_', ' '), 'Feature': f, 'Weight': lime_map.get(i, 0), 'Type': 'LIME'})

display(pd.DataFrame(agreement_xgb))

df_agg = pd.DataFrame(plot_data_xgb)
fig, axes = plt.subplots(1, 3, figsize=(15,4), sharey=True)
for i, p in enumerate([c['Patient'] for c in agreement_xgb]):
    sub = df_agg[df_agg['Patient'] == p]
    sns.barplot(data=sub, x='Weight', y='Feature', hue='Type', ax=axes[i], palette={'SHAP': 'tomato', 'LIME': 'steelblue'})
    axes[i].set_title(p)
    axes[i].set_xlabel("Feature weight (impact on diabetic prediction)")
    axes[i].set_ylabel("Feature" if i == 0 else "")
    axes[i].axvline(0, color='k')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle("XGBoost SHAP vs LIME Exact Magnitudes", y=1.05)
plt.savefig('phase3_xgb_agreement_bar.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# A5. XGBoost Diagnostic Narrative
print("=================== XGBOOST NARRATIVES ===================")
for p, idx in xgb_patients.items():
    if idx is None: continue
    
    name = p.replace('_', ' ')
    row = X_te_xgb_df.iloc[idx].to_dict()
    sv = xgb_sv_dict[p]
    top_f1 = rfe_features[np.argsort(np.abs(sv.values))[::-1][0]]
    
    print(f"\\n[{name.upper()}] XGBoost Evaluation:")
    
    thresholds = {
        'Glucose': 126, 'BMI': 30, 'Age': 45, 'SkinThickness': 35, 'DiabetesPedigreeFunction': 0.5
    }
    
    status = []
    for f in rfe_features:
        if f in thresholds:
            val = row[f]
            is_high = val >= thresholds[f]
            check = "ELEVATED" if is_high else "normal range"
            status.append(f"{f}: {val:.1f} ({check})")
    
    print(" | ".join(status))
    
    if "True" in p and "Positive" in p:
        print(f"The model correctly classified this patient as diabetic. The primary contributing feature was {top_f1}, "
              f"which reached {row[top_f1]:.1f}. This value is within the diabetic range, aligned with clinical logic.")
    elif "False" in p and "Positive" in p:
        print(f"The model incorrectly classified this healthy patient as diabetic. This was primarily driven by the "
              f"elevated value of {top_f1} ({row[top_f1]:.1f}), which overshadowed other normal indicators.")
    else:
        print(f"The model missed this diabetic diagnosis. Indicators like {top_f1} were within the normal range ({row[top_f1]:.1f}), "
              "which suppressed the overall risk score despite the patient's actual condition.")
"""))

    # =========================================================================
    # PART B: LIGHTGBM + BORUTA
    # =========================================================================

    cells.append(nbf.v4.new_markdown_cell("# ===========================================\n# PART B: Paper's Best Model (LightGBM + Boruta)\n# ===========================================\nWe recreate the Boruta selected features exclusively and train the LightGBM pipeline identically to track differences."))

    cells.append(nbf.v4.new_code_cell("""# B1. LightGBM Feature Mapping & Training
# Boruta features exactly as identified in Phase 2 results
boruta_features = ['Glucose', 'BMI', 'DiabetesPedigreeFunction', 'Age']
print(f"Boruta (Paper Configuration) selected {len(boruta_features)} features: {boruta_features}")

X_bor_main = X_res[boruta_features]

X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb = train_test_split(
    X_bor_main, y_res, test_size=0.3, random_state=RANDOM_SEED, stratify=y_res
)

lgbm_model = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)
lgbm_model.fit(X_train_lgb, y_train_lgb)

y_pred_lgb = lgbm_model.predict(X_test_lgb)
y_proba_lgb = lgbm_model.predict_proba(X_test_lgb)[:, 1]

print("\\n--- LIGHTGBM CLASSIFICATION REPORT ---")
print(classification_report(y_test_lgb, y_pred_lgb, target_names=['No Diabetes', 'Diabetes']))

plt.figure(figsize=(5,3))
sns.heatmap(confusion_matrix(y_test_lgb, y_pred_lgb), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: LightGBM + Boruta")
plt.savefig('phase3_lgb_confusion.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# B2. LightGBM Global SHAP & Patient Isolation
X_te_lgb_df = X_test_lgb.reset_index(drop=True)
explainer_lgb = shap.TreeExplainer(lgbm_model)
shap_obj_lgb = explainer_lgb(X_te_lgb_df)

sv_global_lgb = shap_obj_lgb[:, :, 1] if len(shap_obj_lgb.shape) == 3 else shap_obj_lgb

plt.figure(figsize=(7,5))
shap.plots.beeswarm(sv_global_lgb, show=False)
plt.title("LightGBM Global SHAP", pad=15)
plt.savefig('phase3_lgb_global_shap.png', dpi=150, bbox_inches='tight')
plt.show()

y_test_s2 = pd.Series(y_test_lgb).reset_index(drop=True)
y_pred_s2 = pd.Series(y_pred_lgb).reset_index(drop=True)

tp_mask2 = (y_test_s2 == 1) & (y_pred_s2 == 1)
fp_mask2 = (y_test_s2 == 0) & (y_pred_s2 == 1)
fn_mask2 = (y_test_s2 == 1) & (y_pred_s2 == 0)

y_proba_s2 = pd.Series(y_proba_lgb)
tp_idx2 = int(y_proba_s2[tp_mask2].idxmax()) if tp_mask2.sum() > 0 else None
fp_idx2 = int(y_proba_s2[fp_mask2].idxmax()) if fp_mask2.sum() > 0 else None
fn_idx2 = int(y_proba_s2[fn_mask2].idxmin()) if fn_mask2.sum() > 0 else None

lgb_patients = {
    'True_Positive': tp_idx2,
    'False_Positive': fp_idx2,
    'False_Negative': fn_idx2
}

print("Selected Patients (LightGBM):")
for p_type, idx in lgb_patients.items():
    if idx is not None:
        print(f"{p_type}: idx={idx}, actual={y_test_s2[idx]}, pred={y_pred_s2[idx]}, conf={y_proba_lgb[idx]:.4f}")

records_lgb = []
for p, idx in lgb_patients.items():
    if idx is None: continue
    rec = X_te_lgb_df.iloc[idx].to_dict()
    rec['Patient Type'] = p.replace('_', ' ')
    rec['Actual'] = y_test_s2.iloc[idx]
    rec['Predicted'] = y_pred_s2.iloc[idx]
    rec['Confidence'] = f"{y_proba_lgb[idx]*100:.1f}%"
    records_lgb.append(rec)

df_lgb_pat = pd.DataFrame(records_lgb).set_index('Patient Type')
print("\\nLightGBM Elected Patients:")
display(df_lgb_pat)

ref_data_boruta = []
for f in boruta_features:
    if f in full_ref:
        ref_data_boruta.append({'Feature': f, 'Normal range': full_ref[f]['Range'], 'Diabetic indicator': full_ref[f]['Indicator']})

ref_table_boruta = pd.DataFrame(ref_data_boruta)
print("\\nClinical Reference Ranges (Boruta Features):")
display(ref_table_boruta)
"""))

    cells.append(nbf.v4.new_code_cell("""# B3. LightGBM Local SHAP (Waterfall, Force, Decision)
lgb_sv_dict = {}

for p, idx in lgb_patients.items():
    if idx is None: continue
    row = X_te_lgb_df.iloc[[idx]]
    obj = explainer_lgb(row)
    sv = obj[0, :, 1] if len(obj.shape) == 3 else obj[0]
    lgb_sv_dict[p] = sv
    
    plt.figure()
    shap.plots.waterfall(sv, show=False)
    plt.title(f"LightGBM Waterfall | {p.replace('_', ' ')}")
    plt.savefig(f'phase3_lgb_waterfall_{p}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    try:
        base_val_lgb = float(np.atleast_1d(sv.base_values)[0])
        plt.figure()
        shap.decision_plot(base_val_lgb, sv.values, row.iloc[0], feature_names=boruta_features, show=False)
        plt.title(f"LightGBM Decision | {p.replace('_', ' ')}")
        plt.savefig(f'phase3_lgb_decision_{p}.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print("Decision plot parsing bypass:", e)

    try:
        base_val_lgb = float(np.atleast_1d(sv.base_values)[0])
        shap.force_plot(base_val_lgb, sv.values, row.iloc[0], feature_names=boruta_features, matplotlib=True, show=False)
        plt.savefig(f'phase3_lgb_force_{p}.png', dpi=150, bbox_inches='tight')
        plt.show()
    except Exception as e:
        print("Force plot parsing bypass:", e)
"""))

    cells.append(nbf.v4.new_code_cell("""# B4. LightGBM LIME & Agreement Analysis
lime_lgb = lime.lime_tabular.LimeTabularExplainer(
    X_train_lgb.values, feature_names=boruta_features, class_names=['No', 'Yes'], mode='classification', random_state=RANDOM_SEED
)

lime_lgb_results = {}
plot_data_lgb = []
agreement_lgb = []

for p, idx in lgb_patients.items():
    if idx is None: continue
    row = X_te_lgb_df.iloc[idx]
    exp = lime_lgb.explain_instance(row.values, lgbm_model.predict_proba, num_features=5)
    
    exp.as_pyplot_figure(label=1)
    plt.title(f"LightGBM LIME | {p.replace('_', ' ')}")
    plt.xlabel("Feature weight → Diabetes")
    plt.savefig(f'phase3_lgb_lime_{p}.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    lime_map = dict(exp.as_map()[1])
    lime_top_idx = sorted(lime_map, key=lambda i: abs(lime_map[i]), reverse=True)[:3]
    lime_top_names = [boruta_features[i] for i in lime_top_idx]
    
    sv = lgb_sv_dict[p]
    shap_top_idx = np.argsort(np.abs(sv.values))[-3:][::-1]
    shap_top_names = [boruta_features[i] for i in shap_top_idx]
    
    intersect = len(set(lime_top_names) & set(shap_top_names))
    agreement_lgb.append({'Patient': p.replace('_', ' '), 'SHAP Top 3': ", ".join(shap_top_names), 'LIME Top 3': ", ".join(lime_top_names), 'Match': intersect})
    
    for i, f in enumerate(boruta_features):
        plot_data_lgb.append({'Patient': p.replace('_', ' '), 'Feature': f, 'Weight': sv.values[i], 'Type': 'SHAP'})
        plot_data_lgb.append({'Patient': p.replace('_', ' '), 'Feature': f, 'Weight': lime_map.get(i, 0), 'Type': 'LIME'})

display(pd.DataFrame(agreement_lgb))

df_agg2 = pd.DataFrame(plot_data_lgb)
fig, axes = plt.subplots(1, 3, figsize=(15,4), sharey=True)
for i, p in enumerate([c['Patient'] for c in agreement_lgb]):
    sub = df_agg2[df_agg2['Patient'] == p]
    sns.barplot(data=sub, x='Weight', y='Feature', hue='Type', ax=axes[i], palette={'SHAP': 'tomato', 'LIME': 'steelblue'})
    axes[i].set_title(p)
    axes[i].set_xlabel("Feature weight (impact on diabetic prediction)")
    axes[i].set_ylabel("Feature" if i == 0 else "")
    axes[i].axvline(0, color='k')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.suptitle("LightGBM SHAP vs LIME Exact Magnitudes", y=1.05)
plt.savefig('phase3_lgb_agreement_bar.png', dpi=150, bbox_inches='tight')
plt.show()
"""))

    cells.append(nbf.v4.new_code_cell("""# B5. LightGBM Diagnostic Narrative
print("=================== LIGHTGBM NARRATIVES ===================")
for p, idx in lgb_patients.items():
    if idx is None: continue
    name = p.replace('_', ' ')
    row = X_te_lgb_df.iloc[idx].to_dict()
    sv = lgb_sv_dict[p]
    top_f1 = boruta_features[np.argsort(np.abs(sv.values))[::-1][0]]
    
    print(f"\\n[{name.upper()}] LightGBM Evaluation:")
    
    status = []
    for f in boruta_features:
        if f in thresholds:
            val = row[f]
            is_high = val >= thresholds[f]
            check = "ELEVATED" if is_high else "normal range"
            status.append(f"{f}: {val:.1f} ({check})")
    
    print(" | ".join(status))
    
    if "True" in p and "Positive" in p:
        print(f"The model correctly classified this patient as diabetic. This classification was primarily driven by {top_f1} "
              f"({row[top_f1]:.1f}), which is within the elevated range.")
    elif "False" in p and "Positive" in p:
        print(f"The model incorrectly classified this patient as diabetic. This false result was misled by {top_f1} "
              f"reaching {row[top_f1]:.1f}, despite other indicators and the actual status being healthy.")
    else:
        print(f"The model missed this diabetic diagnosis. The primary factor, {top_f1}, was within the normal range ({row[top_f1]:.1f}), "
              "which suppressed the risk score despite the underlying diabetic status.")
"""))

    # ========================== PART C: CUSTOM AUDIT ==========================
    cells.append(nbf.v4.new_markdown_cell("# ## PART C: Custom Instance Audit (User Choice)\nIf you have identified a specific patient from your test set (indices 0 to 230) and wish to perform a deep-dive independent of the automated confidence-based selection above, use this cell.\n\nSimply input the index and choose the model to deconstruct."))
    
    cells.append(nbf.v4.new_code_cell("""# PART C. Custom deep-dive
# --- CONFIGURABLE INPUTS ---
chosen_idx = 0       # Choose ANY index from 0 to 230
use_model = 'XGB'   # Choose 'XGB' (5 features) or 'LGB' (4 features)
# ---------------------------

thresholds = {
    'Glucose': 126, 'BMI': 30, 'Age': 45, 'SkinThickness': 35, 'DiabetesPedigreeFunction': 0.5
}

# Setup feature sets and models
if use_model.upper() == 'XGB':
    model = xgb_model
    explainer = explainer_xgb
    features = rfe_features
    test_df = X_te_xgb_df
    y_test_cur = pd.Series(y_test_xgb).reset_index(drop=True)
    y_pred_cur = pd.Series(y_pred_xgb).reset_index(drop=True)
    y_proba_cur = y_proba_xgb
    lime_explainer = lime_xgb
else:
    model = lgbm_model
    explainer = explainer_lgb
    features = boruta_features
    test_df = X_te_lgb_df
    y_test_cur = pd.Series(y_test_lgb).reset_index(drop=True)
    y_pred_cur = pd.Series(y_pred_lgb).reset_index(drop=True)
    y_proba_cur = y_proba_lgb
    lime_explainer = lime_lgb

# Extract data for the single instance
row_data = test_df.iloc[[chosen_idx]]
vals = row_data.iloc[0].to_dict()
actual = y_test_cur.iloc[chosen_idx]
pred = y_pred_cur.iloc[chosen_idx]
conf = f"{y_proba_cur[chosen_idx]*100:.2f}%"

print(f"--- CUSTOM AUDIT: Instance #{chosen_idx} using {use_model} ---")
print(f"Actual: {actual} | Predicted: {pred} | Confidence: {conf}")

# 1. Waterfall
shap_obj = explainer(row_data)
sv = shap_obj[0, :, 1] if len(shap_obj.shape) == 3 else shap_obj[0]
plt.figure()
shap.plots.waterfall(sv, show=False)
plt.title(f"Custom Audit: Waterfall (Idx {chosen_idx})")
plt.show()

# 2. Decision
try:
    base_val = float(np.atleast_1d(sv.base_values)[0])
    plt.figure()
    shap.decision_plot(base_val, sv.values, row_data.iloc[0], feature_names=features, show=False)
    plt.title(f"Custom Audit: Decision Plot")
    plt.show()
except Exception as e:
    print("Decision bypass:", e)

# 3. LIME Bar
lime_exp = lime_explainer.explain_instance(test_df.iloc[chosen_idx].values, model.predict_proba, num_features=len(features))
lime_exp.as_pyplot_figure(label=1)
plt.title(f"Custom Audit: LIME (Diabetic Impact)")
plt.xlabel("Feature weight → Diabetes")
plt.show()

# 4. Narratives (toned clinical wording)
top_f = features[np.argsort(np.abs(sv.values))[::-1][0]]
print(f"Custom Diagnosis Narrative:")
st_list = [f"{f}: {vals[f]:.1f} ({'ELEVATED' if vals[f] >= thresholds.get(f, 999) else 'normal'})" for f in features]
print(" | ".join(st_list))

print(f"Primary Diagnosis Driver: {top_f} at {vals[top_f]:.1f}.")
if actual == pred:
    print(f"Interpretation: The instance is accurately classified.")
else:
    print(f"Interpretation: The instance is misclassified (Prediction Error). Check if {top_f} misled the model threshold.")
"""))

    # ========================== COMPILE ARTIFACTS ==========================
    cells.append(nbf.v4.new_markdown_cell("# Compile Final Artifacts"))
    cells.append(nbf.v4.new_code_cell("""# Compile Artifacts
with zipfile.ZipFile('phase3_split_analysis_plots.zip', 'w') as zipf:
    for file in glob.glob('phase3_*.png'):
        zipf.write(file)

print("✅ Saved all plots from both configurations to phase3_split_analysis_plots.zip")

try:
    from google.colab import files
    files.download('phase3_split_analysis_plots.zip')
except ImportError:
    pass
"""))

    nb['cells'] = cells
    with open('Phase_3_Local_Explainability.ipynb', 'w', encoding='utf-8') as f:
        nbf.write(nb, f)

create_nb()
print("Massive end-to-end split notebook fully constructed.")
