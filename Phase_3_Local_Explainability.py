# 1. Global Setup & Imports
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
from imblearn.over_sampling import RandomOverSampler
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

ros = RandomOverSampler(random_state=RANDOM_SEED)
X_res, y_res = ros.fit_resample(X, y)


# A1. XGBoost Feature Mapping & Training
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

print("\n--- XGBOOST CLASSIFICATION REPORT ---")
print(classification_report(y_test_xgb, y_pred_xgb, target_names=['No Diabetes', 'Diabetes']))

plt.figure(figsize=(5,3))
sns.heatmap(confusion_matrix(y_test_xgb, y_pred_xgb), annot=True, fmt='d', cmap='Reds')
plt.title("Confusion Matrix: XGBoost + RFE")
plt.savefig('phase3_xgb_confusion.png', dpi=150, bbox_inches='tight')
plt.show()


# A2. XGBoost Global SHAP & Patient Isolation
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

xgb_patients = {
    'True_Positive': y_proba_xgb[tp_mask].argmax() if tp_mask.sum() > 0 else None,
    'False_Positive': y_proba_xgb[fp_mask].argmax() if fp_mask.sum() > 0 else None,
    'False_Negative': y_proba_xgb[fn_mask].argmin() if fn_mask.sum() > 0 else None
}

xgb_patients = {k: masks.index[v] if v is not None else None 
                for k, v, masks in zip(xgb_patients.keys(), xgb_patients.values(), [tp_mask[tp_mask], fp_mask[fp_mask], fn_mask[fn_mask]])}

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
print("\nXGBoost Elected Patients:")
display(df_xgb_pat)


# A3. XGBoost Local SHAP (Waterfall, Force, Decision)
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
    plt.figure()
    shap.plots.decision(sv.base_values, sv.values, row, feature_names=rfe_features, show=False)
    plt.title(f"XGBoost Decision | {p.replace('_', ' ')}")
    plt.savefig(f'phase3_xgb_decision_{p}.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Force
    plt.figure(figsize=(10, 3))
    shap.plots.force(sv.base_values, sv.values, row, feature_names=rfe_features, matplotlib=True, show=False)
    plt.savefig(f'phase3_xgb_force_{p}.png', dpi=150, bbox_inches='tight')
    plt.show()


# A4. XGBoost LIME & Agreement Analysis
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
    exp.as_pyplot_figure()
    plt.title(f"XGBoost LIME | {p.replace('_', ' ')}")
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
    sns.barplot(data=sub, x='Weight', y='Feature', hue='Type', ax=axes[i], palette=['#ff0051', '#008bfb'])
    axes[i].set_title(p)
    axes[i].axvline(0, color='k')
plt.suptitle("XGBoost SHAP vs LIME Exact Magnitudes", y=1.05)
plt.savefig('phase3_xgb_agreement_bar.png', dpi=150, bbox_inches='tight')
plt.show()


# A5. XGBoost Diagnostic Narrative
print("=================== XGBOOST NARRATIVES ===================")
for p, idx in xgb_patients.items():
    if idx is None: continue
    
    name = p.replace('_', ' ')
    row = X_te_xgb_df.iloc[idx].to_dict()
    sv = xgb_sv_dict[p]
    top_f1 = rfe_features[np.argsort(np.abs(sv.values))[::-1][0]]
    
    print(f"\n[{name.upper()}] XGBoost Evaluation:")
    if "True" in p and "Positive" in p:
        print(f"Accurately assessed diabetic risk. Primary instigator: {top_f1} at an elevated raw value of {row[top_f1]:.1f}. XGBoost successfully leveraged this marker.")
    elif "False" in p and "Positive" in p:
        print(f"False Alarm. XGBoost heavily penalized the patient due to {top_f1} resting at {row[top_f1]:.1f}, forcefully dragging the probability into positive bounds incorrectly.")
    else:
        print(f"Dangerous Miss. XGBoost was fatally suppressed by apparently normal markers like {top_f1} ({row[top_f1]:.1f}), which artificially protected the final internal sum.")


# B1. LightGBM Feature Mapping & Training
boruta_features = ['Glucose', 'SkinThickness', 'BMI', 'DiabetesPedigreeFunction', 'Age']
X_bor_main = X_res[boruta_features]

X_train_lgb, X_test_lgb, y_train_lgb, y_test_lgb = train_test_split(
    X_bor_main, y_res, test_size=0.3, random_state=RANDOM_SEED, stratify=y_res
)

lgbm_model = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)
lgbm_model.fit(X_train_lgb, y_train_lgb)

y_pred_lgb = lgbm_model.predict(X_test_lgb)
y_proba_lgb = lgbm_model.predict_proba(X_test_lgb)[:, 1]

print("\n--- LIGHTGBM CLASSIFICATION REPORT ---")
print(classification_report(y_test_lgb, y_pred_lgb, target_names=['No Diabetes', 'Diabetes']))

plt.figure(figsize=(5,3))
sns.heatmap(confusion_matrix(y_test_lgb, y_pred_lgb), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix: LightGBM + Boruta")
plt.savefig('phase3_lgb_confusion.png', dpi=150, bbox_inches='tight')
plt.show()


# B2. LightGBM Global SHAP & Patient Isolation
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

lgb_patients = {
    'True_Positive': y_proba_lgb[tp_mask2].argmax() if tp_mask2.sum() > 0 else None,
    'False_Positive': y_proba_lgb[fp_mask2].argmax() if fp_mask2.sum() > 0 else None,
    'False_Negative': y_proba_lgb[fn_mask2].argmin() if fn_mask2.sum() > 0 else None
}

lgb_patients = {k: masks.index[v] if v is not None else None 
                for k, v, masks in zip(lgb_patients.keys(), lgb_patients.values(), [tp_mask2[tp_mask2], fp_mask2[fp_mask2], fn_mask2[fn_mask2]])}

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
print("\nLightGBM Elected Patients:")
display(df_lgb_pat)


# B3. LightGBM Local SHAP (Waterfall, Force, Decision)
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
    
    plt.figure()
    shap.plots.decision(sv.base_values, sv.values, row, feature_names=boruta_features, show=False)
    plt.title(f"LightGBM Decision | {p.replace('_', ' ')}")
    plt.savefig(f'phase3_lgb_decision_{p}.png', dpi=150, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 3))
    shap.plots.force(sv.base_values, sv.values, row, feature_names=boruta_features, matplotlib=True, show=False)
    plt.savefig(f'phase3_lgb_force_{p}.png', dpi=150, bbox_inches='tight')
    plt.show()


# B4. LightGBM LIME & Agreement Analysis
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
    
    exp.as_pyplot_figure()
    plt.title(f"LightGBM LIME | {p.replace('_', ' ')}")
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
    sns.barplot(data=sub, x='Weight', y='Feature', hue='Type', ax=axes[i], palette=['#ff0051', '#008bfb'])
    axes[i].set_title(p)
    axes[i].axvline(0, color='k')
plt.suptitle("LightGBM SHAP vs LIME Exact Magnitudes", y=1.05)
plt.savefig('phase3_lgb_agreement_bar.png', dpi=150, bbox_inches='tight')
plt.show()


# B5. LightGBM Diagnostic Narrative
print("=================== LIGHTGBM NARRATIVES ===================")
for p, idx in lgb_patients.items():
    if idx is None: continue
    name = p.replace('_', ' ')
    row = X_te_lgb_df.iloc[idx].to_dict()
    sv = lgb_sv_dict[p]
    top_f1 = boruta_features[np.argsort(np.abs(sv.values))[::-1][0]]
    
    print(f"\n[{name.upper()}] LightGBM Evaluation:")
    if "True" in p and "Positive" in p: print(f"LightGBM captured the diabetic risk accurately using structural reliance primarily on {top_f1} measuring precisely {row[top_f1]:.1f}.")
    elif "False" in p and "Positive" in p: print(f"LightGBM falsely tripped its alarm heavily corrupted by an anomaly reading of {top_f1} mapping to {row[top_f1]:.1f}.")
    else: print(f"LightGBM masked an actual risk instance. A false protective metric in {top_f1} at {row[top_f1]:.1f} completely dissolved LightGBM's threshold limits.")


# Compile Artifacts
with zipfile.ZipFile('phase3_split_analysis_plots.zip', 'w') as zipf:
    for file in glob.glob('phase3_*.png'):
        zipf.write(file)

print("✅ Saved all plots from both configurations to phase3_split_analysis_plots.zip")

try:
    from google.colab import files
    files.download('phase3_split_analysis_plots.zip')
except ImportError:
    pass

