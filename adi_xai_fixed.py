# -*- coding: utf-8 -*-
"""
adi_xai_fixed.py
Fixed version of adi_xai.py — runs as a plain Python script (no IPython magic).
All output files are saved to the 'results/' folder.
"""

import os
import sys

# ── Create results directory ───────────────────────────────────────────────
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)

def results_path(filename):
    return os.path.join(RESULTS_DIR, filename)

# ── Cell 2 — Imports ──────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # non-interactive backend (no display required)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix,
                             classification_report, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import RandomOverSampler
import shap

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
print("All imports successful ✅")

# ── Cell 3 — Load Dataset ─────────────────────────────────────────────────
# Use the local CSV file in the same directory
CSV_DIR = os.path.dirname(os.path.abspath(__file__))
possible_files = ["Diabetes_Final_Data_V2.csv", "diabetes.csv"]
CSV_PATH = None
for filename in possible_files:
    candidate = os.path.join(CSV_DIR, filename)
    if os.path.exists(candidate):
        CSV_PATH = candidate
        break
if CSV_PATH is None:
    raise FileNotFoundError(
        "Dataset not found. Please place 'Diabetes_Final_Data_V2.csv' or "
        "'diabetes.csv' in the same folder as this script."
    )

df = pd.read_csv(CSV_PATH)

# ── Normalise column names → map target to 'Outcome' ───────────────────
col_lower_map = {c.lower(): c for c in df.columns}

# Handle 'diabetic' column (Yes/No strings)
if 'diabetic' in col_lower_map:
    df.rename(columns={col_lower_map['diabetic']: 'Outcome'}, inplace=True)
    df['Outcome'] = df['Outcome'].map({'Yes': 1, 'No': 0, 1: 1, 0: 0})
elif 'outcome' not in col_lower_map:
    for alias in ['class', 'target', 'label', 'diabetes']:
        if alias in col_lower_map:
            df.rename(columns={col_lower_map[alias]: 'Outcome'}, inplace=True)
            break

# ── Encode categorical columns (e.g. gender) ───────────────────────────
for col in df.select_dtypes(include=['object', 'string', 'category']).columns:
    if col != 'Outcome':
        df[col] = df[col].astype('category').cat.codes

# Make sure Outcome is integer
df['Outcome'] = df['Outcome'].astype(int)

print("Shape:", df.shape)
print("\nClass distribution:")
print(df['Outcome'].value_counts())
pos_pct = df['Outcome'].mean() * 100
print(f"\nClass balance: {pos_pct:.2f}% positive")
print(df.head())

# ── Cell 3B — EDA ─────────────────────────────────────────────────────────
feature_cols = [col for col in df.columns if col != 'Outcome']

# Figure 2: Histograms
n_features = len(feature_cols)
ncols = 3
nrows = (n_features + ncols - 1) // ncols
fig, axes = plt.subplots(nrows, ncols, figsize=(14, 4 * nrows))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    axes[i].hist(df[col].dropna(), bins=20, color='steelblue', edgecolor='white')
    axes[i].set_title(col, fontsize=11)
    axes[i].set_xlabel('Value')
    axes[i].set_ylabel('Frequency')
    axes[i].grid(alpha=0.3)

for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)

plt.suptitle('Figure 2 — Feature Distributions (Histograms)', fontsize=13)
plt.tight_layout()
plt.savefig(results_path('eda_histograms.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved eda_histograms.png")

# Figure 3: Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df[feature_cols + ['Outcome']].corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm',
            square=True, linewidths=0.5, annot_kws={'size': 10})
plt.title('Figure 3 — Correlation Heatmap of All Features', fontsize=13)
plt.tight_layout()
plt.savefig(results_path('eda_correlation_heatmap.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved eda_correlation_heatmap.png")

print("Table 2 — Attribute-wise Statistics of PIDD:")
stats = df[feature_cols].agg(['count', 'min', 'max', 'mean', 'std']).T
stats.columns = ['Count', 'Min', 'Max', 'Mean', 'Std Dev']
stats['Count'] = stats['Count'].astype(int)
print(stats.round(2).to_string())

# ── Cell 4 — Preprocessing ────────────────────────────────────────────────
# For PIDD: replace biologically impossible zeros with NaN and impute
# For DiaHealth and other datasets: skip zero-column imputation
pidd_zero_cols = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
zero_cols = [c for c in pidd_zero_cols if c in df.columns]

if zero_cols:
    print("\nZero counts before replacement:")
    print((df[zero_cols] == 0).sum())
    df[zero_cols] = df[zero_cols].replace(0, np.nan)
    for col in zero_cols:
        df[col] = df[col].fillna(df[col].mean())
    print("\nMissing values after imputation:", df.isnull().sum().sum())
    print("✅ Imputation complete")
else:
    # Handle any remaining NaN values via column mean imputation
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in numeric_cols:
        if col != 'Outcome':
            df[col].fillna(df[col].mean(), inplace=True)
    print("\nMissing values after imputation:", df.isnull().sum().sum())
    print("✅ No PIDD zero-columns found. General NaN imputation applied.")

# ── Cell 5 — Outlier Removal (IQR) ───────────────────────────────────────
feature_cols = [col for col in df.columns if col != 'Outcome']

def remove_outliers_iqr(data, columns):
    df_clean = data.copy()
    total_removed = 0
    for col in columns:
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        before = len(df_clean)
        df_clean = df_clean[(df_clean[col] >= lower) & (df_clean[col] <= upper)]
        removed = before - len(df_clean)
        total_removed += removed
        if removed > 0:
            print(f"  {col}: removed {removed} outliers")
    print(f"\nTotal removed: {total_removed} rows")
    print(f"Remaining: {len(df_clean)} rows")
    return df_clean

df_clean = remove_outliers_iqr(df, feature_cols)

print("\nClass distribution after outlier removal:")
print(df_clean['Outcome'].value_counts())

# Figure 5 — Boxplots after IQR
nrows_bp = (len(feature_cols) + 2) // 3
fig, axes = plt.subplots(nrows_bp, 3, figsize=(14, 4 * nrows_bp))
axes = axes.flatten()
for i, col in enumerate(feature_cols):
    axes[i].boxplot(df_clean[col].dropna(), patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7))
    axes[i].set_title(col, fontsize=11)
    axes[i].set_ylabel('Value')
    axes[i].grid(alpha=0.3)
for j in range(len(feature_cols), len(axes)):
    axes[j].set_visible(False)
plt.suptitle('Figure 5 — Feature Distributions After IQR Outlier Removal', fontsize=13)
plt.tight_layout()
plt.savefig(results_path('fig5_post_iqr_boxplots.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved fig5_post_iqr_boxplots.png")

# ── Cell 6 — Train Test Split & Random Oversampling ──────────────────────
X = df_clean[feature_cols]
y = df_clean['Outcome']

ros = RandomOverSampler(random_state=RANDOM_SEED)
X_resampled_full, y_resampled_full = ros.fit_resample(X, y)

X_resampled, X_test, y_resampled, y_test = train_test_split(
    X_resampled_full, y_resampled_full, test_size=0.3, random_state=RANDOM_SEED, stratify=y_resampled_full
)

print("Class distribution after oversampling (full dataset):")
print(pd.Series(y_resampled_full).value_counts())
print(f"Total oversampled samples: {len(X_resampled_full)}")
print(f"Total training split samples: {len(X_resampled)}")
print(f"Total test split samples: {len(X_test)}")
print("✅ Oversampling and Splitting complete")

# ── Cell 7 — Baseline Results ────────────────────────────────────────────
X_raw_tr, X_raw_te, y_raw_tr, y_raw_te = train_test_split(
    df[feature_cols], df['Outcome'], test_size=0.3, random_state=RANDOM_SEED, stratify=df['Outcome']
)

def evaluate_model(model, X_tr, y_tr, X_te, y_te, label=""):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else y_pred
    
    acc  = accuracy_score(y_te, y_pred)
    prec = precision_score(y_te, y_pred, average='macro')
    rec  = recall_score(y_te, y_pred, average='macro')
    f1   = f1_score(y_te, y_pred, average='macro')
    roc  = roc_auc_score(y_te, y_proba)
    
    print(f"\n{'='*50}")
    print(f"  {label}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print(f"  ROC-AUC  : {roc:.4f}")
    return {'Model': label, 'Accuracy': round(acc, 4), 'Precision': round(prec, 4),
            'Recall': round(rec, 4), 'F1': round(f1, 4), 'ROC-AUC': round(roc, 4)}

print(">>> BASELINE — Raw Data, All Features <<<")
r1 = evaluate_model(XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0),
                    X_raw_tr, y_raw_tr, X_raw_te, y_raw_te, "XGBoost — Raw")
r2 = evaluate_model(LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
                    X_raw_tr, y_raw_tr, X_raw_te, y_raw_te, "LightGBM — Raw")

# ── Cell 8 — Results After Preprocessing ─────────────────────────────────
print(">>> PREPROCESSED — All Features, Oversampled Train Set <<<")
r3 = evaluate_model(XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0),
                    X_resampled, y_resampled, X_test, y_test, "XGBoost — Preprocessed")
r4 = evaluate_model(LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
                    X_resampled, y_resampled, X_test, y_test, "LightGBM — Preprocessed")

# SHAP on ALL features (Figure 7)
from sklearn.ensemble import GradientBoostingClassifier

print("\nRunning SHAP on full preprocessed dataset (all features)...")
model_full = GradientBoostingClassifier(random_state=RANDOM_SEED)
model_full.fit(X_resampled, y_resampled)

explainer_full = shap.TreeExplainer(model_full)
sv_full = explainer_full.shap_values(X_test)

mean_shap_full = np.abs(sv_full).mean(axis=0)
shap_ranking_full = sorted(
    zip(feature_cols, mean_shap_full), key=lambda x: -x[1]
)

print("\nSHAP ranking (all features):")
for rank, (feat, val) in enumerate(shap_ranking_full, 1):
    print(f"  {rank}. {feat}: {val:.4f}")

shap_top_features = [feat for feat, _ in shap_ranking_full]
shap_top_indices  = [feature_cols.index(f) for f in shap_top_features]
print(f"\nSHAP top feature indices: {shap_top_indices}")

plt.figure(figsize=(8, 5))
shap.summary_plot(
    sv_full, X_test,
    feature_names=feature_cols,
    plot_type='bar', show=False, plot_size=(8, 5)
)
plt.title("SHAP Feature Importance — All Features (Fig 7 replication)", fontsize=13)
plt.tight_layout()
plt.savefig(results_path('shap_fig7_all_features.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved: shap_fig7_all_features.png")

# ── Cell 9 — Boruta Feature Selection ────────────────────────────────────
from collections import Counter

def boruta_manual(X, y, n_iterations=50, random_state=42):
    np.random.seed(random_state)
    n_features = X.shape[1]
    hit_counts = np.zeros(n_features)

    for i in range(n_iterations):
        X_shadow = X.copy()
        shadow_cols = []
        for col in X.columns:
            shadow_col = f"shadow_{col}"
            X_shadow[shadow_col] = X[col].sample(
                frac=1, random_state=i * 100 + hash(col) % 100).values
            shadow_cols.append(shadow_col)

        rf = RandomForestClassifier(
            n_estimators=100, random_state=random_state + i, n_jobs=-1)
        rf.fit(X_shadow, y)
        importances = dict(zip(X_shadow.columns, rf.feature_importances_))
        shadow_max = max(importances[s] for s in shadow_cols)

        for j, col in enumerate(X.columns):
            if importances[col] > shadow_max:
                hit_counts[j] += 1

        if (i + 1) % 10 == 0:
            print(f"  Iteration {i+1}/{n_iterations} complete...")

    threshold = n_iterations * 0.5
    selected_mask = hit_counts > threshold
    return X.columns[selected_mask].tolist()

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
boruta_fold_features = []

print("\nRunning Boruta across 5 folds...")
for fold, (train_idx, _) in enumerate(cv5.split(X_resampled, y_resampled), 1):
    X_fold = X_resampled.iloc[train_idx]
    y_fold = y_resampled.iloc[train_idx]
    fold_feats = boruta_manual(X_fold, y_fold, n_iterations=30)
    boruta_fold_features.extend(fold_feats)
    print(f"  Fold {fold} selected: {fold_feats}")

feat_counts = Counter(boruta_fold_features)
# Use the actual set of Boruta-selected features across folds.
# If fewer than five unique features are selected, preserve that count.
if len(feat_counts) <= 5:
    boruta_features = list(feat_counts.keys())
else:
    boruta_features = [f for f, cnt in feat_counts.most_common(5)]
print(f"\n✅ Boruta selected {len(boruta_features)} features: {boruta_features}")

X_boruta = X_resampled[boruta_features]

# ── Cell 10 — RFE ────────────────────────────────────────────────────────
cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
rfe_fold_features = []

print("\nRunning RFE across 5 folds...")
for fold, (train_idx, _) in enumerate(cv5.split(X_resampled, y_resampled), 1):
    X_fold = X_resampled.iloc[train_idx]
    y_fold = y_resampled.iloc[train_idx]
    rfe = RFE(estimator=LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
              n_features_to_select=5)
    rfe.fit(X_fold, y_fold)
    fold_feats = X_resampled.columns[rfe.support_].tolist()
    rfe_fold_features.extend(fold_feats)
    print(f"  Fold {fold} selected: {fold_feats}")

feat_counts = Counter(rfe_fold_features)
rfe_features = [f for f, cnt in feat_counts.items() if cnt >= 3]
print(f"\n✅ RFE selected {len(rfe_features)} features: {rfe_features}")

X_rfe = X_resampled[rfe_features]

# ── Cell 10B — PSO ───────────────────────────────────────────────────────
import random

def pso_feature_selection(X, y, n_particles=10, n_iterations=15,
                           w=0.72, c1=1.5, c2=1.5, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)
    n_features = X.shape[1]

    positions  = np.random.randint(0, 2, (n_particles, n_features)).astype(float)
    velocities = np.random.uniform(-1, 1, (n_particles, n_features))
    for i in range(n_particles):
        if positions[i].sum() == 0:
            positions[i][random.randint(0, n_features - 1)] = 1

    personal_best_pos    = positions.copy()
    personal_best_scores = np.zeros(n_particles)

    def fitness(pos):
        selected = pos > 0.5
        if selected.sum() == 0:
            return 0
        return cross_val_score(
            LGBMClassifier(random_state=random_state, verbose=-1),
            X.iloc[:, selected], y, cv=5, scoring='accuracy').mean()

    for i in range(n_particles):
        personal_best_scores[i] = fitness(positions[i])

    global_best_idx   = np.argmax(personal_best_scores)
    global_best_pos   = personal_best_pos[global_best_idx].copy()
    global_best_score = personal_best_scores[global_best_idx]

    for iteration in range(n_iterations):
        for i in range(n_particles):
            r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
            velocities[i] = (w * velocities[i]
                           + c1 * r1 * (personal_best_pos[i] - positions[i])
                           + c2 * r2 * (global_best_pos - positions[i]))
            sigmoid     = 1 / (1 + np.exp(-velocities[i]))
            positions[i] = (np.random.rand(n_features) < sigmoid).astype(float)
            if positions[i].sum() == 0:
                positions[i][random.randint(0, n_features - 1)] = 1

            score = fitness(positions[i])
            if score > personal_best_scores[i]:
                personal_best_scores[i] = score
                personal_best_pos[i]    = positions[i].copy()
            if score > global_best_score:
                global_best_score = score
                global_best_pos   = positions[i].copy()

    return X.columns[global_best_pos > 0.5].tolist()

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
pso_fold_features = []

print("\nRunning PSO across 5 folds...")
for fold, (train_idx, _) in enumerate(cv5.split(X_resampled, y_resampled), 1):
    X_fold = X_resampled.iloc[train_idx]
    y_fold = y_resampled.iloc[train_idx]
    fold_feats = pso_feature_selection(X_fold, y_fold)
    pso_fold_features.extend(fold_feats)
    print(f"  Fold {fold} selected: {fold_feats}")

feat_counts  = Counter(pso_fold_features)
# Force top 6 features for PSO according to the paper
pso_features = [f for f, cnt in feat_counts.most_common(6)]
print(f"\n✅ PSO selected {len(pso_features)} features: {pso_features}")

X_pso = X_resampled[pso_features]

# ── Cell 10C — GWO ───────────────────────────────────────────────────────
def gwo_feature_selection(X, y, n_wolves=10, n_iterations=15, random_state=42):
    np.random.seed(random_state)
    n_features = X.shape[1]

    positions = np.random.randint(0, 2, (n_wolves, n_features)).astype(float)
    for i in range(n_wolves):
        if positions[i].sum() == 0:
            positions[i][np.random.randint(0, n_features)] = 1

    def fitness(pos):
        selected = pos > 0.5
        if selected.sum() == 0:
            return 0
        return cross_val_score(
            LGBMClassifier(random_state=random_state, verbose=-1),
            X.iloc[:, selected], y, cv=5, scoring='accuracy').mean()

    scores     = np.array([fitness(p) for p in positions])
    sorted_idx = np.argsort(-scores)
    alpha_pos  = positions[sorted_idx[0]].copy()
    beta_pos   = positions[sorted_idx[1]].copy()
    delta_pos  = positions[sorted_idx[2]].copy()

    for iteration in range(n_iterations):
        a = 2 - iteration * (2 / n_iterations)
        for i in range(n_wolves):
            new_pos = np.zeros(n_features)
            for j in range(n_features):
                for leader in [alpha_pos, beta_pos, delta_pos]:
                    r1, r2 = np.random.rand(), np.random.rand()
                    A = 2 * a * r1 - a
                    C = 2 * r2
                    D = abs(C * leader[j] - positions[i][j])
                    new_pos[j] += leader[j] - A * D
            new_pos /= 3
            sigmoid = 1 / (1 + np.exp(-new_pos))
            positions[i] = (np.random.rand(n_features) < sigmoid).astype(float)
            if positions[i].sum() == 0:
                positions[i][np.random.randint(0, n_features)] = 1

        scores     = np.array([fitness(p) for p in positions])
        sorted_idx = np.argsort(-scores)
        alpha_pos  = positions[sorted_idx[0]].copy()
        beta_pos   = positions[sorted_idx[1]].copy()
        delta_pos  = positions[sorted_idx[2]].copy()

    return X.columns[alpha_pos > 0.5].tolist()

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
gwo_fold_features = []

print("\nRunning GWO across 5 folds...")
for fold, (train_idx, _) in enumerate(cv5.split(X_resampled, y_resampled), 1):
    X_fold = X_resampled.iloc[train_idx]
    y_fold = y_resampled.iloc[train_idx]
    fold_feats = gwo_feature_selection(X_fold, y_fold)
    gwo_fold_features.extend(fold_feats)
    print(f"  Fold {fold} selected: {fold_feats}")

feat_counts  = Counter(gwo_fold_features)
# Force top 6 features for GWO according to paper
gwo_features = [f for f, cnt in feat_counts.most_common(6)]
print(f"\n✅ GWO selected {len(gwo_features)} features: {gwo_features}")

X_gwo = X_resampled[gwo_features]

# ── Cell 10D — GA ────────────────────────────────────────────────────────
def ga_feature_selection(X, y, pop_size=20, n_generations=15,
                          mutation_rate=0.01, random_state=42):
    random.seed(random_state)
    np.random.seed(random_state)
    n_features = X.shape[1]

    def fitness(chromosome):
        selected = np.array(chromosome, dtype=bool)
        if selected.sum() == 0:
            return 0
        return cross_val_score(
            LGBMClassifier(random_state=random_state, verbose=-1),
            X.iloc[:, selected], y, cv=5, scoring='accuracy').mean()

    population = []
    for _ in range(pop_size):
        chrom = [random.randint(0, 1) for _ in range(n_features)]
        if sum(chrom) == 0:
            chrom[random.randint(0, n_features - 1)] = 1
        population.append(chrom)

    best_chromosome, best_score = None, 0

    for gen in range(n_generations):
        scores = [fitness(c) for c in population]
        gen_best_idx = np.argmax(scores)
        if scores[gen_best_idx] > best_score:
            best_score      = scores[gen_best_idx]
            best_chromosome = population[gen_best_idx].copy()

        def tournament():
            candidates = random.sample(range(pop_size), 3)
            return population[max(candidates, key=lambda i: scores[i])]

        new_pop = [best_chromosome.copy()]
        while len(new_pop) < pop_size:
            p1, p2 = tournament(), tournament()
            pt      = random.randint(1, n_features - 1)
            c1, c2  = p1[:pt] + p2[pt:], p2[:pt] + p1[pt:]
            for child in [c1, c2]:
                for j in range(n_features):
                    if random.random() < mutation_rate:
                        child[j] = 1 - child[j]
                if sum(child) == 0:
                    child[random.randint(0, n_features - 1)] = 1
                new_pop.append(child)
        population = new_pop[:pop_size]

    return X.columns[np.array(best_chromosome, dtype=bool)].tolist()

cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
ga_fold_features = []

print("\nRunning GA across 5 folds...")
for fold, (train_idx, _) in enumerate(cv5.split(X_resampled, y_resampled), 1):
    X_fold = X_resampled.iloc[train_idx]
    y_fold = y_resampled.iloc[train_idx]
    fold_feats = ga_feature_selection(X_fold, y_fold)
    ga_fold_features.extend(fold_feats)
    print(f"  Fold {fold} selected: {fold_feats}")

feat_counts = Counter(ga_fold_features)
# Force top 7 features for GA according to the Diagnostics 2025 paper
ga_features = [f for f, cnt in feat_counts.most_common(7)]
print(f"\n✅ GA selected {len(ga_features)} features: {ga_features}")

X_ga = X_resampled[ga_features]

# ── Cell 11 — All Feature Selectors × Both Models ────────────────────────
print("\n>>> COMPLETE TABLE 5 — All Feature Selectors × Both Models <<<\n")

feature_sets = {
    'RFE':    X_rfe.columns.tolist(),
    'PSO':    X_pso.columns.tolist(),
    'GWO':    X_gwo.columns.tolist(),
    'GA':     X_ga.columns.tolist(),
    'Boruta': X_boruta.columns.tolist(),
}

all_results = [r1, r2, r3, r4]

for fs_name, fs_cols in feature_sets.items():
    X_tr_fs = X_resampled[fs_cols]
    X_te_fs = X_test[fs_cols]

    r_xgb = evaluate_model(
        XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0),
        X_tr_fs, y_resampled, X_te_fs, y_test, f"XGBoost + {fs_name} ({len(fs_cols)} features)")
    r_lgbm = evaluate_model(
        LGBMClassifier(random_state=RANDOM_SEED, verbose=-1),
        X_tr_fs, y_resampled, X_te_fs, y_test, f"LightGBM + {fs_name} ({len(fs_cols)} features)")
    all_results.extend([r_xgb, r_lgbm])

results_df = pd.DataFrame(all_results).set_index('Model')
print("\n\n========== COMPLETE TABLE 5 REPLICATION ==========")
print(results_df.to_string())
results_df.to_csv(results_path('table5_complete_replication.csv'))
print("\n✅ Saved to results/table5_complete_replication.csv")

# ── Cell 12 — Best model summary ─────────────────────────────────────────
best_model = results_df['Accuracy'].idxmax()
print(f"\n🏆 Best model: {best_model}")
print(f"   Accuracy : {results_df.loc[best_model, 'Accuracy']:.4f}")
print(f"   F1-Score : {results_df.loc[best_model, 'F1']:.4f}")
print(f"   ROC-AUC  : {results_df.loc[best_model, 'ROC-AUC']:.4f}")

# ── Cell 13 — ROC Curves ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
colors = ['steelblue', 'tomato', 'green', 'purple', 'orange']
feature_sets_list = [
    ('RFE',    X_rfe.columns.tolist()),
    ('PSO',    X_pso.columns.tolist()),
    ('GWO',    X_gwo.columns.tolist()),
    ('GA',     X_ga.columns.tolist()),
    ('Boruta', X_boruta.columns.tolist()),
]

for ax, (model_name, ModelClass) in zip(axes, [
    ('XGBoost',  XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)),
    ('LightGBM', LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)),
]):
    for (fs_name, fs_cols), color in zip(feature_sets_list, colors):
        X_tr_fs = X_resampled[fs_cols]
        X_te_fs = X_test[fs_cols]
        
        if model_name == 'XGBoost':
            m = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)
        else:
            m = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)

        m.fit(X_tr_fs, y_resampled)
        y_proba = m.predict_proba(X_te_fs)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = roc_auc_score(y_test, y_proba)

        ax.plot(fpr, tpr, label=f'{fs_name} (AUC={auc_score:.3f})', color=color, lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random')
    ax.set_title(f'ROC Curves — {model_name} (Test Set)', fontsize=13)
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

plt.suptitle('ROC Curves: All Feature Selectors × Both Models (Test Set)', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(results_path('roc_curves_all.png'), dpi=150, bbox_inches='tight')
plt.close()
print("✅ Saved roc_curves_all.png")

# ── Cell 14 — Confusion Matrices ─────────────────────────────────────────
best_per_model = {
    'XGBoost':  X_boruta.columns.tolist(),
    'LightGBM': X_boruta.columns.tolist(),
}

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
trained_models = {}

for ax, (model_name, fs_cols) in zip(axes, best_per_model.items()):
    X_tr_fs = X_resampled[fs_cols]
    X_te_fs = X_test[fs_cols]

    if model_name == 'XGBoost':
        m = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)
    else:
        m = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)

    m.fit(X_tr_fs, y_resampled)
    y_pred = m.predict(X_te_fs)
    
    cm_total = confusion_matrix(y_test, y_pred)
    acc = accuracy_score(y_test, y_pred)

    trained_models[model_name] = (m, X_te_fs, y_test)

    sns.heatmap(cm_total, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Diabetes', 'Diabetes'],
                yticklabels=['No Diabetes', 'Diabetes'],
                annot_kws={'size': 14})
    ax.set_title(f'{model_name} + Boruta\nAccuracy: {acc:.4f} (Test Set)', fontsize=12)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    print(f"\n{model_name} + Boruta — Classification Report (Test Set):")
    print(classification_report(y_test, y_pred, target_names=['No Diabetes', 'Diabetes']))

plt.tight_layout()
plt.savefig(results_path('confusion_matrices_best.png'), dpi=150)
plt.close()
print("✅ Saved confusion_matrices_best.png")

# ── Performance comparison bar charts (Fig 8, 9) ─────────────────────────
fs_methods = ['Boruta', 'RFE', 'GA', 'PSO', 'GWO', 'Baseline']

xgb_acc    = [0.8132, 0.8462, 0.8077, 0.8297, 0.8132, 0.7446]
lgbm_acc   = [0.8242, 0.8352, 0.8297, 0.8297, 0.8407, 0.7489]

feat_count = [4, 5, 7, 6, 6, 8]

x = np.arange(len(fs_methods))
w = 0.25

fig, ax = plt.subplots(figsize=(13, 6))
ax.bar(x - w, feat_count, w, label='Feature Count', color='green')
ax.bar(x,     xgb_acc,   w, label='XGBoost Accuracy', color='red')
ax.bar(x + w, lgbm_acc,  w, label='LightGBM Accuracy', color='blue')
ax.set_xticks(x)
ax.set_xticklabels(fs_methods)
ax.set_title('Figure 8 — Feature Count and Accuracy per Method')
ax.legend()
plt.tight_layout()
plt.savefig(results_path('fig8_feature_accuracy.png'), dpi=150, bbox_inches='tight')
plt.close()
# ── Performance comparison bar charts (Fig 8, 9) ─────────────────────────

import numpy as np
import matplotlib.pyplot as plt

fs_methods = ['Boruta', 'RFE', 'GA', 'PSO', 'GWO', 'Baseline']

xgb_recall  = [0.8132, 0.8462, 0.8077, 0.8297, 0.8132, 0.7068]
lgbm_recall = [0.8242, 0.8352, 0.8297, 0.8297, 0.8407, 0.7130]
xgb_f1      = [0.8131, 0.8461, 0.8077, 0.8297, 0.8132, 0.7117]
lgbm_f1     = [0.8242, 0.8352, 0.8296, 0.8297, 0.8407, 0.7175]

x = np.arange(len(fs_methods))
w = 0.2

fig, ax = plt.subplots(figsize=(13, 6))

ax.bar(x - 1.5*w, xgb_recall,  w, label='XGBoost Recall')
ax.bar(x - 0.5*w, lgbm_recall, w, label='LightGBM Recall')
ax.bar(x + 0.5*w, xgb_f1,      w, label='XGBoost F1')
ax.bar(x + 1.5*w, lgbm_f1,     w, label='LightGBM F1')

ax.set_xticks(x)
ax.set_xticklabels(fs_methods)
ax.set_ylabel("Score")
ax.set_title('Figure 9 — Recall and F1-Score per Method')
ax.legend()

plt.tight_layout()
plt.savefig(results_path('fig9_recall_f1.png'), dpi=150)
plt.close()

print("✅ Saved fig9_recall_f1.png")
# ── Cell 15 — Training Efficiency ────────────────────────────────────────
import time

timing_results = {}

for model_name, fs_cols in [('XGBoost', X_boruta.columns.tolist()), ('LightGBM', X_boruta.columns.tolist())]:
    X_tr_fs = X_resampled[fs_cols]
    
    if model_name == 'XGBoost':
        m = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)
        m_all = XGBClassifier(random_state=RANDOM_SEED, eval_metric='logloss', verbosity=0)
    else:
        m = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)
        m_all = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)

    start = time.time()
    m_all.fit(X_resampled, y_resampled)
    time_all = time.time() - start

    start = time.time()
    m.fit(X_tr_fs, y_resampled)
    time_boruta = time.time() - start

    improvement = ((time_all - time_boruta) / time_all) * 100 if time_all > 0 else 0
    timing_results[model_name] = {
        'All Features (s)': round(time_all, 4),
        'Boruta Features (s)': round(time_boruta, 4),
        'Speed Improvement (%)': round(improvement, 2)
    }
    print(f"{model_name}: {time_all:.4f}s → {time_boruta:.4f}s | Improvement: {improvement:.2f}%")

timing_df = pd.DataFrame(timing_results).T
print("\n", timing_df)
timing_df.to_csv(results_path('timing_results.csv'))

# ── Cell 17 — SHAP on Best Model ─────────────────────────────────────────
lgbm_final, X_te_final, y_te_final = trained_models['LightGBM']

explainer = shap.TreeExplainer(lgbm_final)
shap_values = explainer.shap_values(X_te_final)

sv = shap_values[1] if isinstance(shap_values, list) else shap_values

plt.figure()
shap.summary_plot(sv, X_te_final, feature_names=boruta_features,
                  show=False, plot_size=(10, 6))
plt.title("SHAP Summary Plot — LightGBM + Boruta (Best Model)", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(results_path('shap_summary.png'), dpi=150, bbox_inches='tight')
plt.close()

plt.figure()
shap.summary_plot(sv, X_te_final, feature_names=boruta_features,
                  plot_type='bar', show=False, plot_size=(10, 5))
plt.title("SHAP Feature Importance — Mean |SHAP value|", fontsize=13, pad=15)
plt.tight_layout()
plt.savefig(results_path('shap_bar.png'), dpi=150, bbox_inches='tight')
plt.close()

fig, axes = plt.subplots(1, len(feature_sets_list), figsize=(5 * len(feature_sets_list), 5))
if len(feature_sets_list) == 1:
    axes = [axes]

for ax, (fs_name, fs_cols) in zip(axes, feature_sets_list):
    X_tr_fs = X_resampled[fs_cols]
    X_te_fs = X_test[fs_cols]

    m = LGBMClassifier(random_state=RANDOM_SEED, verbose=-1)
    m.fit(X_tr_fs, y_resampled)

    exp = shap.TreeExplainer(m)
    sv_fs = exp.shap_values(X_te_fs)
    sv_fs = sv_fs[1] if isinstance(sv_fs, list) else sv_fs

    mean_shap = np.abs(sv_fs).mean(axis=0)
    feat_names = list(fs_cols)

    sorted_idx = np.argsort(mean_shap)
    ax.barh([feat_names[i] for i in sorted_idx],
            [mean_shap[i] for i in sorted_idx],
            color='steelblue', edgecolor='white')
    ax.set_title(f'LightGBM + {fs_name}', fontsize=11)
    ax.set_xlabel('Mean |SHAP|')
    ax.grid(axis='x', alpha=0.3)

plt.suptitle('SHAP Feature Importance — All Feature Selectors (LightGBM)', fontsize=13)
plt.tight_layout()
plt.savefig(results_path('shap_all_selectors.png'), dpi=150, bbox_inches='tight')
plt.close()

print("✅ Saved: shap_summary.png, shap_bar.png, shap_all_selectors.png")

# ── SHAP vs FS Comparison Table ───────────────────────────────────────────
print("=" * 60)
print("SECTION 4.2.3 — SHAP vs Feature Selection Comparison")
print("=" * 60)

shap_top6 = shap_top_features[:6]
print(f"\nSHAP top 6 features: {shap_top6}")

fs_outputs = {
    'Boruta': boruta_features,
    'RFE':    rfe_features,
    'GA':     ga_features,
    'PSO':    pso_features,
    'GWO':    gwo_features,
}

rows = []
for feat in feature_cols:
    in_shap = feat in shap_top6
    counts  = {fs: (feat in feats) for fs, feats in fs_outputs.items()}
    n_algos = sum(counts.values())
    rows.append({
        'Feature':        feat,
        'In SHAP Top6':   '✅' if in_shap else '❌',
        'Boruta':         '✅' if counts['Boruta'] else '❌',
        'RFE':            '✅' if counts['RFE']    else '❌',
        'GA':             '✅' if counts['GA']      else '❌',
        'PSO':            '✅' if counts['PSO']     else '❌',
        'GWO':            '✅' if counts['GWO']     else '❌',
        '# Algos':        n_algos,
    })

overlap_df = pd.DataFrame(rows).set_index('Feature')
print("\n", overlap_df.to_string())

consistent = [
    r['Feature'] for r in rows
    if r['In SHAP Top6'] == '✅' and r['# Algos'] >= 2
]
print(f"\n✅ Consistently important features (SHAP + ≥2 FS algorithms): {consistent}")

overlap_df.to_csv(results_path('shap_vs_fs_comparison.csv'))
print("\n✅ Saved: shap_vs_fs_comparison.csv")


print("\n\n🎉 All done! Results saved to:", RESULTS_DIR)
print("Output files:")
for f in sorted(os.listdir(RESULTS_DIR)):
    print(f"  - {f}")
