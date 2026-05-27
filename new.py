import warnings
warnings.filterwarnings('ignore')

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests

from scipy.io import arff
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, ConfusionMatrixDisplay
)

pd.set_option('future.no_silent_downcasting', True)


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1: LOAD DATASETS
# ══════════════════════════════════════════════════════════════════════════════

def load_arff(path: str) -> pd.DataFrame:
    """Load an .arff file and return a clean DataFrame with int labels."""
    data, _ = arff.loadarff(path)
    df = pd.DataFrame(data)
    df['Result'] = df['Result'].astype(str).str.strip()
    df['Result'] = df['Result'].replace({
        "b'1'":  1,  "b'-1'": -1,
        "1":     1,  "-1":    -1,
        "1.0":   1,  "-1.0":  -1,
    })
    df = df.dropna(subset=['Result'])
    df['Result'] = df['Result'].astype(int)
    return df


print("=" * 60)
print(" LOADING DATASETS")
print("=" * 60)

# Primary UCI dataset
print("\n[1] Loading Training Dataset.arff ...")
df_train = load_arff('Training Dataset.arff')
print(f"    Shape: {df_train.shape}")
print(f"    Labels: {df_train['Result'].value_counts().to_dict()}")

# Older UCI dataset (optional — merge for more training data)
try:
    print("\n[2] Loading .old.arff ...")
    df_old = load_arff('.old.arff')
    print(f"    Shape: {df_old.shape}")
    print(f"    Labels: {df_old['Result'].value_counts().to_dict()}")

    # Only merge if columns match
    if list(df_old.columns) == list(df_train.columns):
        df_combined = pd.concat([df_train, df_old], ignore_index=True).drop_duplicates()
        print(f"    Merged → {df_combined.shape}")
    else:
        print("    Column mismatch — using only Training Dataset.arff")
        df_combined = df_train
except FileNotFoundError:
    print("    .old.arff not found — skipping")
    df_combined = df_train

# Alexa (loaded for reference — used at inference time, not training)
try:
    print("\n[3] Loading Alexa Top 1M ...")
    df_alexa = pd.read_csv('top-1m.csv', header=None, names=['rank', 'domain'], nrows=100_000)
    print(f"    Sample: {df_alexa['domain'].head(3).tolist()}")
except FileNotFoundError:
    print("    top-1m.csv not found — Alexa feature will be inactive at inference")
    df_alexa = None

# PhishTank (optional — download for EDA / future feature engineering)
print("\n[4] Attempting PhishTank download ...")
try:
    url_pt = "http://data.phishtank.com/data/online-valid.csv"
    r = requests.get(url_pt, headers={"User-Agent": "phishtank/student-project"}, timeout=20)
    with open('phishtank.csv', 'wb') as f:
        f.write(r.content)
    df_phishtank = pd.read_csv('phishtank.csv')
    print(f"    Downloaded — shape: {df_phishtank.shape}")
    print(f"    Columns: {df_phishtank.columns.tolist()}")
except Exception as e:
    print(f"    Skipped ({e})")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2: PREPARE DATA
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" PREPARING DATA")
print("=" * 60)

df = df_combined.copy()
X = df.drop('Result', axis=1)
y = df['Result']

print(f"\nFinal dataset: {df.shape[0]:,} samples, {X.shape[1]} features")
print(f"Class balance:")
vc = y.value_counts()
for cls, cnt in vc.items():
    label = "Legitimate" if cls == 1 else "Phishing"
    pct = cnt / len(y) * 100
    print(f"   {cls:2d} ({label}): {cnt:,} ({pct:.1f}%)")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nTrain: {len(X_train):,}  |  Test: {len(X_test):,}")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3: TRAIN & EVALUATE MODELS
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" MODEL TRAINING")
print("=" * 60)

models = {
    "Decision Tree":     DecisionTreeClassifier(random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, random_state=42),
    "SVM":               SVC(kernel='rbf', probability=True, random_state=42),
}

results      = []
trained_models = {}
cv            = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for name, model in models.items():
    print(f"\n▶ {name}")

    # Cross-validation (5-fold) on training set
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='accuracy', n_jobs=1)
    print(f"  CV Accuracy: {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # Full train on X_train, evaluate on X_test
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)  * 100
    prec = precision_score(y_test, y_pred) * 100
    rec  = recall_score(y_test, y_pred)   * 100
    f1   = f1_score(y_test, y_pred)       * 100

    results.append({
        "Model":        name,
        "CV Accuracy":  round(cv_scores.mean() * 100, 2),
        "Accuracy":     round(acc,  2),
        "Precision":    round(prec, 2),
        "Recall":       round(rec,  2),
        "F1-Score":     round(f1,   2),
    })
    trained_models[name] = model

    print(f"  Test  — Acc: {acc:.2f}%  Prec: {prec:.2f}%  Rec: {rec:.2f}%  F1: {f1:.2f}%")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4: COMPARE & SELECT BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print(" RESULTS SUMMARY")
print("=" * 60)

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))
df_results.to_csv('model_results.csv', index=False)
print("\n✓ Saved → model_results.csv")

# Best model = highest test accuracy
best_row   = df_results.loc[df_results['Accuracy'].idxmax()]
best_name  = best_row['Model']
best_model = trained_models[best_name]
print(f"\n★ Best Model: {best_name}  ({best_row['Accuracy']}% accuracy)")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5: SAVE BEST MODEL
# ══════════════════════════════════════════════════════════════════════════════

with open('phishing_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)
print(f"✓ Saved → phishing_model.pkl  (features: {best_model.n_features_in_})")


# ══════════════════════════════════════════════════════════════════════════════
# SECTION 6: CHARTS
# ══════════════════════════════════════════════════════════════════════════════

# 6a: Model comparison bar chart
fig, ax = plt.subplots(figsize=(11, 6))
metrics  = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
x        = np.arange(len(df_results))
width    = 0.18
colors   = ['#6366f1', '#22c55e', '#f59e0b', '#ef4444']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    bars = ax.bar(x + i * width, df_results[metric], width, label=metric, color=color, alpha=0.88)
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                f"{bar.get_height():.1f}", ha='center', va='bottom', fontsize=7.5)

ax.set_xlabel('Model', fontsize=11)
ax.set_ylabel('Score (%)', fontsize=11)
ax.set_title('Phishing Detection — ML Model Comparison', fontsize=13, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(df_results['Model'], rotation=10)
ax.set_ylim(80, 102)
ax.legend(loc='lower right')
ax.grid(axis='y', linestyle='--', alpha=0.4)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.close()
print("✓ Saved → model_comparison.png")

# 6b: Confusion matrix for best model
y_best_pred = best_model.predict(X_test)
cm = confusion_matrix(y_test, y_best_pred, labels=[-1, 1])
fig2, ax2 = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(cm, display_labels=['Phishing', 'Legitimate'])
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title(f'Confusion Matrix — {best_name}', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.close()
print("✓ Saved → confusion_matrix.png")

# 6c: Feature importance (if tree-based)
if hasattr(best_model, 'feature_importances_'):
    feature_names = list(X.columns)
    importances   = best_model.feature_importances_
    indices       = np.argsort(importances)[::-1][:15]  # top 15

    fig3, ax3 = plt.subplots(figsize=(10, 5))
    ax3.bar(range(len(indices)), importances[indices], color='#6366f1', alpha=0.85)
    ax3.set_xticks(range(len(indices)))
    ax3.set_xticklabels([feature_names[i] for i in indices], rotation=45, ha='right', fontsize=8)
    ax3.set_title(f'Top 15 Feature Importances — {best_name}', fontweight='bold')
    ax3.set_ylabel('Importance')
    ax3.grid(axis='y', linestyle='--', alpha=0.4)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.close()
    print("✓ Saved → feature_importance.png")

print("\n" + "=" * 60)
print(" DONE — phishing_model.pkl is ready for app.py")
print("=" * 60)
