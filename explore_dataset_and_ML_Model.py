import pandas as pd
from scipy.io import arff
import requests

# ── Load Training Dataset (UCI) ────
data1, meta1 = arff.loadarff('Training Dataset.arff')
df_train = pd.DataFrame(data1)

# ── Load Old Dataset (UCI) ──────
data2, meta2 = arff.loadarff('.old.arff')
df_old = pd.DataFrame(data2)

# ── Load Alexa Top 1M ────────
df_alexa = pd.read_csv('top-1m.csv',
                        header=None,
                        names=['rank', 'domain'])

# ── Download PhishTank CSV ───────────────
print("Downloading PhishTank dataset...")
try:
    url = "http://data.phishtank.com/data/online-valid.csv"
    headers = {"User-Agent": "phishtank/student-project"}
    response = requests.get(url, headers=headers, timeout=30)
    
    with open('phishtank.csv', 'wb') as f:
        f.write(response.content)
    
    df_phishtank = pd.read_csv('phishtank.csv')
    print("PhishTank downloaded successfully!")
    print("Shape:", df_phishtank.shape)
    print("Columns:", df_phishtank.columns.tolist())
    print("Sample URLs:")
    print(df_phishtank['url'].head(5))

except Exception as e:
    print(f" PhishTank download failed: {e}")
    print("→ Try manually downloading from: http://data.phishtank.com/data/online-valid.csv")

# ── Explore UCI Training Dataset ─────────
print("\n=== UCI TRAINING DATASET ===")
print("Shape:", df_train.shape)
print("Class distribution:")
print(df_train.iloc[:, -1].value_counts())

# ── Explore Old Dataset ──────────────────
print("\n=== UCI OLD DATASET ===")
print("Shape:", df_old.shape)
print("Class distribution:")
print(df_old.iloc[:, -1].value_counts())

# ── Explore Alexa ────────────────────────
print("\n=== ALEXA TOP 1M ===")
print("Shape:", df_alexa.shape)
print("Sample domains:")
print(df_alexa.head(5))

print("\n All 3 datasets loaded successfully!")

# ══════════════════════════════════════════
# Task 3 MODEL TRAINING
# ══════════════════════════════════════════

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

# ── Fix labels from b'1' format ────
print("\nPreparing data for training...")
# Fix labels properly
pd.set_option('future.no_silent_downcasting', True)
df_train['Result'] = df_train['Result'].astype(str)

# Check what values actually exist
print("Unique label values:", df_train['Result'].unique())

# Clean and map all possible formats
df_train['Result'] = df_train['Result'].str.strip()
df_train['Result'] = df_train['Result'].replace({
    "b'1'":  1,
    "b'-1'": -1,
    "1":     1,
    "-1":    -1,
    "1.0":   1,
    "-1.0":  -1
})

# Drop any rows that didnt match
df_train = df_train.dropna(subset=['Result'])
df_train['Result'] = df_train['Result'].astype(int)

print("Clean dataset shape:", df_train.shape)
print("Label distribution:", df_train['Result'].value_counts().to_dict())

# ── Features and Label ─────
X = df_train.drop('Result', axis=1)
y = df_train['Result']

# ── Train/Test Split ─────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples:  {len(X_test)}")

# ── Train 4 Models ────
models = {
    "Decision Tree":     DecisionTreeClassifier(random_state=42),
    "Random Forest":     RandomForestClassifier(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    "SVM":               SVC(random_state=42)
}

results = []

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec  = recall_score(y_test, y_pred) * 100
    f1   = f1_score(y_test, y_pred) * 100

    results.append({
        "Model": name,
        "Accuracy":  round(acc, 2),
        "Precision": round(prec, 2),
        "Recall":    round(rec, 2),
        "F1-Score":  round(f1, 2)
    })

    print(f"  Accuracy:  {acc:.2f}%")
    print(f"  Precision: {prec:.2f}%")
    print(f"  Recall:    {rec:.2f}%")
    print(f"  F1-Score:  {f1:.2f}%")

# ── Final Comparison Table ────
print("\n====== FINAL COMPARISON ======")
df_results = pd.DataFrame(results)
print(df_results.to_string(index=False))
df_results.to_csv('model_results.csv', index=False)
print("\nSaved to model_results.csv")

# ── Chart ──────
df_results.set_index('Model')[['Accuracy','Precision','Recall','F1-Score']].plot(
    kind='bar', figsize=(10,6), colormap='coolwarm'
)
plt.title('ML Model Comparison: Phishing Detection')
plt.ylabel('Score (%)')
plt.xticks(rotation=15)
plt.ylim(80, 100)
plt.tight_layout()
plt.savefig('model_comparison.png')
plt.show()
print("Chart saved as model_comparison.png")
print("\nDay 3 Complete!")
