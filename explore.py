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
