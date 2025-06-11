# genre_model_rf.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import matplotlib.pyplot as plt


# =========================
# 1. Load & Clean the Data
# =========================

df = pd.read_csv("../data/movie_box_office_kr_until_2025_0430.csv")
df = df.dropna(subset=['audience_total', 'audience_seoul', 'genre'])

df['seoul_ratio'] = df['audience_seoul'] / df['audience_total']
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year
df = df[df['release_year'] >= 2010]
df = df.dropna(subset=['nation', 'rating', 'release_year'])

# ================================
# 2. Encode Categorical Variables
# ================================

le_nation = LabelEncoder()
le_rating = LabelEncoder()
le_genre = LabelEncoder()

df['nation_enc'] = le_nation.fit_transform(df['nation'])
df['rating_enc'] = le_rating.fit_transform(df['rating'])
df['genre_enc'] = le_genre.fit_transform(df['genre'])

# =========================
# 3. Prepare Feature & Label
# =========================

X = df[['release_year', 'nation_enc', 'rating_enc', 'seoul_ratio']].dropna()
y = df.loc[X.index, 'genre_enc']

# =====================================
# 4. Train/Test Split & Model Training
# =====================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Random Forest Classifier
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# === Directory setup ===
models_dir = "../models"
data_dir = "../data"
results_dir = "../results"

os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Save model and encoders
joblib.dump(clf, os.path.join(models_dir, "random_forest_model.pkl"))
joblib.dump(le_genre, os.path.join(models_dir, "genre_encoder.pkl"))
joblib.dump(le_nation, os.path.join(models_dir, "nation_encoder.pkl"))
joblib.dump(le_rating, os.path.join(models_dir, "rating_encoder.pkl"))
df.to_csv(os.path.join(data_dir, "cleaned_movie_data.csv"), index=False)

# ========================
# 5. Predict & Evaluate
# ========================

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le_genre.classes_)

results_path = os.path.join(results_dir, "genre_model_rf.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("📄 genre_model_rf Results\n")
    f.write("=" * 30 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print(f"\n✅ Report saved to: {results_path}")

# =========================
# 6. Feature Importance
# =========================

feature_names = ['release_year', 'nation_enc', 'rating_enc', 'seoul_ratio']
importances = clf.feature_importances_

# Save importance values to text
importance_txt_path = os.path.join(results_dir, "feature_importance", "genre_model_rf_feature_importance.txt")
with open(importance_txt_path, "w", encoding="utf-8") as f:
    f.write("📊 Feature Importance (genre_model_rf)\n")
    f.write("=" * 40 + "\n")
    for name, score in zip(feature_names, importances):
        f.write(f"{name}: {score:.4f}\n")

print(f"📁 Feature importance saved to: {importance_txt_path}")

# Save plot to file
plot_path = os.path.join(results_dir, "feature_importance", "genre_model_rf_feature_importance.png")
plt.figure(figsize=(6, 4))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance\n(RF With All Genre)")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"📊 Plot saved to: {plot_path}")

# ========================================
# 7. Example Prediction with Movie Name
# ========================================

sample_idx = X_test.index[2]
original_movie = df.loc[sample_idx]

print("\n🎬 Sample Movie Info:")
print(f"Movie Title: {original_movie.get('title', 'Unknown')}")
print(f"Release Year: {original_movie['release_year']}")
print(f"Nation: {le_nation.inverse_transform([original_movie['nation_enc']])[0]}")
print(f"Rating: {le_rating.inverse_transform([original_movie['rating_enc']])[0]}")
print(f"Seoul Ratio: {original_movie['seoul_ratio']:.2f}")

print(f"\n📌 Predicted Genre: {le_genre.inverse_transform([y_pred[2]])[0]}")
print(f"✅ Actual Genre: {le_genre.inverse_transform([y_test.iloc[2]])[0]}")
