# rf_reduc_cat_no_seoul.py

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

# =========================
# 2. Reduce Genre Categories
# =========================

genre_map = {
    'Action': 'Action',
    'Thriller': 'Action',
    'Adult film': 'Adult',
    'Romance': 'Romance',
    'Drama': 'Romance',
    'Comedy': 'Comedy',
    'Animation': 'Animation',
    'Family': 'Animation',
    'Adventure': 'Other',
    'Fantasy': 'Other',
    'Historical': 'Other',
    'Musical': 'Other',
    'Mystery': 'Other',
    'Other': 'Other',
    'Unknown': 'Other',
    'War': 'Other',
    'Western': 'Other',
    'Sci-Fi': 'Other',
    'Performance': 'Other',
    'Documentary': 'Other',
    'Crime': 'Other',
    'Horror': 'Other'
}

df['genre'] = df['genre'].map(genre_map)
df = df.dropna(subset=['genre'])

# =========================
# 3. Encode Categorical Vars
# =========================

le_nation = LabelEncoder()
le_rating = LabelEncoder()
le_genre = LabelEncoder()

df['nation_enc'] = le_nation.fit_transform(df['nation'])
df['rating_enc'] = le_rating.fit_transform(df['rating'])
df['genre_enc'] = le_genre.fit_transform(df['genre'])

# =========================
# 4. Features & Labels (⚠️ no seoul_ratio)
# =========================

X = df[['release_year', 'nation_enc', 'rating_enc']].dropna()
y = df.loc[X.index, 'genre_enc']

# =========================
# 5. Train/Test Split & RF
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_leaf=3,
    class_weight='balanced',
    random_state=42
)
clf.fit(X_train, y_train)

# =========================
# 6. Save Model & Encoders
# =========================

models_dir = "../models"
data_dir = "../data"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

joblib.dump(clf, os.path.join(models_dir, "rf_reduc_cat_no_seoul_model.pkl"))
joblib.dump(le_genre, os.path.join(models_dir, "genre_encoder_reduced.pkl"))
joblib.dump(le_nation, os.path.join(models_dir, "nation_encoder.pkl"))
joblib.dump(le_rating, os.path.join(models_dir, "rating_encoder.pkl"))

df.to_csv(os.path.join(data_dir, "cleaned_movie_data_reduced_no_seoul.csv"), index=False)

# =========================
# 7. Evaluation
# =========================

y_pred = clf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le_genre.classes_)

results_dir = "../results"
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, "rf_reduc_cat_no_seoul.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("📄 rf_reduc_cat_no_seoul Results\n")
    f.write("=" * 40 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print(f"\n✅ Report saved to: {results_path}")
# =========================
# 7.1 Feature Importance
# =========================

feature_names = ['release_year', 'nation_enc', 'rating_enc']
importances = clf.feature_importances_

# Save importance values to .txt file
importance_txt_path = os.path.join(results_dir, "feature_importance", "rf_reduc_cat_no_seoul_feature_importance.txt")
with open(importance_txt_path, "w", encoding="utf-8") as f:
    f.write("📊 Feature Importance (rf_reduc_cat_no_seoul)\n")
    f.write("=" * 40 + "\n")
    for name, score in zip(feature_names, importances):
        f.write(f"{name}: {score:.4f}\n")

print(f"\n📁 Feature importance saved to: {importance_txt_path}")

# Save plot to .png file
plot_path = os.path.join(results_dir, "feature_importance", "rf_reduc_cat_no_seoul_feature_importance.png")
plt.figure(figsize=(6, 4))
plt.barh(feature_names, importances)
plt.xlabel("Importance")
plt.title("Feature Importance\n(RF - Reduced Genre, No Seoul Ratio)")
plt.tight_layout()
plt.savefig(plot_path, dpi=300)
plt.close()

print(f"📊 Plot saved to: {plot_path}")

# =========================
# 8. Example Prediction
# =========================

sample_idx = X_test.index[2]
original_movie = df.loc[sample_idx]

print("\n🎬 Sample Movie Info:")
print(f"Title: {original_movie.get('title', 'Unknown')}")
print(f"Release Year: {original_movie['release_year']}")
print(f"Nation: {le_nation.inverse_transform([original_movie['nation_enc']])[0]}")
print(f"Rating: {le_rating.inverse_transform([original_movie['rating_enc']])[0]}")
print(f"\n📌 Predicted Genre: {le_genre.inverse_transform([y_pred[2]])[0]}")
print(f"✅ Actual Genre: {le_genre.inverse_transform([y_test.iloc[2]])[0]}")


# Filter to only used rows in this model (no seoul_ratio, but audience_total still valid)
used_rows = df.loc[X.index]

# Group and calculate average audience_total
nation_pref = used_rows.groupby('genre')['audience_total'].mean().sort_values(ascending=False)

# Save to TXT
nation_pref_path = os.path.join(results_dir, "nationwide_preference_by_genre.txt")
nation_pref.to_csv(nation_pref_path, sep='\t', float_format="%.2f")
print(f"\n📁 Nationwide preference saved to: {nation_pref_path}")

# Plot
plt.figure(figsize=(8, 5))
nation_pref.plot(kind='barh', color='lightcoral')
plt.xlabel("Average Nationwide Audience")
plt.title("Nationwide Audience Preference by Genre")
plt.gca().invert_yaxis()
plt.tight_layout()

# Save Plot
nation_plot_path = os.path.join(results_dir, "nationwide_preference_by_genre.png")
plt.savefig(nation_plot_path, dpi=300)
plt.close()

print(f"📊 Nationwide preference plot saved to: {nation_plot_path}")
