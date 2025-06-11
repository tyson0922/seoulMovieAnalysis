# genre_model_knn.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# =========================
# 1. Load & Clean the Data
# =========================

# Load the CSV file (make sure the path is correct)
df = pd.read_csv("../data/movie_box_office_kr_until_2025_0430.csv")

# Drop rows with missing key values
df = df.dropna(subset=['audience_total', 'audience_seoul', 'genre'])

# Create a new column: seoul_ratio = 서울 관객 수 / 전체 관객 수
df['seoul_ratio'] = df['audience_seoul'] / df['audience_total']

# Extract release year from the date column
df['release_year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year

# Filter for recent years only (e.g., from 2010 onward)
df = df[df['release_year'] >= 2010]

# Remove rows with missing values in other key features
df = df.dropna(subset=['nation', 'rating', 'release_year'])

# ================================
# 2. Encode Categorical Variables
# ================================

# Encode 'nation', 'rating', and 'genre' as integers
le_nation = LabelEncoder()
le_rating = LabelEncoder()
le_genre = LabelEncoder()

df['nation_enc'] = le_nation.fit_transform(df['nation'])
df['rating_enc'] = le_rating.fit_transform(df['rating'])
df['genre_enc'] = le_genre.fit_transform(df['genre'])

# =========================
# 3. Prepare Feature & Label
# =========================

# Feature columns: release_year, nation, rating, seoul_ratio
X = df[['release_year', 'nation_enc', 'rating_enc', 'seoul_ratio']].dropna()

# Target column: genre (encoded)
y = df.loc[X.index, 'genre_enc']

# =====================================
# 4. Train/Test Split & Model Training
# =====================================

# Split the data into training and testing sets (80% train / 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Create the KNN classifier with k=5
knn = KNeighborsClassifier(n_neighbors=5)

# Train the model
knn.fit(X_train, y_train)


# === Directory setup ===
models_dir = "../models"
data_dir = "../data"

# Make sure directories exist
os.makedirs(models_dir, exist_ok=True)
os.makedirs(data_dir, exist_ok=True)

# Save model and encoders
joblib.dump(knn, os.path.join(models_dir, "knn_model.pkl"))
joblib.dump(le_genre, os.path.join(models_dir, "genre_encoder.pkl"))
joblib.dump(le_nation, os.path.join(models_dir, "nation_encoder.pkl"))
joblib.dump(le_rating, os.path.join(models_dir, "rating_encoder.pkl"))

# Save cleaned and encoded DataFrame for reuse
df.to_csv(os.path.join(data_dir, "cleaned_movie_data.csv"), index=False)

# ========================
# 5. Predict & Evaluate
# ========================

# Predict the genre for test data
y_pred = knn.predict(X_test)

# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le_genre.classes_)

results_dir = "../results"
os.makedirs(results_dir, exist_ok=True)

results_path = os.path.join(results_dir, "genre_model_knn.txt")
with open(results_path, "w", encoding="utf-8") as f:
    f.write("📄 genre_model_knn Results\n")
    f.write("=" * 30 + "\n")
    f.write(f"Accuracy: {accuracy:.4f}\n\n")
    f.write(report)

print("f\n Repoorts saved to: {results_path}")



# ========================================
# 6. Example Prediction with Movie Name
# ========================================

# Get the index of the first test sample
sample_idx = X_test.index[2]

# Get original movie info
original_movie = df.loc[sample_idx]

# Show the input features
print("\n🎬 Sample Movie Info:")
print(f"Movie Title: {original_movie.get('title', 'Unknown')}")
print(f"Release Year: {original_movie['release_year']}")
print(f"Nation: {le_nation.inverse_transform([original_movie['nation_enc']])[0]}")
print(f"Rating: {le_rating.inverse_transform([original_movie['rating_enc']])[0]}")
print(f"Seoul Ratio: {original_movie['seoul_ratio']:.2f}")

# Show prediction result
print(f"\n📌 Predicted Genre: {le_genre.inverse_transform([y_pred[2]])[0]}")
print(f"✅ Actual Genre: {le_genre.inverse_transform([y_test.iloc[2]])[0]}")