import pandas as pd
import joblib
import os

# ===========================
# 1. Load Models & Encoders
# ===========================

models_dir = "../models"
model_files_with_seoul = {
    "KNN": "knn_model.pkl",
    "Decision Tree": "decision_tree_model.pkl",
    "Random Forest": "random_forest_model.pkl",
    "KNN (Reduced Cat)": "knn_model_reduced.pkl",
    "Decision Tree (Reduced Cat)": "dt_reduc_cat_model.pkl",
    "Random Forest (Reduced Cat)": "rf_reduc_cat_model.pkl"
}

model_files_no_seoul = {
    "Random Forest (No Seoul)": "random_forest_model_no_seoul.pkl",
    "Random Forest (Reduced Cat, No Seoul)": "rf_reduc_cat_no_seoul_model.pkl"
    # You don't have a dt_model_no_seoul.pkl file, so omit it
}

# Load encoders
le_genre = joblib.load(os.path.join(models_dir, "genre_encoder.pkl"))
le_nation = joblib.load(os.path.join(models_dir, "nation_encoder.pkl"))
le_rating = joblib.load(os.path.join(models_dir, "rating_encoder.pkl"))

# ===========================
# 2. User Input
# ===========================

print("🔎 Enter movie details:\n")
title = input("Movie Title: ")
release_year = int(input("Release Year (e.g., 2023): "))
nation_input = input("Nation (KR = Korean, anything else = US): ").strip().upper()
nation = "KR" if nation_input == "KR" else "US"
rating = input(f"Rating (choose from {list(le_rating.classes_)}): ").strip()
seoul_ratio = float(input("Seoul Audience Ratio (e.g., 0.24): "))

try:
    nation_enc = le_nation.transform([nation])[0]
    rating_enc = le_rating.transform([rating])[0]
except ValueError as e:
    print("\n❌ Error:", e)
    exit()

# ===========================
# 3. Input DataFrames
# ===========================

input_with_seoul = pd.DataFrame([{
    'release_year': release_year,
    'nation_enc': nation_enc,
    'rating_enc': rating_enc,
    'seoul_ratio': seoul_ratio
}])

input_no_seoul = pd.DataFrame([{
    'release_year': release_year,
    'nation_enc': nation_enc,
    'rating_enc': rating_enc
}])

# ===========================
# 4. Predict with All Models
# ===========================

print(f"\n🎬 Movie: {title} ({release_year}), {nation}, {rating}, Seoul Ratio: {seoul_ratio:.2f}")
print("\n📊 Genre Predictions:")

# Models WITH seoul_ratio
for label, file in model_files_with_seoul.items():
    try:
        model = joblib.load(os.path.join(models_dir, file))
        pred = model.predict(input_with_seoul)[0]
        genre = le_genre.inverse_transform([pred])[0]
        print(f"- {label}: {genre}")
    except Exception as e:
        print(f"- {label}: ❌ Failed ({e})")

# Models WITHOUT seoul_ratio
for label, file in model_files_no_seoul.items():
    try:
        model = joblib.load(os.path.join(models_dir, file))
        pred = model.predict(input_no_seoul)[0]
        genre = le_genre.inverse_transform([pred])[0]
        print(f"- {label}: {genre}")
    except Exception as e:
        print(f"- {label}: ❌ Failed ({e})")
