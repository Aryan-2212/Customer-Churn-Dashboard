import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

# -----------------------------
# Load Dataset
# -----------------------------
df = pd.read_csv("BankChurners_Cleaned.csv")

# -----------------------------
# Define Features and Target
# -----------------------------
X = df.drop("Attrition_Flag", axis=1)
y = df["Attrition_Flag"]

# -----------------------------
# Identify Column Types
# -----------------------------
categorical_cols = X.select_dtypes(include=["object"]).columns
numerical_cols = X.select_dtypes(exclude=["object"]).columns

# -----------------------------
# Preprocessing Pipeline
# -----------------------------
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# -----------------------------
# Random Forest Model
# -----------------------------
rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            random_state=42
        ))
    ]
)

# Train Model
rf_model.fit(X_train, y_train)

# -----------------------------
# Extract Feature Importance
# -----------------------------
feature_names = rf_model.named_steps["preprocessor"].get_feature_names_out()

importances = rf_model.named_steps["classifier"].feature_importances_

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
})

# Clean feature names
importance_df["Feature"] = importance_df["Feature"].str.replace("num__", "")
importance_df["Feature"] = importance_df["Feature"].str.replace("cat__", "")

importance_df = importance_df.sort_values(by="Importance", ascending=False)

print("\nTop Churn Drivers:")
print(importance_df.head(10))

# -----------------------------
# Calculate Churn Rate
# -----------------------------
churn_rate = (df["Attrition_Flag"] == "Attrited Customer").mean()

# -----------------------------
# Generate Insights JSON
# -----------------------------
insights = {
    "dataset_size": int(len(df)),
    "churn_rate": float(round(churn_rate, 4)),
    "best_model": "Random Forest",
    "top_churn_drivers": importance_df.head(5)["Feature"].tolist()
}

# Save JSON
with open("churn_insights.json", "w") as f:
    json.dump(insights, f, indent=4)

print("\nInsights JSON generated successfully.")
print(insights)

# -----------------------------
# Calculate Churn Rate
# -----------------------------

churn_rate = df["Attrition_Flag"].mean()

print(f"\nOverall Churn Rate: {churn_rate:.2%}")