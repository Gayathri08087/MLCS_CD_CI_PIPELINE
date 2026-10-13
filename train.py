
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from skl2onnx import to_onnx
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# Project paths
DATA = Path("data/drug200.csv")
MODEL_DIR = Path("model")
RESULTS_DIR = Path("results")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# 1. Load the dataset
if not DATA.exists():
    raise FileNotFoundError(
        f"Dataset not found: {DATA}. "
        "Place drug200.csv inside the data folder."
    )

df = pd.read_csv(DATA)

required_columns = [
    "Age",
    "Sex",
    "BP",
    "Cholesterol",
    "Na_to_K",
    "Drug",
]

missing = set(required_columns) - set(df.columns)
if missing:
    raise ValueError(f"Missing dataset columns: {sorted(missing)}")

df = df[required_columns].dropna()

X = df.drop(columns=["Drug"])
y = df["Drug"]

# 2. Identify categorical and numerical features
categorical = ["Sex", "BP", "Cholesterol"]
numerical = ["Age", "Na_to_K"]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical,
        ),
        ("numerical", "passthrough", numerical),
    ]
)

# 3. Create the Random Forest pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=100,
                random_state=42,
            ),
        ),
    ]
)

# 4. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

# 5. Train and evaluate
pipeline.fit(X_train, y_train)
predictions = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
f1 = f1_score(
    y_test,
    predictions,
    average="weighted",
)

report = classification_report(
    y_test,
    predictions,
    zero_division=0,
)

# 6. Save the trained Python model
joblib.dump(
    pipeline,
    MODEL_DIR / "drug_model.joblib",
)

# 7. Save evaluation metrics
with open(RESULTS_DIR / "metrics.txt", "w", encoding="utf-8") as file:
    file.write(f"Accuracy: {accuracy:.4f}\n")
    file.write(f"Weighted F1-score: {f1:.4f}\n\n")
    file.write(report)

# 8. Generate the confusion matrix
ConfusionMatrixDisplay.from_predictions(
    y_test,
    predictions,
    xticks_rotation="vertical",
)

plt.tight_layout()
plt.savefig(RESULTS_DIR / "confusion_matrix.png")
plt.close()

# 9. Export the fitted pipeline to ONNX
sample = X_train.iloc[:1]

onnx_model = to_onnx(
    pipeline,
    sample,
    target_opset=17,
    options={id(pipeline): {"zipmap": False}},
)

with open(MODEL_DIR / "drug_model.onnx", "wb") as file:
    file.write(onnx_model.SerializeToString())

# 10. Display results
print(f"Dataset rows: {len(df)}")
print(f"Training rows: {len(X_train)}")
print(f"Testing rows: {len(X_test)}")
print(f"Accuracy: {accuracy:.4f}")
print(f"Weighted F1-score: {f1:.4f}")
print("Saved model: model/drug_model.joblib")
print("Saved ONNX model: model/drug_model.onnx")
print("Saved metrics: results/metrics.txt")
print("Saved confusion matrix: results/confusion_matrix.png")
print("Training and evaluation completed.")
