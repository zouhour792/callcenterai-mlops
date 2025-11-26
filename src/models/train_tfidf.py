import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score, f1_score
import joblib
import mlflow

# === Chemins ===
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_PATH = "models/tfidf.joblib"

# === Charger les données ===
train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)

X_train, y_train = train["Document"], train["Topic_group"]
X_test, y_test = test["Document"], test["Topic_group"]

# === Démarrer l’expérience MLflow ===
mlflow.set_experiment("CallCenterAI-TFIDF")

with mlflow.start_run():
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=5000, ngram_range=(1, 2))),
            ("svm", CalibratedClassifierCV(LinearSVC(), cv=3)),
        ]
    )

    print("🚀 Entraînement du modèle TF-IDF + SVM...")
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")

    print(f"✅ Accuracy: {acc:.3f} | F1: {f1:.3f}")

    # === Logs dans MLflow ===
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_score", f1)

    # === Sauvegarde du modèle ===
    joblib.dump(pipeline, MODEL_PATH)
    mlflow.log_artifact(MODEL_PATH)
