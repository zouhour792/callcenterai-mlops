import os
import pandas as pd
import numpy as np
import evaluate
import mlflow
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report
import torch

# === Chemins ===
TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models/transformer"
os.makedirs(MODEL_DIR, exist_ok=True)

# === Charger les données ===
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Rééquilibrage simple si certaines classes sont rares
min_samples = train_df["Topic_group"].value_counts().min()
train_df = train_df.groupby("Topic_group").apply(
    lambda x: x.sample(min_samples, replace=True)
).reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# === Tokenizer francophone ===
model_name = "distilbert-base-multilingual-cased"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def preprocess_function(examples):
    return tokenizer(examples["Document"], truncation=True, max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# === Encodage des labels ===
labels = sorted(train_df["Topic_group"].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

def map_labels(example):
    return {"labels": label2id[example["Topic_group"]]}

train_dataset = train_dataset.map(map_labels)
test_dataset = test_dataset.map(map_labels)

# === Modèle ===
model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=len(labels), id2label=id2label, label2id=label2id
)

# === Entraînement ===
args = TrainingArguments(
    output_dir=MODEL_DIR,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_dir="./logs",
    save_strategy="epoch",
    evaluation_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    greater_is_better=True,
)

metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    probs = torch.softmax(torch.tensor(preds), dim=1)
    preds = np.argmax(preds, axis=1)
    f1 = metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    acc = np.mean(preds == labels)
    return {"accuracy": acc, "f1": f1}

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

mlflow.set_experiment("CallCenterAI-Transformer")

with mlflow.start_run():
    print("🚀 Entraînement CamemBERT...")
    trainer.train()
    print("💾 Sauvegarde du modèle...")
    trainer.save_model(MODEL_DIR)
    tokenizer.save_pretrained(MODEL_DIR)
    mlflow.log_artifact(MODEL_DIR)

    print("📊 Évaluation finale...")
    y_true = [label2id[l] for l in test_df["Topic_group"]]
    preds = np.argmax(trainer.predict(test_dataset).predictions, axis=1)
    report = classification_report(y_true, preds, target_names=labels, digits=3)
    print(report)

    with open("classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    mlflow.log_metrics(trainer.evaluate())
    print("✅ Modèle sauvegardé !")
