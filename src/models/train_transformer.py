import os
import pandas as pd
import numpy as np
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import classification_report
from huggingface_hub import login
from dotenv import load_dotenv

# ============================================================
# === 1. CHARGEMENT DES VARIABLES D’ENVIRONNEMENT ============
# ============================================================

load_dotenv()  # charge .env
HF_TOKEN = os.getenv("HF_TOKEN")  # token uniquement
if HF_TOKEN:
    print("🔑 Token Hugging Face détecté.")
    login(token=HF_TOKEN)
else:
    print("⚠️ Aucun token HF trouvé — push vers HuggingFace désactivé.")

# ============================================================
# === 2. DÉTECTION AUTOMATIQUE DU GPU ========================
# ============================================================

if torch.cuda.is_available():
    device = "cuda"
    print("🚀 GPU détecté : utilisation de CUDA")
    print("GPU utilisé :", torch.cuda.get_device_name(0))
else:
    device = "cpu"
    print("🐌 Aucun GPU détecté — utilisation du CPU (lent)")

# ============================================================
# === 3. CONFIGURATION GÉNÉRALE ==============================
# ============================================================

TRAIN_PATH = "data/processed/train.csv"
TEST_PATH = "data/processed/test.csv"
MODEL_DIR = "models/transformer"
MODEL_NAME = "distilbert-base-multilingual-cased"
HF_REPO_NAME = "callcenterai_mopls"

os.makedirs(MODEL_DIR, exist_ok=True)

# ============================================================
# === 4. CHARGEMENT ET PRÉPARATION DES DONNÉES ==============
# ============================================================

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# Rééquilibrage (oversampling)
min_samples = train_df["Topic_group"].value_counts().min()
train_df = train_df.groupby("Topic_group").apply(
    lambda x: x.sample(min_samples, replace=True, random_state=42)
).reset_index(drop=True)

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

# ============================================================
# === 5. TOKENIZATION =========================================
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    return tokenizer(examples["Document"], truncation=True, max_length=128)

train_dataset = train_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

# ============================================================
# === 6. ENCODAGE DES LABELS =================================
# ============================================================

labels = sorted(train_df["Topic_group"].unique())
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}

def map_labels(example):
    return {"labels": label2id[example["Topic_group"]]}

train_dataset = train_dataset.map(map_labels)
test_dataset = test_dataset.map(map_labels)

# ============================================================
# === 7. INITIALISATION DU MODÈLE =============================
# ============================================================

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# ============================================================
# === 8. CONFIGURATION D'ENTRAÎNEMENT ========================
# ============================================================

args = TrainingArguments(
    output_dir=MODEL_DIR,
    do_eval=True,
    per_device_train_batch_size=16 if device == "cuda" else 8,
    per_device_eval_batch_size=16 if device == "cuda" else 8,
    num_train_epochs=2,
    learning_rate=3e-5,
    weight_decay=0.01,
    logging_steps=50,
    save_safetensors=False,
)

# ============================================================
# === 9. MÉTRIQUES ===========================================
# ============================================================

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    preds = np.argmax(preds, axis=1)
    acc = (preds == labels).mean()
    return {"accuracy": acc}

# ============================================================
# === 10. ENTRAÎNEMENT =======================================
# ============================================================

print("🚀 Début de l'entraînement (GPU si disponible)...")

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

trainer.train()

# ============================================================
# === 11. ÉVALUATION =========================================
# ============================================================

print("📊 Évaluation finale sur test set...")
preds = np.argmax(trainer.predict(test_dataset).predictions, axis=1)
y_true = [label2id[x] for x in test_df["Topic_group"]]

report = classification_report(y_true, preds, target_names=labels)
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# ============================================================
# === 12. SAUVEGARDE HUGGING FACE ============================
# ============================================================

if HF_TOKEN:
    print("☁️ Upload sur Hugging Face…")
    try:
        model.push_to_hub(HF_REPO_NAME)
        tokenizer.push_to_hub(HF_REPO_NAME)
        print(f"✅ Modèle publié : https://huggingface.co/zouhour792/{HF_REPO_NAME}")
    except Exception as e:
        print("❌ Erreur push :", e)
else:
    print("ℹ️ Aucun token HF → upload ignoré.")

print("🏁 Script terminé avec succès.")
