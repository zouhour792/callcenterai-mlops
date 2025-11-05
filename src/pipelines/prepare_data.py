import pandas as pd
import re
import os
from sklearn.model_selection import train_test_split

RAW_PATH = "data/raw/all_tickets_processed_improved_v3.csv"
PROCESSED_DIR = "data/processed"

def clean_text(text: str) -> str:
    """Nettoyage basique du texte"""
    text = str(text).lower()
    text = re.sub(r"http\S+", "", text)
    text = re.sub(r"[^a-zA-Z0-9À-ž\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("📥 Lecture du dataset brut...")
    df = pd.read_csv(RAW_PATH)

    if not {"Document", "Topic_group"}.issubset(df.columns):
        raise ValueError(f"Colonnes manquantes dans {RAW_PATH} : {df.columns}")

    print("🧹 Nettoyage du texte...")
    df["Document"] = df["Document"].apply(clean_text)
    df = df.dropna(subset=["Document", "Topic_group"]).reset_index(drop=True)

    print("✂️ Séparation train / test...")
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["Topic_group"]
    )

    print("💾 Sauvegarde des fichiers...")
    train_df.to_csv(os.path.join(PROCESSED_DIR, "train.csv"), index=False)
    test_df.to_csv(os.path.join(PROCESSED_DIR, "test.csv"), index=False)

    print("✅ Données prêtes dans data/processed/")

if __name__ == "__main__":
    main()
