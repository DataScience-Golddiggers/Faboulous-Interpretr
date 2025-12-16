import argparse
import pandas as pd
import torch
import os
import logging
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    TrainingArguments, 
    Trainer, 
    DataCollatorWithPadding
)
from peft import (
    get_peft_model, 
    LoraConfig, 
    TaskType, 
    PeftModel, 
    PeftConfig
)

# Configurazione Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Mappatura label di default (può essere sovrascritta o adattata)
DEFAULT_LABEL_MAP = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
    # Aggiungi varianti comuni per robustezza
    "neg": 0, "neu": 1, "pos": 2,
    "negativo": 0, "neutro": 1, "positivo": 2
}

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = predictions.argmax(axis=1)
    
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    
    return {"accuracy": acc, "f1": f1}

def load_and_clean_data(file_path, text_col, label_col, test_size=0.2):
    """
    Carica il CSV, rinomina le colonne, mappa le label e divide in train/val.
    """
    logger.info(f"Caricamento dati da {file_path}...")
    df = pd.read_csv(file_path)
    
    # Verifica colonne
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(f"Colonne '{text_col}' o '{label_col}' non trovate nel CSV. Colonne disponibili: {df.columns.tolist()}")
    
    # Rinomina per standardizzazione
    df = df.rename(columns={text_col: "text", label_col: "label"})
    
    # Rimuovi righe con valori nulli
    initial_len = len(df)
    df = df.dropna(subset=["text", "label"])
    if len(df) < initial_len:
        logger.warning(f"Rimosse {initial_len - len(df)} righe con valori mancanti.")
    
    # Pulizia base testo (opzionale, si affida al tokenizer per il grosso)
    df["text"] = df["text"].astype(str).str.strip()
    
    # Mappatura label
    # Se le label sono già interi, verifica che siano 0, 1, 2
    if pd.api.types.is_integer_dtype(df["label"]):
        unique_labels = sorted(df["label"].unique())
        logger.info(f"Label numeriche rilevate: {unique_labels}")
        if not set(unique_labels).issubset({0, 1, 2}):
             raise ValueError(f"Label numeriche fuori range (atteso 0,1,2), trovato: {unique_labels}")
    else:
        # Tenta mappatura stringa -> int
        df["label"] = df["label"].astype(str).str.lower().map(DEFAULT_LABEL_MAP)
        if df["label"].isnull().any():
            unmapped = df[df["label"].isnull()][label_col].unique()
            raise ValueError(f"Impossibile mappare alcune label: {unmapped}. Verifica DEFAULT_LABEL_MAP o i dati.")
        df["label"] = df["label"].astype(int)
        
    logger.info(f"Dataset pronto. {len(df)} righe totali.")
    
    # Split
    train_df, val_df = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=42)
    
    return DatasetDict({
        "train": Dataset.from_pandas(train_df, preserve_index=False),
        "validation": Dataset.from_pandas(val_df, preserve_index=False)
    })

def main():
    parser = argparse.ArgumentParser(description="Fine-tuning LoRA per Sentiment Analysis con XLM-RoBERTa")
    parser.add_argument("--data_path", type=str, required=True, help="Percorso al file CSV dei dati")
    parser.add_argument("--output_dir", type=str, default="models/sentiment_lora", help="Directory output modello")
    parser.add_argument("--model_name", type=str, default="xlm-roberta-base", help="Modello base Hugging Face")
    parser.add_argument("--text_col", type=str, default="text", help="Nome colonna testo nel CSV")
    parser.add_argument("--label_col", type=str, default="label", help="Nome colonna label nel CSV")
    parser.add_argument("--epochs", type=int, default=3, help="Numero di epoche")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    
    args = parser.parse_args()
    
    # 1. Preparazione Dati
    dataset = load_and_clean_data(args.data_path, args.text_col, args.label_col)
    
    # 2. Tokenizer
    logger.info(f"Caricamento tokenizer {args.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)
    
    tokenized_datasets = dataset.map(tokenize_function, batched=True)
    
    # 3. Modello Base
    logger.info("Caricamento modello base...")
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=3,
        id2label={0: "negative", 1: "neutral", 2: "positive"},
        label2id={"negative": 0, "neutral": 1, "positive": 2}
    )
    
    # 4. Configurazione LoRA
    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS, 
        inference_mode=False, 
        r=16,            # Rango intrinseco (più alto = più parametri)
        lora_alpha=32,   # Scaling factor
        lora_dropout=0.1,
        target_modules=["query", "value"] # Target tipici per modelli BERT-like
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # 5. Training
    training_args = TrainingArguments(
        output_dir=f"{args.output_dir}_checkpoints",
        learning_rate=args.lr,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_dir=f"{args.output_dir}_logs",
        logging_steps=10,
        use_cpu=not torch.cuda.is_available()
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    
    logger.info("Avvio training...")
    trainer.train()
    
    # 6. Salvataggio
    logger.info(f"Salvataggio modello LoRA in {args.output_dir}...")
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    logger.info("Fatto.")

if __name__ == "__main__":
    main()
