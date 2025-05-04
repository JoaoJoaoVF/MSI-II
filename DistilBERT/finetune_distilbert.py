#!/usr/bin/env python3
"""Fine‑tune DistilBERT (ou outro modelo BERT‑like) usando arquivos CSV de
tráfego de rede estruturado, convertendo cada linha em representação de texto.

O script lê dois arquivos CSV:
  * TRAIN_CSV – usado para treino (part-00000-...).
  * TEST_CSV  – usado para avaliação final (part-00135-...).

Requisitos (todos instalados no venv criado pelo setup_ml.sh):
  transformers, datasets, evaluate, scikit-learn, pandas, torch

Execute:
  python fine_tune_distilbert.py

A pasta com o checkpoint treinado será salva em OUTPUT_DIR e poderá ser
exportada/quantizada nos passos seguintes.
"""

from pathlib import Path
from typing import Dict, List
import os
import pandas as pd
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)
import evaluate

# ──────────────────────────────────────────────────────────────────────────────
# Configurações do usuário – ajuste apenas estas três variáveis se quiser mudar
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_CSV = Path("../data/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
TEST_CSV  = Path("../data/part-00135-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
OUTPUT_DIR = Path("/home/jvf/testes/meu_bert_finetune")
MODEL_NAME = "distilbert/distilbert-base-uncased"  # troque se desejar outra base
MAX_LENGTH = 128  # tokens por exemplo
EPOCHS = 2        # aumente conforme disponibilidade de GPU/CPU
BATCH_SIZE = 16

# ──────────────────────────────────────────────────────────────────────────────
# Leitura dos CSV e pré‑processamento
# ──────────────────────────────────────────────────────────────────────────────
print("Lendo CSVs…")
train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

# Identifica coluna de rótulo e colunas numéricas (tudo exceto "label")
if "label" not in train_df.columns:
    raise ValueError("A coluna 'label' não foi encontrada no CSV de treino.")
feature_cols: List[str] = [c for c in train_df.columns if c != "label"]

# Converte cada linha em string do tipo "f1:val1 f2:val2 …".
print("Convertendo linhas em texto para BERT…")
train_df["text"] = train_df[feature_cols].astype(str).agg(
    lambda row: " ".join(f"{col}:{val}" for col, val in zip(feature_cols, row)), axis=1
)

# Mapeia labels string → int
label_list: List[str] = sorted(train_df["label"].unique().tolist())
label2id: Dict[str, int] = {lbl: idx for idx, lbl in enumerate(label_list)}
print("Rótulos detectados:", label2id)

train_df["labels"] = train_df["label"].map(label2id)

# Repete transformação para o CSV de teste
missing_labels = set(test_df["label"]) - set(label2id)
if missing_labels:
    print("⚠️  Aviso: Rótulos no teste não vistos no treino serão convertidos para -1:", missing_labels)

test_df["text"] = test_df[feature_cols].astype(str).agg(
    lambda row: " ".join(f"{col}:{val}" for col, val in zip(feature_cols, row)), axis=1
)

test_df["labels"] = test_df["label"].map(label2id).fillna(-1).astype(int)

# Cria objetos Dataset
train_ds = Dataset.from_pandas(train_df[["text", "labels"]])
valid_ds = Dataset.from_pandas(test_df[["text", "labels"]])

# ──────────────────────────────────────────────────────────────────────────────
# Tokenização
# ──────────────────────────────────────────────────────────────────────────────
print("Carregando tokenizer/modelo base…")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(batch["text"], truncation=True, padding="max_length", max_length=MAX_LENGTH)

train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
valid_ds = valid_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

# ──────────────────────────────────────────────────────────────────────────────
# Modelo
# ──────────────────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label={i: l for l, i in label2id.items()},
    label2id=label2id,
)

# ──────────────────────────────────────────────────────────────────────────────
# Métricas
# ──────────────────────────────────────────────────────────────────────────────
accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"],
    }

# ──────────────────────────────────────────────────────────────────────────────
# Treinamento
# ──────────────────────────────────────────────────────────────────────────────
print("Iniciando treinamento…")

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR / "runs",
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_dir=OUTPUT_DIR / "logs",
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics,
)

trainer.train()

# ──────────────────────────────────────────────────────────────────────────────
# Salvando checkpoint e tokenizer
# ──────────────────────────────────────────────────────────────────────────────
print("Salvando modelo em", OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅  Fine‑tune concluído.")
