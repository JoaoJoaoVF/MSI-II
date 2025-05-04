#!/usr/bin/env python3
"""
Fine‑tune DistilBERT (ou outro modelo BERT‑like) para classificação multiclasse de tráfego de rede,
sem carregar todo o CSV em memória, usando 🤗 Datasets (memory-mapped).

Requisitos:
  pip install transformers datasets evaluate scikit-learn torch

Como usar:
  Ajuste os caminhos TRAIN_CSV e TEST_CSV para seus arquivos.
  Execute: python finetune_distilbert.py
"""
import os
from datasets import load_dataset, ClassLabel
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import evaluate

# ----- Configurações -----
TRAIN_CSV  = "part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
TEST_CSV   = "part-00135-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv"
MODEL_NAME = "distilbert/distilbert-base-uncased"
OUTPUT_DIR = "/home/jvf/testes/meu_bert_finetune"
BATCH_SIZE = 8
EPOCHS     = 3

# ----- Carregar datasets sem pandas -----
raw_datasets = load_dataset(
    "csv",
    data_files={"train": TRAIN_CSV, "test": TEST_CSV},
    cache_dir=os.path.expanduser("~/.cache/huggingface/datasets"),
    keep_in_memory=False
)

# Detectar e mapear rótulos
labels = sorted(raw_datasets["train"].unique("label"))
label2id = {label: idx for idx, label in enumerate(labels)}
id2label = {idx: label for label, idx in label2id.items()}
print("Rótulos detectados:", label2id)

# Converter coluna `label` para ClassLabel
raw_datasets = raw_datasets.cast_column(
    "label", ClassLabel(names=labels)
)

# Carregar tokenizer e modelo
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id
)

# ----- Pré-processamento -----
# Concatena todas as colunas (exceto `label`) em um único texto
def concat_fields(example):
    text = " ".join(f"{k}:{v}" for k, v in example.items() if k != "label")
    return {"text": text}

# Aplica concatenação e tokenização em lotes
def preprocess(examples):
    texts = [" ".join(f"{k}:{v}" for k, v in zip(examples.keys(), cols) if k != "label")
             for cols in zip(*[examples[c] for c in examples.keys()])]
    tokenized = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=128
    )
    tokenized["labels"] = [label2id[l] for l in examples["label"]]
    return tokenized

processed = raw_datasets.map(
    preprocess,
    batched=True,
    batch_size=BATCH_SIZE,
    remove_columns=raw_datasets["train"].column_names
)

# Formatar para PyTorch
processed.set_format(type="torch",
                     columns=["input_ids", "attention_mask", "labels"])

train_dataset = processed["train"]
eval_dataset  = processed["test"]

# ----- Métricas -----
metric_acc = evaluate.load("accuracy")
metric_f1  = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": metric_acc.compute(predictions=preds, references=labels)["accuracy"],
        "f1": metric_f1.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# ----- Treino -----
args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=EPOCHS,
    logging_steps=50,
    load_best_model_at_end=True,
    metric_for_best_model="f1"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
print(f"Modelo salvo em {OUTPUT_DIR}")
