#!/usr/bin/env python3
"""
Fine‑tune DistilBERT (ou outro modelo BERT‑like) para classificação multiclasse de tráfego de rede,
sin precisar carregar todo o CSV em memória, usando 🤗 Datasets (memory-map).

O script lê dois arquivos CSV:
  * TRAIN_CSV – usado para treino (part-00000-...).
  * TEST_CSV  – usado para avaliação final (part-00135-...).

Requisitos (instalados no venv criado pelo setup_ml.sh):
  pip install transformers datasets evaluate scikit-learn torch

Execute:
  python fine_tune_distilbert.py

O checkpoint treinado será salvo em OUTPUT_DIR para exportação/quantização.
"""
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    TrainingArguments,
    Trainer
)
import evaluate

# ───────────────────────────────────────────────────────────────────────────
# Configurações do usuário – ajuste estas variáveis para caminhos e hiperparâmetros
# ───────────────────────────────────────────────────────────────────────────
TRAIN_CSV  = Path("../data/part-00000-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
TEST_CSV   = Path("../data/part-00135-363d1ba3-8ab5-4f96-bc25-4d5862db7cb9-c000.csv")
OUTPUT_DIR = Path("/home/jvf/testes/meu_bert_finetune")
MODEL_NAME = "distilbert/distilbert-base-uncased"  # troque se desejar outra base
MAX_LENGTH = 128  # tokens por exemplo
EPOCHS     = 2    # aumente conforme recursos
BATCH_SIZE = 16

# ───────────────────────────────────────────────────────────────────────────
# Carrega datasets sem ler tudo num pandas
# ───────────────────────────────────────────────────────────────────────────
print("Carregando datasets em formato Arrow…")
raw = load_dataset(
    "csv",
    data_files={"train": str(TRAIN_CSV), "test": str(TEST_CSV)},
    keep_in_memory=False
)
train_ds = raw["train"]
valid_ds = raw["test"]

# Verifica colunas e rótulos
if "label" not in train_ds.column_names:
    raise ValueError("Coluna 'label' não encontrada no CSV de treino.")
feature_cols = [c for c in train_ds.column_names if c != "label"]
label_list = sorted(train_ds.unique("label"))
label2id = {lbl: idx for idx, lbl in enumerate(label_list)}
print("Rótulos detectados:", label2id)

# Converte cada exemplo para um texto BERT‑friendly + mapeia label

def convert_example(ex):
    text = " ".join(f"{col}:{ex[col]}" for col in feature_cols)
    lbl  = label2id.get(ex["label"], -1)
    return {"text": text, "labels": lbl}

train_ds = train_ds.map(convert_example, batched=False)
valid_ds = valid_ds.map(convert_example, batched=False)

# ───────────────────────────────────────────────────────────────────────────
# Tokenização
# ───────────────────────────────────────────────────────────────────────────
print("Carregando tokenizer / modelo base…")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_fn(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        padding="max_length",
        max_length=MAX_LENGTH
    )

train_ds = train_ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=feature_cols + ["label", "text"]
)
valid_ds = valid_ds.map(
    tokenize_fn,
    batched=True,
    remove_columns=feature_cols + ["label", "text"]
)

# Formata para PyTorch
train_ds.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)
valid_ds.set_format(
    type="torch", columns=["input_ids", "attention_mask", "labels"]
)

# ───────────────────────────────────────────────────────────────────────────
# Modelo
# ───────────────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=len(label_list),
    id2label={i: l for l, i in label2id.items()},
    label2id=label2id
)

# ───────────────────────────────────────────────────────────────────────────
# Métricas
# ───────────────────────────────────────────────────────────────────────────
accuracy = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy.compute(predictions=preds, references=labels)["accuracy"],
        "f1": f1_metric.compute(predictions=preds, references=labels, average="weighted")["f1"]
    }

# ───────────────────────────────────────────────────────────────────────────
# Treinamento
# ───────────────────────────────────────────────────────────────────────────
print("Iniciando treinamento…")

training_args = TrainingArguments(
    output_dir=str(OUTPUT_DIR / "runs"),
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    metric_for_best_model="f1",
    load_best_model_at_end=True,
    logging_dir=str(OUTPUT_DIR / "logs"),
    learning_rate=2e-5,
    weight_decay=0.01,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=valid_ds,
    compute_metrics=compute_metrics
)

trainer.train()

# ───────────────────────────────────────────────────────────────────────────
# Salvando checkpoint e tokenizer
# ───────────────────────────────────────────────────────────────────────────
print("Salvando modelo em", OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("✅  Fine‑tune concluído.")
