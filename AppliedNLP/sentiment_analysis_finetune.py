import os
import numpy as np
import random
import torch
import evaluate
from dataclasses import dataclass
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
# --- 1. CONFIGURATION (The Control Center) ---
# Professional engineers keep all "knobs" in one place using dataclasses.
@dataclass
class ModelConfig:
    model_name: str = "distilbert-base-uncased"
    dataset_name: str = "imdb"
    num_labels: int = 2
    output_dir: str = "./results_level2"
    seed: int = 42

@dataclass
class TrainConfig:
    learning_rate: float = 2e-5
    batch_size: int = 16
    epochs: int = 2
    weight_decay: float = 0.01

# Initialize our configs
model_cfg = ModelConfig()
train_cfg = TrainConfig()
# --- 2. UTILITIES (Helper Functions) ---
def set_seed(seed_value):
    """Ensures the run is reproducible."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

def compute_metrics(eval_pred):
    """Calculates Accuracy and F1 score."""
    accuracy_metric = evaluate.load("accuracy")
    f1_metric = evaluate.load("f1")

    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    acc = accuracy_metric.compute(predictions=predictions, references=labels)
    f1 = f1_metric.compute(predictions=predictions, references=labels)
    return {**acc, **f1}

def get_tokenized_data(config: ModelConfig):
    """Loads and tokenizes the dataset."""
    print(f"Loading {config.dataset_name}...")
    dataset = load_dataset(config.dataset_name)
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)

    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)

    print("Tokenizing dataset...")
    tokenized = dataset.map(tokenize_function, batched=True)

    # We return the tokenizer too, because we need it for inference later
    return tokenized, tokenizer
    # --- 3. MAIN EXECUTION ---
def run_training():
    # A. Setup
    set_seed(model_cfg.seed)

    # B. Data
    tokenized_datasets, tokenizer = get_tokenized_data(model_cfg)

    # Use smaller subsets for demonstration (Remove these lines for full training)
    train_dataset = tokenized_datasets["train"].shuffle(seed=model_cfg.seed).select(range(1000))
    eval_dataset = tokenized_datasets["test"].shuffle(seed=model_cfg.seed).select(range(500))

    # C. Model
    print(f"Initializing {model_cfg.model_name}...")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg.model_name,
        num_labels=model_cfg.num_labels
    )

    # D. Trainer Setup
    training_args = TrainingArguments(
        output_dir=model_cfg.output_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=train_cfg.learning_rate,
        per_device_train_batch_size=train_cfg.batch_size,
        per_device_eval_batch_size=train_cfg.batch_size,
        num_train_epochs=train_cfg.epochs,
        weight_decay=train_cfg.weight_decay,
        load_best_model_at_end=True,
        report_to="none", # Keeping WandB off for now
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
    )

    # E. Train & Evaluate
    print("Starting training...")
    trainer.train()

    print("Evaluating...")
    results = trainer.evaluate()
    print(f"Final Metrics: {results}")

    # F. Save the final artifact (Model + Tokenizer)
    # This is critical for production: always save the tokenizer with the model!
    save_path = f"{model_cfg.output_dir}/final_model"
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Model saved to {save_path}")
    # Run the script
if __name__ == "__main__":
    run_training()