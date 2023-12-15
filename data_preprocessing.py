import os
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("t5-small")

def preprocess_data(examples):
    inputs = ["summarize: " + doc for doc in examples["article"]]
    model_inputs = tokenizer(inputs, max_length=512, padding="max_length", truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(examples["highlights"], max_length=150, padding="max_length", truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def load_and_preprocess_data():
    dataset = load_dataset("cnn_dailymail", "3.0.0", cache_dir="./datasets/cnn_dailymail_clean")
    tokenized_dataset = dataset.map(preprocess_data, batched=True)
    tokenized_dataset.save_to_disk("./tokenized_cnn_dailymail")
    return tokenized_dataset


