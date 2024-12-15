from transformers import T5ForConditionalGeneration, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from transformers import AutoTokenizer

#Train Model Code
def train_model(tokenized_dataset, save_path="./new_model"):
    model = T5ForConditionalGeneration.from_pretrained("t5-small")
    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=8,
        num_train_epochs=3,
        weight_decay=0.01,
        save_total_limit=2,
        fp16=True,
        logging_dir="./logs",
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"].select(range(1000)),
        eval_dataset=tokenized_dataset["validation"].select(range(1000)),
        data_collator=data_collator,
        tokenizer=tokenizer,
    )
    trainer.train()
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

