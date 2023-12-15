from transformers import T5ForConditionalGeneration, AutoTokenizer
from datasets import load_metric

def evaluate_model(dataset, model_path):
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    rouge = load_metric("rouge")

    all_predictions, all_references = [], []
    for example in dataset:
        input_ids = tokenizer("summarize: " + example["article"], return_tensors="pt", max_length=512, truncation=True).input_ids
        outputs = model.generate(input_ids, max_length=150, min_length=40, num_beams=4)
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        all_predictions.append(prediction)
        all_references.append(example["highlights"])
    results = rouge.compute(predictions=all_predictions, references=all_references, use_stemmer=True)
    return results
