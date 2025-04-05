# Fine-Tuning BERT for Amazon Review Classification

This project fine-tunes a pretrained BERT model (`bert-base-uncased`) to classify Amazon product reviews into multiple categories using a dataset from Kaggle.

## 📌 Goal

The goal is to train a robust text classification model using fine-tuning techniques with Hugging Face Transformers.

## 📊 Dataset

Source: [Kaggle - Consumer Reviews of Amazon Products](https://www.kaggle.com/datasets/datafiniti/consumer-reviews-of-amazon-products)

## 🛠️ Tools & Libraries

- BERT (`bert-base-uncased`)
- Hugging Face Transformers
- PyTorch / Trainer API
- scikit-learn
- Pandas

---
## 👩‍💻 Author

**Sara Moshtaghi** — NLP Researcher & Machine Learning Engineer  
[Hugging Face](https://huggingface.co/saramoshtaghi) | [LinkedIn](https://linkedin.com/in/saramoshtaghi) | [GitHub](https://github.com/saramoshtaghi)
---

## 🚀 Inference Example

You can use your fine-tuned model for real-time predictions using Hugging Face `pipeline`.

```python
from transformers import pipeline, BertForSequenceClassification, AutoTokenizer

# Load your fine-tuned model and tokenizer
model = BertForSequenceClassification.from_pretrained("./models/bert-amazon-finetuned")
tokenizer = AutoTokenizer.from_pretrained("./models/bert-amazon-finetuned")

# Create sentiment classifier
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

# Predict
text = "The product is amazing, totally worth it!"
result = classifier(text)

print(result)

---

## 🔧 Project Structure

```text
bert-amazon-review-classification/
├── data/                        # Contains CSV dataset
├── models/                      # Saved fine-tuned model
├── notebook/                   # Jupyter Notebooks
├── utils/                       # Preprocessing scripts
├── requirements.txt
├── README.md
├── .gitignore
