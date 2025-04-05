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
