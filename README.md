# Brazilian E-commerce Sentiment Analysis

Natural Language Processing (NLP) project focused on classifying Brazilian e-commerce customer reviews using traditional Machine Learning techniques and Transformer-based models.

## Project Overview

This project explores sentiment classification in Portuguese customer reviews, comparing traditional NLP approaches with a fine-tuned BERTimbau model.

The objective is to classify customer feedback into three sentiment categories:

- **Negative**
- **Neutral**
- **Positive**

The project covers data preprocessing, exploratory analysis, feature extraction, model training, class imbalance handling, evaluation, error analysis and model publishing.

## Dataset

The project uses the [Brazilian E-Commerce Public Dataset by Olist](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce).

The original review dataset contains 99,224 records. After removing duplicate review IDs, filtering reviews without text and preparing the data, the final dataset contains **42,370 reviews**.

The sentiment labels were derived from the review score:

| Review score | Sentiment |
|---|---|
| 1–2 | Negative |
| 3 | Neutral |
| 4–5 | Positive |

**Important:** These are proxy labels derived from numerical review scores, not manually annotated sentiment labels. Therefore, some reviews may contain text whose expressed sentiment does not perfectly match the assigned category.

The dataset is imbalanced, with positive reviews representing the majority class.

The project code is licensed under the MIT License. The dataset remains subject to its original license and terms of use.

## Methodology

### 1. Data Preparation

- Analysis of missing values and duplicate review IDs.
- Combination of review titles and review messages.
- Text cleaning and preparation.
- Creation of sentiment labels from review scores.
- Analysis of class distribution.

### 2. Traditional Machine Learning

TF-IDF was used to transform text into numerical features, followed by Logistic Regression models.

Three configurations were evaluated:

1. Logistic Regression with standard TF-IDF.
2. Logistic Regression with `class_weight="balanced"`.
3. Improved TF-IDF with unigrams, bigrams and trigrams, combined with balanced Logistic Regression.

The experiments explored the impact of class weighting and feature extraction improvements on model performance.

### 3. Transformer Fine-Tuning

The pretrained Portuguese language model [BERTimbau](https://huggingface.co/neuralmind/bert-base-portuguese-cased) was fine-tuned for three-class sentiment classification.

Training included:

- Train, validation and test splits.
- Dynamic padding and tokenization.
- Class-weighted loss to address class imbalance.
- GPU training in Google Colab.
- Evaluation using classification metrics.

## Results

Models were evaluated using Accuracy, Macro F1, Weighted F1 and class-specific metrics.

### Traditional Machine Learning

| Model | Accuracy | Macro F1 | Weighted F1 | Neutral F1 |
|---|---:|---:|---:|---:|
| Logistic Regression | 0.8586 | 0.6230 | 0.8319 | 0.1115 |
| Balanced Logistic Regression | 0.7987 | 0.6688 | 0.8213 | 0.3100 |
| Improved TF-IDF + Balanced LR | 0.8060 | 0.6700 | 0.8248 | 0.3047 |

### BERTimbau

| Metric | Test Result |
|---|---:|
| Accuracy | 0.8436 |
| Macro F1 | 0.6935 |
| Weighted F1 | 0.8510 |
| Neutral F1 | 0.3153 |
| Neutral Recall | 0.3582 |

### Key Findings

- Standard Logistic Regression achieved the highest accuracy among the traditional models, but struggled to identify neutral reviews.
- Class weighting substantially improved neutral-class recognition.
- Improved TF-IDF achieved the best Macro F1 among traditional approaches.
- Fine-tuned BERTimbau achieved the best overall Macro F1 (0.6935) among the evaluated models.
- Neutral sentiment remained the most challenging class across all approaches.

These results highlight the importance of evaluating imbalanced classification tasks with metrics beyond accuracy.

## Error Analysis

Error analysis was performed on the balanced Logistic Regression model to investigate misclassified reviews.

Examples revealed challenges involving:

- Short or ambiguous comments.
- Reviews focused on delivery time or logistics.
- Differences between numerical ratings and the sentiment expressed in the text.

This analysis helped identify limitations in the dataset labels and the classification approach.

## Model

The fine-tuned BERTimbau model is publicly available on Hugging Face:

**[Brazilian E-commerce Sentiment Model](https://huggingface.co/brunafagundesb/brazilian-food-delivery-sentiment)**

## Repository Structure

```text
brazilian-food-delivery-nlp/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   ├── data_preparation.py
│   ├── 02_model_training.ipynb
│   └── 03_bertimbau_fine_tuning.ipynb
│
├── reports/
│   ├── model_comparison.csv
│   └── baseline_error_analysis.csv
│
├── docs/
├── src/
├── tests/
│
├── requirements.txt
├── requirements-colab.txt
├── README.md
└── .gitignore
```

The dataset files are not included in this repository. Download the original dataset from Kaggle and place the review CSV in `data/raw/` before running the data preparation workflow.

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- TF-IDF
- Logistic Regression
- Hugging Face Transformers
- BERTimbau
- PyTorch
- Google Colab
- Jupyter Notebook
- Git and GitHub

## Limitations and Future Improvements

- Proxy labels derived from review scores may introduce noise.
- The dataset has significant class imbalance.
- Neutral sentiment remains difficult to classify.
- Future work could explore manual annotation, data augmentation, hyperparameter optimization and additional Portuguese language models.

## Author

**Bruna Fagundes**

GitHub: [@brunafagundesb](https://github.com/brunafagundesb)

Hugging Face: [brunafagundesb](https://huggingface.co/brunafagundesb)
