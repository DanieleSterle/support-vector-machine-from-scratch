# Spam Detection using Custom SVM with TF-IDF and Preprocessing

## Overview

This project is a spam message classifier using a custom implementation of the Support Vector Machine (SVM) algorithm with options for linear, polynomial, and RBF kernels. It processes raw text data (SMS messages), applies preprocessing techniques including TF-IDF, and classifies messages as spam or ham.

---

## Dataset

The dataset should be a CSV file named `Spam_detection.csv` containing at least the following columns:

- `text`: The message content (SMS text).
- `label`: The label, either `"ham"` or `"spam"`.

---

## Features

- Custom implementation of the SVM training algorithm (SMO-based).
- Preprocessing: URL, mention, and hashtag removal.
- Negation handling by prefixing words with `NOT_`.
- Tokenization, lemmatization, and stop-word filtering.
- TF-IDF vectorization.
- Kernel support: Linear, Polynomial, and RBF.
- PCA visualization of training data.

---

## Preprocessing Steps

1. **Cleaning**: Remove URLs, mentions, hashtags, extra whitespace, and convert to lowercase.
2. **Negation Handling**: Detect negation terms and prefix the next word with `NOT_`.
3. **Lemmatization**: Normalize words using WordNet lemmatizer.
4. **Stop-word Removal**: Remove common English stop-words.
5. **TF-IDF Vectorization**: Transform cleaned text into numerical features.

---

## Model Training

The SVM is trained using a simplified version of the Sequential Minimal Optimization (SMO) algorithm:

- Input: TF-IDF vectors, labels
- Output: Support vectors, Lagrange multipliers (alphas), bias term
- KKT conditions are checked and updated during training.

---

## Kernel Functions

- **Linear**: \( K(x_i, x_j) = x_i \cdot x_j \)
- **Polynomial**: \( K(x_i, x_j) = (x_i \cdot x_j + c_0)^d \)
- **RBF**: \( K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2) \)

---

## Output

- Accuracy and classification report.
- List of misclassified samples.
- PCA scatter plot of the training data.
  
<br />

##### Sample Output

    Test Accuracy: 0.96

              precision    recall  f1-score   support

          -1       0.98      0.99      0.99       865
           1       0.92      0.84      0.88       149

    accuracy                           0.97      1014
    macro avg       0.95      0.91     0.93      1014
    weighted avg    0.97      0.97     0.97      1014
