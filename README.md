# Support Vector Machine for Text Classification

A from-scratch implementation of Support Vector Machines (SVM) for binary text classification using the Sequential Minimal Optimization (SMO) algorithm.

## Overview

This project implements a complete text classification pipeline featuring:
- Custom TF-IDF vectorization with lemmatization and negation handling
- Three kernel types: Linear, Polynomial, and RBF (Radial Basis Function)
- SMO algorithm for efficient SVM training
- Comprehensive text preprocessing with stop word removal

## Features

- **Text Preprocessing Pipeline**
  - URL and mention removal
  - Negation handling (e.g., "not good" → "not NOT_good")
  - WordNet lemmatization
  - Stop word filtering
  - TF-IDF vectorization with custom vocabulary building

- **Kernel Functions**
  - **Linear**: $K(x, y) = x^T y$
  - **Polynomial**: $K(x, y) = (x^T y + coef0)^{degree}$
  - **RBF**: $K(x, y) = exp(-γ ||x-y||²)$

- **Training Algorithm**
  - Simplified Sequential Minimal Optimization (SMO)
  - KKT condition checking
  - Support vector extraction

## Installation

### Requirements

```bash
pip install numpy pandas scikit-learn nltk
```

### NLTK Data

Download required NLTK data:

```python
import nltk
nltk.download('wordnet')
nltk.download('omw-1.4')
```

## Project Structure

```
SUPPORT_VECTOR_MACHINE/
├── data/
│   ├── df.csv                       # Sample dataset
│   └── Spam_detection.csv           # Spam detection dataset
├── src/
│   └── support_vector_machine.py    # SVM training and prediction
├── utils/
│   ├── criteria.py                  # Text preprocessing and kernels
│   ├── data.py                      # Data loading and splitting
│   ├── utils.py                     # CLI argument parsing
│   └── validation.py                # Input validation
├── .gitignore                       # Git ignore file
├── main.py                          # Entry point
└── README.md                        # Documentation
```

## Usage

### Command Format

```bash
python main.py <kernel> -f <file> -d <text_column> -l <label_column> -c <C> -t <tolerance> -e <epochs> -r <ratio> [kernel_params]
```

### Parameters

#### Common Parameters (all kernels)
- `-f`: Path to CSV file
- `-d`: Column name containing text data
- `-l`: Column name containing labels
- `-c`: Regularization parameter C (positive float)
- `-t`: Tolerance for convergence (positive float)
- `-e`: Number of training epochs (positive integer)
- `-r`: Train/test split ratio (0 < ratio < 1)

#### Kernel-Specific Parameters

**Polynomial Kernel**
- `--coef0`: Independent term (non-negative float)
- `--degree`: Polynomial degree (positive integer)

**RBF Kernel**
- `-g` or `--gamma`: Kernel coefficient (non-negative float)

### Examples

#### Linear Kernel
```bash
python main.py linear
  -f data/reviews.csv
  -d "text"
  -l "sentiment"
  -c 1.0
  -t 0.001
  -e 100
  -r 0.8
```

#### Polynomial Kernel
```bash
python main.py polynomial
  -f data/reviews.csv
  -d "text"
  -l "sentiment"
  -c 1.0
  -t 0.001
  -e 100
  -r 0.8
  --coef0 1.0
  --degree 3
```

#### RBF Kernel
```bash
python main.py rbf
  -f data/reviews.csv
  -d "text"
  -l "sentiment"
  -c 1.0
  -t 0.001
  -e 100
  -r 0.8
  -g 0.1
```

## Input Data Format

The input CSV must contain:
- A text column with string data
- A label column with binary labels (will be mapped to -1 and 1)

Example CSV:
```csv
text,sentiment
"This product is amazing!",positive
"Terrible experience, would not recommend",negative
"Not bad, could be better",neutral
```

## Output

The program prints:
1. **First 5 test predictions** with actual labels
2. **First 5 misclassified samples** (if any)
3. **Evaluation metrics**:
   - Test accuracy
   - Classification report (precision, recall, F1-score)

Example output:
```
 --- --- --- Test set predictions --- --- --- 

   text                              sentiment  prediction
0  great service fast delivery       1          1
1  not impressed with quality        -1         -1
...

 --- --- --- Evaluation --- --- --- 

Test Accuracy: 0.87

              precision    recall  f1-score   support
          -1       0.85      0.88      0.86       120
           1       0.89      0.86      0.87       130
    accuracy                           0.87       250
```

## Algorithm Details

### Text Preprocessing
1. **Cleaning**: Remove URLs, mentions, hashtags, normalize whitespace
2. **Negation Handling**: Prefix words after negations with "NOT_"
3. **Lemmatization**: Reduce words to base form using WordNet
4. **Stop Word Removal**: Filter common words
5. **TF-IDF**: Weight terms by frequency and document rarity

### SMO Training
The Simplified SMO algorithm:
1. Select pairs of Lagrange multipliers (alphas)
2. Check KKT conditions for violations
3. Optimize alpha pairs analytically
4. Update bias term
5. Iterate until convergence or max epochs

### Prediction
Decision function: $f(x) = Σ(αᵢ yᵢ K(xᵢ, x)) + b$

Where:
- $αᵢ$: Lagrange multipliers
- $yᵢ$: Labels
- $K$: Kernel function
- $b$: Bias term

## Hyperparameter Tuning Tips

### C (Regularization)
- **Small C** (0.1 - 1.0): More regularization, simpler decision boundary
- **Large C** (10 - 100): Less regularization, fits training data closely
- Start with C=1.0 and adjust based on overfitting/underfitting

### Tolerance
- Typical range: 0.001 - 0.01
- Smaller values = more precise convergence but slower training

### Epochs
- Start with 100-200 epochs
- Increase if model hasn't converged
- Monitor training progress

### Kernel-Specific

**Polynomial**
- degree=2 or 3 for most tasks
- coef0=0 or 1 typically works well

**RBF**
- gamma=1/(n_features) is a good starting point
- Smaller gamma = smoother decision boundary
- Larger gamma = more complex boundary

## Limitations

- Binary classification only
- Simplified SMO (random second alpha selection)
- No multi-class support
- Memory-intensive for large datasets (full kernel matrix)
- No incremental learning