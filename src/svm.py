# Import necessary libraries
import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import PCA
import re

from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score, f1_score

# Shuffle and split the data into train, validation, and test sets
def split_data(df):
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)  # Shuffle dataset
    
    train_size = int(0.8 * len(df))
    train_df = df[:train_size]
    test_df = df[train_size:]
    
    return train_df, test_df

# Basic cleaning of tweets (remove links, mentions, hashtags, and whitespace)
def clean_text(text):
    if pd.isna(text):
        return ""
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)  # Remove URLs
    text = re.sub(r'@\w+|#\w+', '', text)  # Remove mentions and hashtags
    text = re.sub(r'\s+', ' ', text).strip()  # Remove extra spaces
    
    return text.lower()  # Convert to lowercase

# Handle negation by prefixing the next word with "NOT_"
def handle_negations(text):
    negations = [
        "not", "no", "never", "neither", "nobody", "nothing", "nowhere",
        "isn't", "wasn't", "shouldn't", "wouldn't", "couldn't", "won't",
        "can't", "don't", "doesn't", "didn't"
    ]
    
    words = text.split()
    for i, word in enumerate(words):
        if word in negations and i + 1 < len(words):
            words[i + 1] = "NOT_" + words[i + 1]
    
    return " ".join(words)

# Tokenize and preprocess text data; optionally build or use a given vocabulary
def tokenization(df, min_freq=1, vocab=None):
    def lemmatize_word(word):
        lemmatizer = WordNetLemmatizer()
        if word.startswith('NOT_'):
            return 'NOT_' + lemmatizer.lemmatize(word[4:])
        return lemmatizer.lemmatize(word)

    def process_text_fully(text):
        if pd.isna(text) or text == "":
            return ""
        words = text.split()
        processed_words = []

        for word in words:
            if re.match(r'^[a-zA-Z_]+$', word):
                lemmatized = lemmatize_word(word)
                if lemmatized not in stop_words:
                    processed_words.append(lemmatized)
        return " ".join(processed_words)

    # Define stop words (words to ignore during processing)
    stop_words = {
        # Common English stop words
        "a", "an", "and", "are", "as", "at", "be", "been", "by", "for", "from",
        "has", "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", 
        "was", "were", "will", "with", "would", "have", "had", "this", "these", 
        "they", "them", "their", "there", "then", "than", "or", "do", "does", 
        "did", "doing", "done", "can", "could", "should", "shall", "may", 
        "might", "must", "ought", "i", "you", "your", "yours", "we", "our", 
        "ours", "she", "her", "hers", "him", "his", "us", "me", "my", "mine", 
        "am", "up", "down", "out", "off", "over", "under", "again", "further", 
        "once", "here", "where", "when", "why", "how", "all", "any", "both", 
        "each", "few", "more", "most", "other", "some", "such", "only", "own", 
        "same", "so", "through", "until", "while", "about", "against", "between", 
        "into", "during", "before", "after", "above", "below", "if", "because", 
        "since", "what", "which", "who", "whom", "whose", "get", "go", "going", 
        "gone", "got", "getting", "come", "coming", "came", "take", "taking", 
        "took", "taken", "put", "putting", "make", "making", "made", "see", 
        "seeing", "saw", "seen"
    }

    df_copy = df.copy()
    df_copy['processed_text'] = df_copy['text'].apply(lambda x: handle_negations(x) if pd.notna(x) and x != "" else "")
    df_copy['fully_processed_text'] = df_copy['processed_text'].apply(process_text_fully)

    if vocab is None:
        # Build vocabulary
        words_series = df_copy['fully_processed_text'].str.split().explode().dropna()
        words_series = words_series[words_series != ""]
        word_counts = words_series.value_counts()

        tweet_counts = {}
        for word in word_counts.index:
            if word_counts[word] >= min_freq:
                tweet_count = df_copy['fully_processed_text'].apply(lambda x: word in x.split()).sum()
                tweet_counts[word] = tweet_count

        vocab = {
            word: [int(word_counts[word]), int(tweet_counts[word])]
            for word in word_counts.index if word_counts[word] >= min_freq
        }

    return df_copy, vocab

# Validate if a word contains only letters
def is_valid_word(word):
    if pd.isna(word):
        return False
    return bool(re.match(r'^[a-zA-Z]+$', str(word)))

# Compute TF-IDF score for a given word
def tf_idf(vocab, word, count, number_of_tweets):
    tf = 1 + np.log(count)
    idf = np.log(number_of_tweets / (vocab[word][1] + 1)) + 1
    return float(tf * idf)

# Generate TF-IDF matrix from dataset and vocabulary
def tf_idf_matrix(df, vocab):
    if vocab is None:
        clean_df, vocab = tokenization(df)
    else:
        clean_df, _ = tokenization(df, vocab=vocab)

    scof_x_i = []
    for _, tweet in clean_df.iterrows():
        tf_idf_vector = []
        text = tweet['fully_processed_text']

        for word in vocab.keys():
            words = text.split() if pd.notna(text) and text.strip() else []
            if word in words:
                count = words.count(word)
                tf_idf_vector.append(tf_idf(vocab, word, count, len(clean_df)))
            else:
                tf_idf_vector.append(0)
        scof_x_i.append(tf_idf_vector)

    return scof_x_i, vocab

def linear_kernel(X):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = np.dot(X[i], X[j])
    
    return K

def polynomial_kernel(X, coef0, degree):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            K[i, j] = (np.dot(X[i], X[j]) + coef0) ** degree
    
    return K

def rbf_kernel(X, gamma):
    n_samples = X.shape[0]
    K = np.zeros((n_samples, n_samples))

    for i in range(n_samples):
        for j in range(n_samples):
            diff = X[i] - X[j]
            K[i, j] = np.exp(-gamma * np.dot(diff, diff))
    
    return K

def train_svm(X, labels, kernel, C, tolerance, epochs, **kwargs):

    n_samples  = len(X)
    alphas = np.zeros((n_samples, 1))
    bias = 0

    kernel = kernel.lower() if isinstance(kernel, str) else "linear"

    kernel_parameters = {}

    for k, val in kwargs.items():
        kernel_parameters[k] = val  # Fill the dictionary

    if kernel  ==  "linear":
        K = linear_kernel(X)
            
    elif  kernel  ==  "polynomial":
        K = polynomial_kernel(X, kernel_parameters["coef0"], kernel_parameters["degree"])

    elif  kernel  ==  "rbf":
        K = rbf_kernel(X, kernel_parameters["gamma"])

    else:
        raise ValueError(f"Unknown kernel: {kernel}")

    for epoch in range(1, epochs):

        for i in range(n_samples):
            f_x_i  = 0
            y_i = labels[i]

            for j in range(n_samples):
                y_j = labels[j]
                f_x_i += alphas[j] * y_j * K[i,j]

            f_x_i += bias
            error_i = f_x_i - y_i

            kkt_violated = False

            if alphas[i] < 1e-5 and y_i * f_x_i < 1 - tolerance:
                kkt_violated = True
            elif 0 < alphas[i] < C and abs(y_i * f_x_i - 1) > tolerance:
                kkt_violated = True
            elif alphas[i] > C - 1e-5 and y_i * f_x_i > 1 + tolerance:
                kkt_violated = True
            
            if kkt_violated:
                
                j = np.random.choice([x for x in range(n_samples) if x != i])

                f_x_j  = 0
                y_j = labels[j]

                for k in range(n_samples):
                    y_k = labels[k]  
                    f_x_j += alphas[k] * y_k * K[i,j]

                f_x_j += bias
                error_j = f_x_j - y_j

                eta = K[i, i] + K[j, j] - 2 * K[i, j]

                if eta <= 0:
                    continue

                new_alpha_j = alphas[j] + (y_j * (error_i - error_j)) / eta

                if y_i  ==  y_j:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min (C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min (C, C + alphas[j] - alphas[i])

                if new_alpha_j < L:
                    new_alpha_j = L

                if new_alpha_j > H:
                    new_alpha_j = H

                if np.abs(new_alpha_j - alphas[j]) < tolerance:
                    continue

                new_alpha_i = alphas[i] + y_i * y_j * (alphas[j] - new_alpha_j)

                kernel_i_i = np.dot(X[i], X[i])
                kernel_j_j = np.dot(X[j], X[j])
                kernel_j_i = np.dot(X[j], X[i])
                b_1 = bias - error_i - y_i * (new_alpha_i - alphas[i]) * kernel_i_i - y_j * (new_alpha_j - alphas[j]) * kernel_j_i
                b_2 = bias - error_j - y_i * (new_alpha_i - alphas[i]) * kernel_j_i - y_j * (new_alpha_j - alphas[j]) * kernel_j_j
                
                
                if 0 < new_alpha_i < C:
                    bias = b_1
                elif 0 < new_alpha_j < C:
                    bias = b_2 
                else:
                    bias = (b_1 + b_2) / 2

                alphas[i] = new_alpha_i
                alphas[j] = new_alpha_j
    
    sv_indices = np.where(alphas.flatten() > 0)[0]
    sv_labels = labels.iloc[(alphas > 0).flatten()]
    sv_alphas = alphas[sv_indices].flatten()
    sv_vectors = [X[i] for i in sv_indices]
    sv_labels = labels.iloc[sv_indices].values

    return sv_vectors, sv_alphas, sv_labels, float(bias)

def predict(train_df, test_df, kernel, C, tolerance, epochs, **kwargs):
    
    tf_idf_mtx, vocab = tf_idf_matrix(train_df, None)
    tf_idf_mtx = normalize(tf_idf_mtx, norm='l2')

    support_vectors, sv_alphas, sv_labels, bias = train_svm(tf_idf_mtx, train_df["label"], kernel, C, tolerance, epochs, **kwargs)

    predictions = []
    X, _ = tf_idf_matrix(test_df, vocab)
    X = normalize(X, norm='l2')
    n_samples  = len(X)

    kernel = kernel.lower() if isinstance(kernel, str) else "linear"

    kernel_parameters = {}

    for k, val in kwargs.items():
        kernel_parameters[k] = val

    for i in range(n_samples): 
        f_x = 0

        for sv, alpha, label in zip(support_vectors, sv_alphas, sv_labels):
            
            if kernel  ==  "linear":
                K = np.dot(sv, X[i])
                    
            elif  kernel  ==  "polynomial":
                K = (np.dot(sv, X[i]) + kernel_parameters["coef0"]) ** kernel_parameters["degree"]

            elif  kernel  ==  "rbf":
                diff = sv - X[i]
                K = np.exp(-kernel_parameters["gamma"] * np.dot(diff, diff))

            else:
                raise ValueError(f"Unknown kernel: {kernel}")
            
            f_x += (alpha * label * K)

        f_x += bias 
        predictions.append(1 if f_x >= 0 else -1)
    
    return predictions        

if __name__ == "__main__":
    
    df = pd.read_csv("Spam_detection.csv")
    df = df.dropna()

    # Clean the tweets
    df["label"] = np.where(df["label"]  ==  "ham", -1, 1)
    df["text"] = df["text"].apply(clean_text)

    # Split dataset
    train_df, test_df = split_data(df)

    # Kernel options: "linear", "polynomial", "rbf"
    predictions = predict(train_df, test_df, "linear", 10, 1e-3, 5)
    test_df.insert(1, "prediction", predictions)
    print(test_df.head(10))
    print("\n --- --- --- Misclassified samples --- --- --- \n")
    print(test_df[test_df["prediction"] != test_df["label"]])

    print("\n --- --- --- Evaluation --- --- --- \n")
    
    correct = sum(1 for true, pred in zip(test_df["label"], test_df["prediction"]) if true == pred)
    accuracy = correct / len(test_df)
    print(f"Test Accuracy: {accuracy:.2f}")

    print(classification_report(test_df["label"], test_df["prediction"]))

    # --- --- --- Plot --- --- --- 

    X, _ = tf_idf_matrix(train_df, None)
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    X1 = X_reduced[:, 0]
    X2 = X_reduced[:, 1]

    plt.figure(figsize=(6, 4))
    plt.scatter(X1, X2, c = train_df["label"], s = 50, edgecolors = "black")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.title("Data Visualization")
    plt.grid(True)
    plt.tight_layout()
    plt.show()