import numpy as np
from sklearn.preprocessing import normalize

import utils.criteria as criteria

def train_support_vector_machine(X, labels, kernel, C, tolerance, epochs, kernel_parameters):
    # Train a Support Vector Machine using simplified SMO.
    # Returns support vectors, their alphas, labels, and the bias term.
    
    n_samples = len(X)
    alphas = np.zeros((n_samples, 1))  # Lagrange multipliers
    bias = 0.0  # Intercept term

    # Compute the kernel matrix
    if kernel == "linear":
        K = criteria.linear_kernel(X)
    elif kernel == "polynomial":
        K = criteria.polynomial_kernel(X, kernel_parameters["coef0"], kernel_parameters["degree"])
    elif kernel == "rbf":
        K = criteria.rbf_kernel(X, kernel_parameters["gamma"])

    # Main SMO optimization loop
    for _ in range(1, epochs):
        for i in range(n_samples):
            f_x_i = 0
            y_i = labels[i]

            # Compute the model output for sample i
            for j in range(n_samples):
                y_j = labels[j]
                f_x_i += alphas[j] * y_j * K[i, j]
            f_x_i += bias

            error_i = f_x_i - y_i

            # Check KKT conditions to see if alpha[i] needs update
            kkt_violated = False
            if alphas[i] < 1e-5 and y_i * f_x_i < 1 - tolerance:
                kkt_violated = True
            elif 0 < alphas[i] < C and abs(y_i * f_x_i - 1) > tolerance:
                kkt_violated = True
            elif alphas[i] > C - 1e-5 and y_i * f_x_i > 1 + tolerance:
                kkt_violated = True

            if kkt_violated:
                # Select second alpha randomly (i != j)
                j = np.random.choice([x for x in range(n_samples) if x != i])
                f_x_j = 0
                y_j = labels[j]

                # Compute output for sample j
                for k in range(n_samples):
                    y_k = labels[k]
                    f_x_j += alphas[k] * y_k * K[i, j]
                f_x_j += bias
                error_j = f_x_j - y_j

                # Compute eta (second derivative of the objective)
                eta = K[i, i] + K[j, j] - 2 * K[i, j]
                if eta <= 0:
                    continue

                # Update alpha[j] with clipped value
                new_alpha_j = alphas[j] + (y_j * (error_i - error_j)) / eta

                # Compute bounds L and H based on constraints
                if y_i == y_j:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                else:
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])

                # Clip alpha[j] to [L, H]
                new_alpha_j = np.clip(new_alpha_j, L, H)

                # Skip tiny updates
                if np.abs(new_alpha_j - alphas[j]) < tolerance:
                    continue

                # Update alpha[i] accordingly
                new_alpha_i = alphas[i] + y_i * y_j * (alphas[j] - new_alpha_j)

                # Update bias term
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

                # Commit updates to alphas
                alphas[i] = new_alpha_i
                alphas[j] = new_alpha_j

    # Extract support vectors (alphas > 0)
    sv_indices = np.where(alphas.flatten() > 0)[0]
    sv_alphas = alphas[sv_indices].flatten()
    sv_vectors = [X[i] for i in sv_indices]
    sv_labels = labels.iloc[sv_indices].values

    return sv_vectors, sv_alphas, sv_labels, float(bias)

def predict(train_df, test_df, text, labels, kernel, C, tolerance, epochs, kernel_parameters):
    # Predict labels for test data using trained SVM model

    # Build TF-IDF matrix for training data
    tf_idf_mtx, vocab = criteria.tf_idf_matrix(train_df, text, None)
    tf_idf_mtx = normalize(tf_idf_mtx, norm='l2')

    # Train SVM model
    support_vectors, sv_alphas, sv_labels, bias = train_support_vector_machine(
        tf_idf_mtx, train_df[labels], kernel, C, tolerance, epochs, kernel_parameters
    )

    # Build TF-IDF for test data using same vocabulary
    X, _ = criteria.tf_idf_matrix(test_df, text, vocab)
    X = normalize(X, norm='l2')
    n_samples = len(X)
    predictions = []

    for i in range(n_samples):
        f_x = 0
        # Compute decision function using support vectors
        for sv, alpha, label in zip(support_vectors, sv_alphas, sv_labels):
            if kernel == "linear":
                K = np.dot(sv, X[i])
            elif kernel == "polynomial":
                K = (np.dot(sv, X[i]) + kernel_parameters["coef0"]) ** kernel_parameters["degree"]
            elif kernel == "rbf":
                diff = sv - X[i]
                K = np.exp(-kernel_parameters["gamma"] * np.dot(diff, diff))

            f_x += alpha * label * K

        f_x += bias
        # Predict class label
        predictions.append(1 if f_x >= 0 else -1)

    return predictions