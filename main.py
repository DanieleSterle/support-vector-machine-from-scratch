import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report

import utils.validation as validation
import utils.data as data
import utils.utils as utils
import utils.criteria as criteria
import src.support_vector_machine as svm

if __name__ == "__main__":
    
    args = utils.get_argv()

    # Load and validate dataset + CLI arguments
    df = data.load_data(args.file_path)
    df = validation.validate_dataset(df, args.text, args.labels)
    kernel_parameters = validation.validate_arguments(args)

    # Text preprocessing
    df[args.text] = df[args.text].apply(criteria.clean_text)

    # Train / test split
    train_df, test_df = data.split_data(df, args.ratio)

    # Train SVM and generate predictions
    predictions = svm.predict(
        train_df,
        test_df,
        args.text,
        args.labels,
        args.kernel,
        args.C,
        args.tolerance,
        args.epochs,
        kernel_parameters
    )
    test_df["prediction"] = predictions

    print("\n --- --- --- Test set predictions --- --- --- \n")
    print(test_df[[args.text, args.labels, "prediction"]].sample(10))
    
    print("\n --- --- --- Misclassified samples --- --- --- \n")
    print(test_df[test_df["prediction"] != test_df[args.labels]].head(5))

    print("\n --- --- --- Evaluation --- --- --- \n")
    accuracy = np.mean(test_df["prediction"] == test_df[args.labels].values)
    print(f"Test Accuracy: {accuracy:.2f}")
    print(classification_report(test_df[args.labels], test_df["prediction"], zero_division=0))