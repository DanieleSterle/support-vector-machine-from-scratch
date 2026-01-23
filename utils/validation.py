from pandas.api.types import is_object_dtype, is_numeric_dtype

def validate_dataset(df, text, labels):

    # Validate text column
    if text not in df.columns:
        raise ValueError(f"Column '{text}' does not exist in the DataFrame.")

    if not is_object_dtype(df[text]):
        raise ValueError(f"Column '{text}' must be text-like.")

    # Validate labels column
    if labels not in df.columns:
        raise ValueError(f"Column '{labels}' does not exist in the DataFrame.")

    col = df[labels].dropna().unique()
    if len(col) != 2:
        raise ValueError("Column must have exactly 2 unique values")

    mapping = {col[0]: -1, col[1]: 1}
    df[labels] = df[labels].map(mapping)

    return df.dropna()

def validate_arguments(args):

    kernel_parameters = {}

    # Generic SVM hyperparameters
    if args.kernel not in ["linear", "polynomial", "rbf"]:
        raise ValueError("Kernel must be 'linear', 'polynomial', or 'rbf'")

    if args.C <= 0:
        raise ValueError("C must be a positive value")

    if args.tolerance <= 0:
        raise ValueError("Tolerance must be a positive value")

    if args.epochs <= 0:
        raise ValueError("Epochs must be a positive integer")

    if args.ratio <= 0 or args.ratio >= 1:
        raise ValueError("Ratio must be a float value between 0 and 1")

    # Kernel-specific parameters
    if args.kernel == "polynomial":
        if args.coef0 < 0:
            raise ValueError("Coef0 must be non-negative for polynomial kernel")

        if args.degree <= 0:
            raise ValueError("Degree must be a positive integer (at least 1) for polynomial kernel")

        kernel_parameters["coef0"] = args.coef0
        kernel_parameters["degree"] = args.degree

    elif args.kernel == "rbf":
        if args.gamma < 0:
            raise ValueError("Gamma must be non-negative for RBF kernel")

        kernel_parameters["gamma"] = args.gamma

    return kernel_parameters