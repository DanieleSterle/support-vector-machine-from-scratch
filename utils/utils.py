import argparse

def get_argv():

    # Arguments shared across all kernels
    def add_common_args(p):
        p.add_argument("-f", dest="file_path", required=True, type=str, help="Path to .csv file")
        p.add_argument("-d", dest="text", required=True, type=str, help="Column name for text data")
        p.add_argument("-l", dest="labels", required=True, type=str, help="Column name for labels")

        p.add_argument("-c", dest="C", required=True, type=float, help="Regularization parameter C")
        p.add_argument("-t", dest="tolerance", required=True, type=float, help="Tolerance value")
        p.add_argument("-e", dest="epochs", required=True, type=int, help="Number of epochs")
        p.add_argument("-r", dest="ratio", required=True, type=float, help="Train/test split ratio")

    # Parser with subcommands per kernel
    parser = argparse.ArgumentParser(description="Train ML models")
    subparsers = parser.add_subparsers(dest="kernel", required=True, help="Kernel type: linear, polynomial, rbf")

    linear_parser = subparsers.add_parser("linear", help="Linear kernel")
    add_common_args(linear_parser)

    polynomial_parser = subparsers.add_parser("polynomial", help="Polynomial kernel")
    add_common_args(polynomial_parser)
    polynomial_parser.add_argument("--coef0", dest="coef0", required=True, type=float, help="Polynomial kernel coefficient")
    polynomial_parser.add_argument("--degree", dest="degree", required=True, type=int, help="Polynomial degree")

    rbf_parser = subparsers.add_parser("rbf", help="RBF kernel")
    add_common_args(rbf_parser)
    rbf_parser.add_argument("-g", "--gamma", dest="gamma", required=True, type=float, help="RBF kernel gamma")

    return parser.parse_args()