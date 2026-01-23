import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

def load_data(file_path):
    # Load CSV file and ignore commented lines
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        raise FileNotFoundError("File not found.")

    return df

def split_data(df, ratio):
    # Shuffle dataset and split into train and test sets
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    split_idx = int(ratio * len(df))

    train_df = df[:split_idx]
    test_df = df[split_idx:]

    return train_df, test_df
    