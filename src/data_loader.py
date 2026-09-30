import pandas as pd


def load_data(file_path):
    df = pd.read_csv(file_path, encoding="latin-1")

    # Keep only required columns
    df = df[['v1', 'v2']]

    # Rename columns
    df.columns = ['label', 'email']

    return df


if __name__ == "__main__":
    df = load_data("data/raw/spam.csv")

    print(df.head())
    print("\nDataset Shape:", df.shape)