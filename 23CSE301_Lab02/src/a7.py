import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity


def load_data():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, "..", "datasets", "Lab Session Data.xlsx")

    return pd.read_excel(
        file_path,
        sheet_name="thyroid0387_UCI",
        na_values=["?"]
    )


def preprocess_data(df):

    encoder = LabelEncoder()

    for column in df.columns:

        if pd.api.types.is_numeric_dtype(df[column]):
            df[column] = df[column].fillna(df[column].mean())

        else:
            df[column] = df[column].fillna(df[column].mode()[0])
            df[column] = encoder.fit_transform(df[column].astype(str))

    return df


def main():

    df = load_data()

    df = preprocess_data(df)

    # Use first 20 observations (recommended by the assignment)
    data = df.iloc[:20]

    similarity_matrix = cosine_similarity(data)

    print("Cosine Similarity Matrix:\n")
    print(similarity_matrix)

    plt.figure(figsize=(10, 8))

    sns.heatmap(
        similarity_matrix,
        annot=True,
        cmap="viridis",
        fmt=".2f",
        square=True
    )

    plt.title("Cosine Similarity Heatmap (First 20 Observations)")
    plt.xlabel("Observations")
    plt.ylabel("Observations")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()