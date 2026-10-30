"""Export the scikit-learn breast cancer dataset to a CSV file versioned by DVC."""
import argparse
from pathlib import Path

from sklearn.datasets import load_breast_cancer

EXPECTED_N_FEATURES = 30
TARGET_COLUMN = "target"
DEFAULT_OUTPUT = Path("data/raw/breast_cancer.csv")


def build_dataframe():
    """Load the dataset as a DataFrame with underscore column names plus a target column."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    df.columns = [column.replace(" ", "_") for column in df.columns]
    return df


def validate(df):
    """Fail loudly if the dataset is not in the expected shape."""
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing target column '{TARGET_COLUMN}'.")
    if df.isnull().values.any():
        raise ValueError("Dataset contains null values.")
    n_features = df.shape[1] - 1
    if n_features != EXPECTED_N_FEATURES:
        raise ValueError(f"Expected {EXPECTED_N_FEATURES} features, got {n_features}.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    df = build_dataframe()
    validate(df)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)

    counts = df[TARGET_COLUMN].value_counts().sort_index()
    print(f"Wrote {len(df)} rows x {df.shape[1]} columns to {args.output}")
    print(f"Class counts: malignant (0) = {counts[0]}, benign (1) = {counts[1]}")


if __name__ == "__main__":
    main()
