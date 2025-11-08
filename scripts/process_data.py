import os
import pandas as pd
from sklearn.model_selection import train_test_split

def impute_missing(df):
    # Impute numeric columns using mean of last 10 samples per species
    for col in df.select_dtypes(include='number').columns:
        for species, group in df.groupby('species'):
            mask = df['species'] == species
            mean_val = group[col].tail(10).mean()
            df.loc[mask, col] = df.loc[mask, col].fillna(mean_val)
    return df

def main():
    raw_path = "data/raw"
    processed_path = "data/processed"
    os.makedirs(processed_path, exist_ok=True)

    # Load and merge all CSVs
    csv_files = [f for f in os.listdir(raw_path) if f.endswith(".csv")]
    dfs = [pd.read_csv(os.path.join(raw_path, f)) for f in csv_files]
    df = pd.concat(dfs, ignore_index=True)

    print(f"Loaded {len(df)} rows from {len(csv_files)} CSV files.")

    # Clean / impute
    df = impute_missing(df)

   
    # Encode target (adjusted for your dataset)
    mapping = {"setosa": 0, "versicolor": 1, "virginica": 2}
    df["target"] = df["species"].map(mapping)

    # Drop rows with unknown or missing species
    before = len(df)
    df = df.dropna(subset=["target"])
    after = len(df)
    print(f"Removed {before - after} rows with missing/unknown species.")



    # Split into train/val for reproducibility
    train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["target"])

    # Save outputs
    train_df.to_parquet(os.path.join(processed_path, "train.parquet"), index=False)
    val_df.to_parquet(os.path.join(processed_path, "val.parquet"), index=False)

    print("✅ Data processed and saved to data/processed/")

if __name__ == "__main__":
    main()
