import os
import zipfile
import subprocess
import pandas as pd
import numpy as np
import re
from datetime import datetime



# 1. Setup directories
DATA_DIR = "data"
KAGGLE_DIR = os.path.join(DATA_DIR, "kaggle")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")

os.makedirs(KAGGLE_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)


# 2. Download Kaggle Datasets
def download_kaggle_datasets():
    """
    Uses Kaggle CLI to download two public datasets:
    - fronkongames/steam-games-dataset
    - nikdavis/steam-store-games
    """
    print("Checking Kaggle API setup...")
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        raise FileNotFoundError(
            "Kaggle API credentials not found.\n"
            "Please create an account on Kaggle, "
            "go to 'My Account' → 'API' → 'Create New Token', "
            "and place kaggle.json inside ~/.kaggle/"
        )

    datasets = {
        "fronkongames/steam-games-dataset": {
            "zip": "steam-games-dataset.zip",
            "csv": "games.csv"  # Actual filename in the zip
        },
        "nikdavis/steam-store-games": {
            "zip": "steam-store-games.zip",
            "csv": "steam.csv"  # Actual filename in the zip
        },
    }

    for dataset, files in datasets.items():
        zip_name = files["zip"]
        csv_name = files["csv"]
        
        print(f"Downloading {dataset} ...")
        subprocess.run(
            ["kaggle", "datasets", "download", "-d", dataset, "-p", KAGGLE_DIR, "--force"],
            check=True,
        )

        zip_path = os.path.join(KAGGLE_DIR, zip_name)
        print(f"Extracting {zip_name} (looking for {csv_name})...")
        
        # Extract only the CSV file we need, not everything
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            # Check if our target CSV exists in the zip
            file_list = zip_ref.namelist()
            if csv_name in file_list:
                # Extract only the CSV we need
                zip_ref.extract(csv_name, KAGGLE_DIR)
                print(f"  ✓ Extracted {csv_name}")
            else:
                # If not found, extract all and show what's available
                print(f"  ⚠ {csv_name} not found in zip. Available files:")
                for f in file_list[:10]:  # Show first 10 files
                    print(f"    - {f}")
                zip_ref.extractall(KAGGLE_DIR)
        
        # Remove zip file after extraction (if it exists)
        try:
            if os.path.exists(zip_path):
                os.remove(zip_path)
                print(f"  Removed {zip_name}")
        except OSError as e:
            print(f"  Warning: Could not remove {zip_name}: {e}")

    print("All Kaggle datasets downloaded and extracted.")


# 3. Load Datasets
def load_kaggle_datasets():
    print("Loading CSVs...")
    # Use the actual filenames that get extracted
    kaggle_1_path = os.path.join(KAGGLE_DIR, "games.csv")  # fronkongames
    kaggle_2_path = os.path.join(KAGGLE_DIR, "steam.csv")   # nikdavis
    
    # Check if files exist, if not try alternative names
    if not os.path.exists(kaggle_1_path):
        # Try to find the right file
        possible_names = ["games.csv", "steam_games.csv", "steam-games.csv"]
        for name in possible_names:
            path = os.path.join(KAGGLE_DIR, name)
            if os.path.exists(path):
                kaggle_1_path = path
                print(f"  Found: {name}")
                break
    
    if not os.path.exists(kaggle_2_path):
        possible_names = ["steam.csv", "steam_store_games.csv", "steam-store-games.csv"]
        for name in possible_names:
            path = os.path.join(KAGGLE_DIR, name)
            if os.path.exists(path):
                kaggle_2_path = path
                print(f"  Found: {name}")
                break
    
    kaggle_1 = pd.read_csv(kaggle_1_path, encoding="utf-8")
    kaggle_2 = pd.read_csv(kaggle_2_path, encoding="utf-8")

    print(f"FronkonGames dataset: {len(kaggle_1)} rows, NikDavis dataset: {len(kaggle_2)} rows")
    print(f"FronkonGames dataset columns: {kaggle_1.columns.tolist()}")
    print(f"NikDavis dataset columns: {kaggle_2.columns.tolist()}")
    return kaggle_1, kaggle_2

# 4. Clean and Merge
def clean_and_merge(kaggle_1, kaggle_2):
    print("Cleaning and merging datasets...")
    
    # Print actual column names for debugging
    print(f"kaggle_1 columns: {kaggle_1.columns.tolist()[:10]}...")
    print(f"kaggle_2 columns: {kaggle_2.columns.tolist()[:10]}...")


    
    rename_map_1 = {}
    # Check and rename kaggle_1 (games.csv) columns
    if "app_name" in kaggle_1.columns:
        rename_map_1["app_name"] = "Name"

    
    if "Release date" in kaggle_1.columns:
        rename_map_1["Release date"] = "Release"
    elif "release_date" in kaggle_1.columns:
        rename_map_1["release_date"] = "Release"
    
    if "Price" in kaggle_1.columns:
        rename_map_1["Price"] = "Price"
    elif "price" in kaggle_1.columns:
        rename_map_1["price"] = "Price"
    
    if "Developers" in kaggle_1.columns:
        rename_map_1["Developers"] = "Developer"
    elif "developer" in kaggle_1.columns:
        rename_map_1["developer"] = "Developer"
    
    if "Publishers" in kaggle_1.columns:
        rename_map_1["Publishers"] = "Publisher"
    elif "publisher" in kaggle_1.columns:
        rename_map_1["publisher"] = "Publisher"
    
    if rename_map_1:
        kaggle_1.rename(columns=rename_map_1, inplace=True)
        print(f"Renamed kaggle_1 columns: {rename_map_1}")

    rename_map_2 = {}
    # Check and rename kaggle_2 (steam.csv) columns
    if "name" in kaggle_2.columns:
        rename_map_2["name"] = "Name"
    elif "title" in kaggle_2.columns:
        rename_map_2["title"] = "Name"
    elif "Name" in kaggle_2.columns:
        rename_map_2["Name"] = "Name"
    
    if "release_date" in kaggle_2.columns:
        rename_map_2["release_date"] = "Release"
    elif "Release date" in kaggle_2.columns:
        rename_map_2["Release date"] = "Release"
    
    if "price" in kaggle_2.columns:
        rename_map_2["price"] = "Price"
    elif "Price" in kaggle_2.columns:
        rename_map_2["Price"] = "Price"
    
    if "developer" in kaggle_2.columns:
        rename_map_2["developer"] = "Developer"
    elif "Developers" in kaggle_2.columns:
        rename_map_2["Developers"] = "Developer"
    
    if "publisher" in kaggle_2.columns:
        rename_map_2["publisher"] = "Publisher"
    elif "Publishers" in kaggle_2.columns:
        rename_map_2["Publishers"] = "Publisher"
    
    if "genres" in kaggle_2.columns:
        rename_map_2["genres"] = "Genre"
    elif "genre" in kaggle_2.columns:
        rename_map_2["genre"] = "Genre"
    
    if rename_map_2:
        kaggle_2.rename(columns=rename_map_2, inplace=True)
        print(f"Renamed kaggle_2 columns: {rename_map_2}")

    # Ensure "Name" column exists in both dataframes
    if "Name" not in kaggle_1.columns:
        raise ValueError(f"Name column not found in kaggle_1. Available columns: {kaggle_1.columns.tolist()}")
    if "Name" not in kaggle_2.columns:
        raise ValueError(f"Name column not found in kaggle_2. Available columns: {kaggle_2.columns.tolist()}")

    # --- Normalize casing and handle missing values ---
    for df in [kaggle_1, kaggle_2]:
        df["Name"] = df["Name"].astype(str).str.lower().str.strip()
        # Replace empty strings and 'nan' strings with actual NaN
        df["Name"] = df["Name"].replace(['', 'nan', 'none'], np.nan)

    # --- Merge ---
    # Drop rows where Name is NaN before merging
    kaggle_1_clean = kaggle_1.dropna(subset=["Name"]).copy()
    kaggle_2_clean = kaggle_2.dropna(subset=["Name"]).copy()
    
    print(f"After dropping NaN names: kaggle_1={len(kaggle_1_clean)} rows, kaggle_2={len(kaggle_2_clean)} rows")
    
    # Get columns that exist in both dataframes (excluding Name)
    common_cols = set(kaggle_1_clean.columns) & set(kaggle_2_clean.columns) - {"Name"}
    if common_cols:
        print(f"Warning: Common columns (will get suffixes): {list(common_cols)[:5]}...")
    
    merged_df = pd.merge(
        kaggle_1_clean,
        kaggle_2_clean,
        on="Name",
        how="outer",
        suffixes=("_games", "_store"),
        indicator=True  # Add merge indicator for debugging
    )
    
    print(f"🔗 Merged dataset: {len(merged_df)} rows")
    print(f"Merge statistics: {merged_df['_merge'].value_counts().to_dict()}")
    
    # Drop the merge indicator column
    merged_df.drop(columns=['_merge'], inplace=True, errors='ignore')

    # --- Clean numeric and date fields ---
    # Handle Price column - might have multiple versions (Price, Price_games, Price_store)
    price_cols = [col for col in merged_df.columns if 'Price' in col]
    print(f"Price columns found: {price_cols}")
    
    # Use the first available Price column, or combine them
    if "Price" in merged_df.columns:
        price_col = "Price"
    elif "Price_games" in merged_df.columns:
        price_col = "Price_games"
    elif "Price_store" in merged_df.columns:
        price_col = "Price_store"
    else:
        raise ValueError("No Price column found after merge")
    
    # Create a unified Price column (if needed) and clean it
    # Handle NaN values before string conversion
    price_series = merged_df[price_col].copy()
    merged_df["Price"] = (
        price_series
        .fillna("0")  # Fill NaN with "0" before string conversion
        .astype(str)
        .str.replace(r"[\$,]", "", regex=True)
        .str.extract(r"(\d+(\.\d+)?)")[0]
        .replace("", "0")  # Replace empty strings with "0"
        .astype(float)
    )

    # Handle Release column - might have multiple versions
    release_cols = [col for col in merged_df.columns if 'Release' in col or 'release' in col.lower()]
    print(f"Release columns found: {release_cols}")
    
    # Use the first available Release column, or combine them
    if "Release" in merged_df.columns:
        release_col = "Release"
    elif "Release_games" in merged_df.columns:
        release_col = "Release_games"
    elif "Release_store" in merged_df.columns:
        release_col = "Release_store"
    else:
        # Try to find any release-related column
        release_col = release_cols[0] if release_cols else None
        if release_col:
            merged_df["Release"] = merged_df[release_col]
            release_col = "Release"
    
    if release_col:
        # Convert to datetime, handling errors gracefully
        merged_df["Release"] = pd.to_datetime(merged_df[release_col], errors="coerce")
        merged_df["release_year"] = merged_df["Release"].dt.year
        merged_df["game_age_years"] = datetime.now().year - merged_df["release_year"]
        # Fill NaN years with median
        median_year = merged_df["release_year"].median()
        if pd.notna(median_year):
            merged_df["release_year"].fillna(int(median_year), inplace=True)
            merged_df["game_age_years"] = datetime.now().year - merged_df["release_year"]
    else:
        print("Warning: No Release column found, setting default values")
        merged_df["release_year"] = 2020
        merged_df["game_age_years"] = 5

    # --- Fill missing numeric fields ---
    for col in ["Price", "game_age_years"]:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce")
            median_val = merged_df[col].median()
            if pd.notna(median_val):
                merged_df[col].fillna(median_val, inplace=True)
            else:
                # If all values are NaN, use a default
                merged_df[col].fillna(0 if col == "Price" else 5, inplace=True)

    # --- Create a synthetic on_sale label ---
    # Only create if Price column exists and has valid data
    if "Price" in merged_df.columns and merged_df["Price"].notna().any():
        price_threshold = merged_df["Price"].quantile(0.3)
        merged_df["on_sale"] = (merged_df["Price"] < price_threshold).astype(int)
    else:
        print("Warning: Could not create on_sale label, Price column missing or invalid")
        merged_df["on_sale"] = 0

    return merged_df


# 5. Save Output
def save_dataset(df):
    output_path = os.path.join(PROCESSED_DIR, "steam_kaggle_hybrid.csv")
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
    print(f"Final shape: {df.shape}")


# 6. Main
def main():
    print("Starting Steam Kaggle Data Parser")
    try:
        # Skip download if files already exist (comment out to force download)
        if not (os.path.exists(os.path.join(KAGGLE_DIR, "games.csv")) and 
                os.path.exists(os.path.join(KAGGLE_DIR, "steam.csv"))):
            download_kaggle_datasets()
        else:
            print("CSV files already exist, skipping download.")
        
        kaggle_1, kaggle_2 = load_kaggle_datasets()
        hybrid = clean_and_merge(kaggle_1, kaggle_2)
        save_dataset(hybrid)
        print("Done. Dataset ready for EDA and modeling.")
    except Exception as e:
        print(f"Error in main: {e}")
        raise


if __name__ == "__main__":
    main()
