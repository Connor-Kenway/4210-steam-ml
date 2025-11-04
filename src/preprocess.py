import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def create_features(df: pd.DataFrame):
    # 1. Target: is on sale right now?
    df["on_sale"] = df["Discount"].notna() & (df["Discount"] != "")

    # 2. Clean price
    df["Price"] = df["Price"].replace("[\$,]", "", regex=True).astype(float)

    # 3. Discount % → numeric
    df["discount_pct"] = df["Discount"].replace(
        "[%-]", "", regex=True
    ).replace("", np.nan).astype(float).fillna(0)

    # 4. Rating → numeric
    df["rating_pct"] = df["Rating"].replace("%", "", regex=True).astype(float)

    # 5. Release year
    df["release_year"] = pd.to_datetime(df["Release"], errors="coerce").dt.year
    df["game_age_years"] = 2025 - df["release_year"]

    # 6. Days until sale ends (rough proxy for urgency)
    df["ends_days"] = df["Ends"].str.extract(r"(\d+) days?").astype(float).fillna(30)

    # 7. Started ago → numeric
    df["started_days_ago"] = df["Started"].str.extract(r"(\d+) days?").astype(float).fillna(0)

    # 8. Note flags
    df["is_new_low"] = df["Note"].str.contains("new historical low|all-time low", case=False, na=False)

    num_features = [
        "Price", "discount_pct", "rating_pct",
        "game_age_years", "ends_days", "started_days_ago"
    ]
    cat_features = []  # add "Developer" later if you merge more data
    bool_features = ["is_new_low"]

    X = df[num_features + bool_features]
    y = df["on_sale"].astype(int)

    return X, y, num_features, cat_features, bool_features

def process_no_sales(df_no_sales):
    df_no_sales.rename(columns={"Release date": 'Release'}, inplace=True)
    df_no_sales = df_no_sales.reindex(columns=["Discount", "Rating", "Release", "Ends", "Price", "Started", "Note"])
    
    # Replace features that don't exist with filler values
    for col in df_no_sales:
        if col in ["Ends", "Started", "Note"]: # String value features
            df_no_sales[col] = df_no_sales[col].fillna("")
        else:                                  # Int value features
            if (col == "Discount"):
                df_no_sales[col] = df_no_sales[col].fillna("")
                continue
            df_no_sales[col] = df_no_sales[col].fillna(0)

    # Filter out games that are free
    df_no_sales = df_no_sales[df_no_sales["Price"] != 0]
    return df_no_sales

def build_pipeline(num_features, cat_features):
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ],
        remainder="passthrough"
    )
    return preprocessor