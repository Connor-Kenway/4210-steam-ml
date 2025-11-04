import pandas as pd

def load():
    df = pd.read_csv("../data/steam_dataset.csv") # TODO: Change to actual data name
    print(f"Loaded {len(df)} rows")
    return df

def load_no_sales():
    df = pd.read_csv("../data/games.csv", nrows=50, index_col=False )# TODO: Change to actual data name
    print(f"Loaded {len(df)} rows")
    return df
