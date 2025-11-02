import pandas as pd

def load():
    df = pd.read_csv("../data/steam_dataset.csv") # TODO: Change to actual data name
    print(f"Loaded {len(df)} rows")
    return df