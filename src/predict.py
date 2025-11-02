import joblib
import pandas as pd

model = joblib.load("../models/logreg_model.pkl")

def predict_game(price, rating_pct, age_years, is_new_low=False):
    data = pd.DataFrame([{
        "Price": price,
        "discount_pct": 0,          # assume not on sale yet
        "rating_pct": rating_pct,
        "game_age_years": age_years,
        "ends_days": 30,
        "started_days_ago": 0,
        "is_new_low": is_new_low
    }])
    prob = model.predict_proba(data)[0][1]
    return f"{prob:.1%} chance of going on sale soon"

# Example
print(predict_game(price=59.99, rating_pct=92, age_years=2))