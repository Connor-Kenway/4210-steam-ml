import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.inspection import DecisionBoundaryDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from load_data import load, load_no_sales
from preprocess import create_features, build_pipeline, process_no_sales

df = load()
X, y, num_f, cat_f, _ = create_features(df)

df_no_sales = load_no_sales()
df_no_sales = process_no_sales(df_no_sales)

print(df_no_sales)
X2, y2, _, _, _ = create_features(df_no_sales)

X = pd.concat([X, X2])
y = pd.concat([y, y2])
    

preprocessor = build_pipeline(num_f, cat_f)

model = Pipeline(steps=[
    ("prep", preprocessor),
    ("clf", LogisticRegression(class_weight="balanced", max_iter=1000))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix")
plt.show()

# Save
joblib.dump(model, "../models/logreg_model.pkl")
print("Model saved → models/logreg_model.pkl")