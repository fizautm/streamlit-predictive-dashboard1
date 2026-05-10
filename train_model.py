import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import joblib

# Load dataset
df = pd.read_csv("data/sales.csv")

# Remove missing values
df = df.dropna()

# Define features and target
X = df[["units_sold","region","product"]]
y = df["revenue"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training samples:",len(X_train))
print("Testing samples:",len(X_test))

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    [("onehot",OneHotEncoder(),["region","product"])],
    remainder="passthrough"
)

# Model pipeline
model = Pipeline([
    ("preprocess",preprocessor),
    ("regressor",LinearRegression())
])

# Train model
model.fit(X_train,y_train)

# Evaluate model
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test,predictions)

print("Model Mean Absolute Error:",mae)

# Save model
joblib.dump(model,"models/revenue_model.pkl")

print("Model saved successfully.")

