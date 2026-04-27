# model/test_model_load.py
import joblib
import pandas as pd

model = joblib.load("saved_model.pkl")
print("✅ Model loaded successfully!")
print(f"   Model type: {type(model).__name__}")

# Test it can make a prediction with exactly the 13 features expected by the model
sample = pd.DataFrame([{
    "age": 35,
    "workclass": 0,
    "education": 10,
    "education_num": 10,
    "marital_status": 0,
    "occupation": 0,
    "relationship": 0,
    "race": 0,
    "sex": 0,
    "capital_gain": 0,
    "capital_loss": 0,
    "hours_per_week": 40,
    "native_country": 0
}])
pred = model.predict(sample)
print(f"   Test prediction: {pred[0]}")
