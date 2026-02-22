import joblib
model = joblib.load('models/model.pkl')
print(f"Classes: {model.classes_}")
