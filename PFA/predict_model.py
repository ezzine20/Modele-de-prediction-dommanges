# use_model.py
import joblib

# Load the trained model
model = joblib.load('trained_model.pkl')
print("Model loaded.")

# Get user input for distance, piece, and partie
while True:
    dist = (input('Enter distance: '))
    if dist=='done':
        exit()
    dist = float(dist)
    idP = float(input('Enter piece ID: '))
    idT = float(input('Enter partie ID: '))
    # Predict using the trained model
    output = model.predict([[dist, idP, idT]])
    print(f"Predicted impact percentage: {output[0]}")
    