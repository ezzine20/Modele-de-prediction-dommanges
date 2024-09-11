import streamlit as st
import joblib

# Charger le modèle entraîné
model = joblib.load('trained_model.pkl')

# Interface utilisateur
st.title("Prédiction des dommages automobiles")

# Formulaire pour entrer les données
distance = st.number_input("Distance (en cm)", min_value=0.0, step=0.1)
id_piece = st.number_input("ID de la pièce", min_value=1, max_value=77)
id_partie = st.number_input("ID de la partie", min_value=1, max_value=8)

# Prédiction du modèle
if st.button("Prédire"):
    prediction = model.predict([[distance, id_piece, id_partie]])
    st.write(f"Pourcentage d'impact prédit : {prediction[0]}")
