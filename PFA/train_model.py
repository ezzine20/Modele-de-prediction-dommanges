# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sqlalchemy import create_engine
import joblib  # For saving the model

# Database connection details
username = 'root'
password = 'yasser_emsi'
host = 'localhost'
port = 3306
database = 'EstimationDommage'

# Create connection URL
connection_url = f'mysql+mysqlconnector://{username}:{password}@{host}:{port}/{database}'

# Create engine object
engine = create_engine(connection_url)

# Import data from tables
estimation_data = pd.read_sql('SELECT * FROM estimation', engine)
pieces_data = pd.read_sql('SELECT * FROM pieces', engine)
marques_data = pd.read_sql('SELECT * FROM marques', engine)
type_data = pd.read_sql('SELECT * FROM type', engine)

# Merge tables
merged_data = estimation_data.merge(pieces_data, left_on='idP', right_on='idPiece')
merged_data = merged_data.merge(marques_data, left_on='idM', right_on='idMarque')
merged_data = merged_data.merge(type_data, left_on='idT', right_on='idT')

# Select relevant columns
X = merged_data[['distance_cm', 'idP', 'idT']]
y = merged_data['impact_percentage']

# Handle missing values
X = X.fillna(0)
y = y.fillna(y.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Absolute Error: {mae}")
print(f"R^2 Score: {r2}")

# Save the trained model to a file
joblib.dump(model, 'trained_model.pkl')
print("Model saved to 'trained_model.pkl'")
