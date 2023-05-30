import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from preprocessing.preprocessing import preprocess_data
from models.log_reg_model import LogRegModel
from models.rf_model import RFModel
from models.gb_model import GBModel
import joblib

# Load and preprocess data
df = pd.read_csv('data/dataset.csv')
X, y = preprocess_data('data/dataset.csv')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if the model.pkl file exists
if os.path.exists('model.pkl'):
    # Load the model from the pickle file
    best_model = joblib.load('model.pkl')
else:
    # Define models
    models = [LogRegModel(), RFModel(), GBModel()]

    # Train and evaluate models
    best_model = None
    best_auc = 0
    for model in models:
        model.train(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]  # Change here
        auc = roc_auc_score(y_test, y_pred_proba)  # And here
        print(f'Model: {model.__class__.__name__}, AUC: {auc}')
        if auc > best_auc:
            best_model = model
            best_auc = auc

    print(f'Best model: {best_model.__class__.__name__}, AUC: {best_auc}')

    # Save the best model
    joblib.dump(best_model, 'model.pkl')

# Use the model to predict probabilities for all data
y_proba = best_model.predict_proba(X)[:, 1]

# Create a new dataframe with customer ids and predicted probabilities
output_df = pd.DataFrame({
    'customer_id': df['uuid'],
    'probability': y_proba
})

# Write the dataframe to a new csv file
output_df.to_csv('output.csv', index=False)
