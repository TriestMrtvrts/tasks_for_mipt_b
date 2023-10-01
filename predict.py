# Import necessary libraries
import pandas as pd
import joblib

# Load and preprocess test data
test_data = pd.read_csv('/content/mars-private_test-class.csv')

# Load the trained model
ensemble_model = joblib.load("./ens_mod_class.joblib")

# Make predictions
ensemble_predictions = ensemble_model.predict(test_data)

# Create a DataFrame with predictions
result_df = pd.DataFrame({'Тип марсианина': ensemble_predictions})

# Save predictions to a CSV file
result_df.to_csv('mars-predictions_ens_of_3.csv', index=False)
