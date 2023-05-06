import pandas as pd
from sklearn.model_selection import train_test_split
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.arima.model import ARIMAResults
from salesprediction import ProductCatalouge
from pmdarima.arima import auto_arima
import pickle
from pmdarima import auto_arima

# Load the dataset
df = pd.read_csv('cleaned_Laptops.csv')

# Split the dataset into training and test sets
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

# Save the test set to a CSV file
test_data.to_csv('test_data.csv', index=False)

with open('auto_arima_model.pkl', 'wb') as f:
    pickle.dump(auto_arima_model, f)


# Load test data
test_df = pd.read_csv('test.csv', index_col='date', parse_dates=['date'])

# Load the ARIMA model from file
with open('auto_arima_model.pkl', 'rb') as f:
    arima_model = pickle.load(f)

# Generate predictions using the ARIMA model
predictions = arima_model.predict(n_periods=len(test_df))

# Save the predictions to file
with open('arima_predictions.pkl', 'wb') as f:
    pickle.dump(predictions, f)
