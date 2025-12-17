from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import joblib
from backend.utils import load_data

# Load regression part of data
_, _, _, _, X_train_r, X_test_r, y_train_r, y_test_r = load_data()

regressor_model = LinearRegression()
regressor_model.fit(X_train_r, y_train_r)
y_pred = regressor_model.predict(X_test_r)

mae = mean_absolute_error(y_test_r, y_pred)
mse = mean_squared_error(y_test_r, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test_r, y_pred)

print("Regression Model Evaluation")
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.4f}")

# Save model
joblib.dump(regressor_model, "backend/models/regressor_model.pkl")
print("Saved: backend/models/regressor_model.pkl")
