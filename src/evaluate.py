import joblib
from sklearn.metrics import mean_squared_error, r2_score
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess('data/restaurant_data.csv')
model = joblib.load('models/model.pkl')

y_pred = model.predict(X_test)
print(f"MSE: {mean_squared_error(y_test, y_pred):.2f}")
print(f"R²: {r2_score(y_test, y_pred):.2f}")