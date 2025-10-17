import joblib
from sklearn.linear_model import LinearRegression
from preprocess import preprocess

X_train, X_test, y_train, y_test = preprocess('data/restaurant_data.csv')

model = LinearRegression()
model.fit(X_train, y_train)

joblib.dump(model, 'models/model.pkl')