# 🍽️ Restaurant Recommendation System

This project predicts and recommends restaurants based on user preferences using a machine learning model. It features a **Streamlit web app**, is fully **Dockerized**, and includes modular scripts for preprocessing, training, and recommendation logic.

---

## 🚀 Features

- 🧠 ML-Powered Recommendations: Suggests restaurants based on predicted ratings and user input  
- 📊 Streamlit Interface: Interactive UI for exploring recommendations  
- 🐳 Dockerized App: Containerized for consistent deployment across environments  
- 📁 Modular Codebase: Clean separation of preprocessing, training, and recommendation logic  
- 📈 Exploratory Analysis: Jupyter notebook for data insights and feature exploration  

---

## 🛠️ Tech Stack

| Component      | Tools Used               |
|----------------|--------------------------|
| Web Interface  | Streamlit                |
| ML Pipeline    | scikit-learn, pandas     |
| Containerization| Docker                   |
| Data Analysis  | Jupyter Notebook         |

---

## 📁 Project Structure
RESTAURANT-RECOMMENDATION/ ├── app/ │   └── streamlit_app.py ├── data/ │   └── restaurant_data.csv ├── notebooks/ │   └── exploration.ipynb ├── src/ │   ├── preprocess.py │   ├── recommend.py │   ├── train_model.py │   └── utils.py ├── Dockerfile ├── .dockerignore └── requirements.txt

---

## Project structure

### 1. app/ streamlit_app.py
py```python
import streamlit as st
import pandas as pd
from src.preprocess import preprocess
from src.recommend import recommend

st.title("🍽️ Restaurant Recommendation System")

# Load and preprocess data
X_train, X_test, y_train, y_test = preprocess("data/restaurant_data.csv")
df = pd.read_csv("data/restaurant_data.csv")

# User input
city = st.selectbox("Select City", df["City"].dropna().unique())
cuisine = st.selectbox("Select Cuisine", df["Cuisines"].dropna().unique())
price_range = st.slider("Select Price Range", 1, 4)

user_input = {
    "city": city,
    "cuisines": cuisine,
    "price_range": price_range
}

# Recommend
st.subheader("Recommended Restaurants")
recommendations = recommend(user_input, df)
st.write(recommendations)
```

### 2. data/ restaurant recommendation
```
"add your own data" ```
### 3. notebooks/ explorations.ipynb
 ```python```
### 4. src
###preprocess.py
```python
def preprocess(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')
    df['cuisines'] = df['cuisines'].fillna('Unknown')
    df = df.dropna(subset=['aggregate_rating'])

    drop_cols = ['restaurant_id', 'restaurant_name', 'address', 'locality', 'locality_verbose', 'currency', 'rating_text']
    df = df.drop(columns=drop_cols, errors='ignore')

    df = pd.get_dummies(df, columns=['city', 'cuisines', 'has_table_booking', 'has_online_delivery',
                                     'is_delivering_now', 'switch_to_order_menu', 'rating_color'], drop_first=True)

    scaler = StandardScaler()
    for col in ['price_range', 'votes', 'longitude', 'latitude', 'average_cost_for_two']:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])

    X = df.drop('aggregate_rating', axis=1)
    y = df['aggregate_rating']
    return train_test_split(X, y, test_size=0.2, random_state=42)
###recommend.py
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def recommend(user_input, df, top_n=5):
    user_df = pd.DataFrame([user_input])
    user_df = pd.get_dummies(user_df).reindex(columns=df.columns, fill_value=0)
    similarity = cosine_similarity(user_df, df)
    top_indices = similarity[0].argsort()[-top_n:][::-1]
    return df.iloc[top_indices][['restaurant_name', 'city', 'cuisines', 'aggregate_rating']]
```
###train_model.py
```python
from preprocess import preprocess
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = preprocess('data/restaurant_data.csv')
model = LinearRegression()
model.fit(X_train, y_train)
print("Model trained. R² score:", model.score(X_test, y_test))
```
###utils.py
```python
- normalize_columns(df)
- encode_features(df)
- scale_numerics(df)
- load_data(path)
```
### .dockerignore
```python
__pycache__/
*.pyc
*.ipynb_checkpoints
.env
```
###Dockerfile
```
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY data/ data/
COPY app/ app/
CMD ["streamlit", "run", "app/streamlit_app.py"]
```
###requirements.txt
```
pandas
scikit-learn
streamlit
numpy
```

---

## ⚙️ How to Run

### 5. Build and Run with Docker
```bash
docker build -t restaurant-recommender .
docker run -p 8501:8501 restaurant-recommender
```
### 6.Run without docker
```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```
