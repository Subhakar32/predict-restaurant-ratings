import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess(path):
    # Load the dataset
    df = pd.read_csv(path)
    print("Columns in dataset:", df.columns.tolist())

    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

    # Fill missing values in 'cuisines' with 'Unknown'
    df['cuisines'] = df['cuisines'].fillna('Unknown')

    # Drop rows with missing 'aggregate_rating'
    df = df.dropna(subset=['aggregate_rating'])

    # Drop non-numeric or irrelevant columns
    drop_cols = [
        'restaurant_id', 'restaurant_name', 'address', 'locality',
        'locality_verbose', 'currency', 'rating_text'
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    # Encode categorical features
    df = pd.get_dummies(df, columns=[
        'city', 'cuisines', 'has_table_booking', 'has_online_delivery',
        'is_delivering_now', 'switch_to_order_menu', 'rating_color'
    ], drop_first=True)

    # Scale numeric features
    scaler = StandardScaler()
    numeric_cols = ['price_range', 'votes', 'longitude', 'latitude', 'average_cost_for_two']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = scaler.fit_transform(df[[col]])

    # Split features and target
    X = df.drop('aggregate_rating', axis=1)
    y = df['aggregate_rating']

    return train_test_split(X, y, test_size=0.2, random_state=42)