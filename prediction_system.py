import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import logging


class MultiModelPredictionSystem:
    def __init__(self):
        self.models = {
            'linear_regression': LinearRegression(),
            'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'kmeans': KMeans(n_clusters=3, random_state=42)
        }
        self.scaler = StandardScaler()
        self.label_encoders = {}

    def preprocess_data(self, df, training=True):
        """Preprocess data for training or prediction."""
        processed_df = df.copy()

        # Handle categorical columns
        categorical_cols = ['Product type', 'Customer demographics']
        for col in categorical_cols:
            if col not in processed_df.columns:
                continue

            if training:
                self.label_encoders[col] = LabelEncoder()
                processed_df[col] = self.label_encoders[col].fit_transform(processed_df[col])
            else:
                processed_df[col] = self.label_encoders[col].transform(processed_df[col])

        # Scale numerical features
        numerical_cols = processed_df.select_dtypes(include=['float64', 'int64']).columns
        if training:
            processed_df[numerical_cols] = self.scaler.fit_transform(processed_df[numerical_cols])
        else:
            processed_df[numerical_cols] = self.scaler.transform(processed_df[numerical_cols])

        return processed_df

    def train(self, df):
        """Train models."""
        # Define features and targets
        features = ['Product type', 'Price', 'Availability', 'Customer demographics']
        targets = ['Revenue generated', 'Number of products sold', 'Defect rates']

        processed_df = self.preprocess_data(df, training=True)

        for model_name, model in self.models.items():
            if model_name == 'linear_regression':
                X = processed_df[features]
                y = processed_df['Revenue generated']
                model.fit(X, y)
            elif model_name == 'random_forest':
                X = processed_df[features]
                y = processed_df['Number of products sold']
                model.fit(X, y)
            elif model_name == 'kmeans':
                X = processed_df[features]
                model.fit(X)

    def predict(self, input_data):
        """Make predictions using all models."""
        processed_input = self.preprocess_data(input_data, training=False)
        predictions = {}

        # Linear Regression
        lin_reg_preds = self.models['linear_regression'].predict(processed_input)
        predictions['linear_regression'] = lin_reg_preds[0]

        # Random Forest
        rf_preds = self.models['random_forest'].predict(processed_input)
        predictions['random_forest'] = rf_preds[0]

        # KMeans
        cluster = self.models['kmeans'].predict(processed_input)
        predictions['kmeans_cluster'] = int(cluster[0])

        return predictions

    def save_models(self, path):
        """Save the trained model."""
        joblib.dump(self, path)
        logging.info(f"Models saved to {path}")

    @classmethod
    def load_models(cls, path):
        """Load a saved model."""
        return joblib.load(path)


if __name__ == "__main__":
    # Assuming you have your dataset
    import pandas as pd

    DATA_FILE_PATH = "updated_supply_chain_data.csv"
    MODEL_OUTPUT_PATH = "models/prediction_system_updated.joblib"

    try:
        df = pd.read_csv(DATA_FILE_PATH)

        # Initialize and train the prediction system
        prediction_system = MultiModelPredictionSystem()
        prediction_system.train(df)

        # Save the trained model
        prediction_system.save_models(MODEL_OUTPUT_PATH)
        print(f"Model successfully trained and saved at {MODEL_OUTPUT_PATH}")

    except Exception as e:
        print(f"An error occurred: {e}")
