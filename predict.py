import pandas as pd
import joblib
import os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, accuracy_score


class SupplyChainPredictionSystem:
    def __init__(self, data_path=None):
        self.pipelines = {
            'sales': None,
            'revenue': None,
            'shipping': None,
            'transportation': None,
            'routes': None
        }
        self.required_features = [
            'Product type', 'Price', 'Availability', 'Customer demographics',
            'Location', 'Shipping carriers', 'Transportation modes', 'Routes',
            'Number of products sold', 'Revenue generated'
        ]
        self.data = None
        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path):
        """Load and validate training data from CSV"""
        try:
            self.data = pd.read_csv("updated_supply_chain_data.csv")
            missing_cols = set(self.required_features) - set(self.data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def create_pipeline(self, model_type='regressor'):
        """Create preprocessing and model pipeline"""
        categorical_features = ['Product type', 'Customer demographics', 'Location']
        numerical_features = ['Price', 'Availability']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        model = RandomForestRegressor(n_estimators=200, random_state=42) if model_type == 'regressor' \
            else RandomForestClassifier(n_estimators=200, random_state=42)

        return Pipeline([
            ('preprocessor', preprocessor),
            ('model', model)
        ])

    def train(self):
        """Train all models using loaded data"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        X = self.data[self.required_features[:-2]]
        targets = {
            'sales': ('Number of products sold', 'regressor'),
            'revenue': ('Revenue generated', 'regressor'),
            'shipping': ('Shipping carriers', 'classifier'),
            'transportation': ('Transportation modes', 'classifier'),
            'routes': ('Routes', 'classifier')
        }

        for name, (target_col, model_type) in targets.items():
            pipeline = self.create_pipeline(model_type)
            X_train, X_test, y_train, y_test = train_test_split(
                X, self.data[target_col], test_size=0.2, random_state=42
            )
            pipeline.fit(X_train, y_train)
            self.pipelines[name] = pipeline

            # Evaluate model
            y_pred = pipeline.predict(X_test)
            metric = mean_squared_error if model_type == 'regressor' else accuracy_score
            score = metric(y_test, y_pred)
            print(f"{name.capitalize()} model score: {score:.4f}")

    def predict(self, features_dict):
        """Make predictions using trained models"""
        if not all(self.pipelines.values()):
            raise ValueError("Models not trained. Call train() first.")

        input_df = pd.DataFrame([features_dict])

        return {
            'Number of products sold': int(self.pipelines['sales'].predict(input_df)[0]),
            'Revenue generated': round(float(self.pipelines['revenue'].predict(input_df)[0]), 2),
            'Best shipping carrier': self.pipelines['shipping'].predict(input_df)[0],
            'Best transportation mode': self.pipelines['transportation'].predict(input_df)[0],
            'Best route': self.pipelines['routes'].predict(input_df)[0]
        }

    def save_models(self, path):
        """Save trained models"""
        if not os.path.exists(path):
            os.makedirs(path)
        for name, pipeline in self.pipelines.items():
            if pipeline:
                joblib.dump(pipeline, os.path.join(path, f'{name}_pipeline.joblib'))

    def load_models(self, path):
        """Load saved models"""
        for name in self.pipelines:
            model_path = os.path.join(path, f'{name}_pipeline.joblib')
            if os.path.exists(model_path):
                self.pipelines[name] = joblib.load(model_path)
            else:
                print(f"No saved model found for {name}")