import pandas as pd
import joblib
import os
import re
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
            'SKU', 'Product type', 'Price', 'Availability', 'Customer demographics',
            'Location', 'Shipping carriers', 'Transportation modes', 'Routes',
            'Number of products sold', 'Revenue generated'
        ]
        self.data = None
        if data_path:
            self.load_data(data_path)

    def validate_sku(self, sku):
        """Validate SKU format using regex"""
        if not isinstance(sku, str):
            return False
        return bool(re.match(r'^SKU[0-9]{1,2}$', sku))

    def load_data(self, data_path):
        """Load and validate training data from CSV"""
        try:
            self.data = pd.read_csv("updated_supply_chain_data.csv")
            if 'SKU' in self.data.columns:
                invalid_skus = [sku for sku in self.data['SKU'] if not self.validate_sku(sku)]
                if invalid_skus:
                    print(f"Warning: Found invalid SKU formats: {invalid_skus[:5]}...")

            missing_cols = set(self.required_features) - set(self.data.columns)
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
        except Exception as e:
            raise Exception(f"Error loading data: {str(e)}")

    def create_pipeline(self, model_type='regressor'):
        """Create preprocessing and model pipeline"""
        categorical_features = ['SKU', 'Product type', 'Customer demographics', 'Location']
        numerical_features = ['Price', 'Availability']

        preprocessor = ColumnTransformer([
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])

        if model_type == 'regressor':
            model = RandomForestRegressor(
                n_estimators=200,
                random_state=42,
                max_features='sqrt',  # Added to handle increased feature dimension from SKU
                min_samples_leaf=2  # Helps prevent overfitting with more granular data
            )
        else:
            model = RandomForestClassifier(
                n_estimators=200,
                random_state=42,
                max_features='sqrt',
                min_samples_leaf=2
            )

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

        print("Training models with SKU features...")
        for name, (target_col, model_type) in targets.items():
            pipeline = self.create_pipeline(model_type)
            X_train, X_test, y_train, y_test = train_test_split(
                X, self.data[target_col], test_size=0.2, random_state=42,
                stratify=self.data['SKU'] if model_type == 'classifier' else None
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

        # Create a copy of features_dict to avoid modifying the original
        features = features_dict.copy()

        # Validate SKU format if present, but don't block prediction on invalid format
        if 'SKU' in features:
            if not self.validate_sku(features['SKU']):
                print(f"Warning: SKU format '{features['SKU']}' does not match expected pattern (SKU0-SKU99)")

        # Ensure all required features are present
        required_pred_features = ['SKU', 'Product type', 'Price', 'Availability', 'Customer demographics',
                                  'Location']
        missing_features = set(required_pred_features) - set(features.keys())
        if missing_features:
            raise ValueError(f"Missing required features for prediction: {missing_features}")

        input_df = pd.DataFrame([features_dict])

        try:
            return {
                'Number of products sold': int(self.pipelines['sales'].predict(input_df)[0]),
                'Revenue generated': round(float(self.pipelines['revenue'].predict(input_df)[0]), 2),
                'Best shipping carrier': self.pipelines['shipping'].predict(input_df)[0],
                'Best transportation mode': self.pipelines['transportation'].predict(input_df)[0],
                'Best route': self.pipelines['routes'].predict(input_df)[0]
            }
        except Exception as e:
            raise Exception(f"Prediction error: {str(e)}")

    def get_sku_analysis(self, sku):
        """Get aggregate analysis for a specific SKU"""
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")

        try:
            sku_data = self.data[self.data['SKU'] == sku]
            if sku_data.empty:
                raise ValueError(f"No data found for {sku}")

            return {
                'average_sales': sku_data['Number of products sold'].mean(),
                'average_revenue': sku_data['Revenue generated'].mean(),
                'most_common_shipping': sku_data['Shipping carriers'].mode().iloc[0],
                'most_common_transport': sku_data['Transportation modes'].mode().iloc[0],
                'most_successful_route': sku_data['Routes'].mode().iloc[0],
                'total_revenue': sku_data['Revenue generated'].sum()
            }
        except Exception as e:
            # Return default values on error
            print(f"Warning: Error in SKU analysis: {str(e)}")
            return {
                'average_sales': 0,
                'average_revenue': 0,
                'most_common_shipping': 'N/A',
                'most_common_transport': 'N/A',
                'most_successful_route': 'N/A',
                'total_revenue': 0
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
