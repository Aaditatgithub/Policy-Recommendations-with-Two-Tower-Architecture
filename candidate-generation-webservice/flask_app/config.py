import os

# Base directory of the application
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Database URI for PostgreSQL (update with your PostgreSQL connection details)
DB_URI = "postgresql://postgres:7780@localhost:5432/insurance_recommendation"

# Model path (ensure you have the model saved in the correct location)
MODEL_PATH = os.path.join(BASE_DIR, 'model', 'twotower.h5')

# Preprocessing configuration
PREPROCESS_CONFIG = {
    'customer_numeric_cols': ['age', 'policy_ownership_count', 'credit_score'],
    'customer_categorical_cols': ['gender', 'income_bracket', 'employment_status',
                                  'marital_status', 'location_city', 'preferred_policy_type'],
    'policy_numeric_cols': ['sum_assured', 'premium_amount', 'policy_duration_years'],
    'policy_categorical_cols': ['policy_type', 'risk_category', 'customer_target_group'],
    'interaction_numeric_cols': ['clicked', 'viewed_duration', 'comparison_count', 'abandoned_cart']
}
