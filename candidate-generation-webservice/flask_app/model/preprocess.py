import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sqlalchemy import create_engine
from config import DB_URI

# Create a database connection
engine = create_engine(DB_URI)

def preprocess_data(customer_df, policy_df, interaction_df, config):
    """
    Preprocesses customer, policy, and interaction data using freshly fitted transformers.
    """
    # Fetch full customer data from the database to fit transformers
    full_customers = pd.read_sql("SELECT * FROM public.customers", engine)
    full_customer_features = full_customers[config['customer_numeric_cols'] + config['customer_categorical_cols']]
    
    # Fit customer preprocessor
    customer_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), config['customer_numeric_cols']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), config['customer_categorical_cols'])
        ]
    )
    customer_preprocessor.fit(full_customer_features)
    customer_features = customer_preprocessor.transform(customer_df)
    
    # Fetch full policy data from the database to fit transformers
    policies_full = pd.read_sql("SELECT * FROM public.policies", engine)
    interactions_full = pd.read_sql("SELECT * FROM public.interactions", engine)
    
    # Merge to reproduce training data (as done in the Colab notebook)
    data_full = interactions_full.merge(full_customers, on='customer_id', how='left') \
                                 .merge(policies_full, on='policy_id', how='left')
    policy_fit_df = data_full[config['policy_numeric_cols'] + config['policy_categorical_cols']]
    
    # Clean numeric columns in policy data
    for col in ['sum_assured', 'premium_amount']:
        if col in policy_fit_df.columns:
            policy_fit_df[col] = policy_fit_df[col].astype(str).str.replace(',', '').astype(float)
    
    # Fit policy preprocessor
    policy_preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), config['policy_numeric_cols']),
            ('cat', OneHotEncoder(handle_unknown='ignore'), config['policy_categorical_cols'])
        ]
    )
    policy_preprocessor.fit(policy_fit_df)
    policy_features = policy_preprocessor.transform(policy_df)
    
    # Fetch and process interaction data for fitting
    full_interaction_features = pd.read_sql("SELECT * FROM public.interactions", engine)[config['interaction_numeric_cols']]
    interaction_preprocessor = StandardScaler()
    interaction_preprocessor.fit(full_interaction_features)
    interaction_features = interaction_preprocessor.transform(interaction_df[config['interaction_numeric_cols']])
    
    return customer_features, policy_features, interaction_features
