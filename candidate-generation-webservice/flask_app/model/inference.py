import pandas as pd
from tensorflow.keras.models import load_model
from config import MODEL_PATH, PREPROCESS_CONFIG
from sqlalchemy import create_engine
from config import DB_URI
from model.preprocess import preprocess_data


# Create a database connection
engine = create_engine(DB_URI)

def recommend_policies(customer_id, top_n=5):
    # Fetch customer data from the database
    customer_data = pd.read_sql(f"SELECT * FROM public.customers WHERE customer_id = {int(customer_id)}", engine)
    if customer_data.empty:
        return {'error': 'Customer not found'}

    # Fetch interactions data for the customer
    interactions = pd.read_sql("SELECT * FROM public.interactions", engine)
    customer_interactions = interactions[interactions['customer_id'] == int(customer_id)]
    if not customer_interactions.empty:
        interaction_data = customer_interactions.iloc[-1:]
    else:
        interaction_data = pd.DataFrame([{'clicked': 0, 'viewed_duration': 0, 'comparison_count': 0, 'abandoned_cart': 0}])

    # Fetch all policy data
    policies = pd.read_sql("SELECT * FROM public.policies", engine)

    # Use all policies as candidates
    candidate_policies = policies.copy()
    num_candidates = candidate_policies.shape[0]

    # Replicate customer and interaction data to match candidate policies count
    customer_features_df = pd.concat([customer_data] * num_candidates, ignore_index=True)
    interaction_features_df = pd.concat([interaction_data] * num_candidates, ignore_index=True)

    # Preprocess data using freshly fitted transformers
    customer_features, policy_features, interaction_features = preprocess_data(
        customer_features_df, candidate_policies, interaction_features_df, PREPROCESS_CONFIG
    )

    # Load the pre-trained model
    model = load_model(MODEL_PATH)

    # Get predictions. Model expects a list: [customer_features, interaction_features, policy_features]
    predictions = model.predict([customer_features, interaction_features, policy_features])

    # Add predictions to candidate policies and return top recommendations
    candidate_policies['score'] = predictions
    recommended = candidate_policies.sort_values(by='score', ascending=False).head(top_n)
    result = recommended.to_dict(orient='records')
    
    return result
