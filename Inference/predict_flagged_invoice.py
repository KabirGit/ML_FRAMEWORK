import joblib
import pandas as pd
import os
from config import MODEL_DIR

# --------------------------------------------------
# 1. Load Model
# --------------------------------------------------
def load_model(model_path):
    
    model = joblib.load(model_path)
    return model


# --------------------------------------------------
# 2. Load Scaler
# --------------------------------------------------
def load_scaler(scaler_path):
    
    scaler = joblib.load(scaler_path)
    return scaler


# --------------------------------------------------
# 3. Prediction Function
# --------------------------------------------------
def predict_invoice_flag(input_data: dict):
    """
    input_data: dictionary with required features
    returns: DataFrame with predictions
    """

    # Convert dict to DataFrame
    df = pd.DataFrame(input_data)

    # Ensure correct feature order (VERY IMPORTANT)
    expected_features = [
        'invoice_quantity',
        'invoice_dollars',
        'total_quantity',
        'total_dollars',
        'average_receiving_delay'
    ]

    df = df[expected_features]

    # Load scaler & model
    scaler = load_scaler(MODEL_DIR / "scaler.pkl")
    model = load_model(MODEL_DIR / "predict_flag_invoice.pkl")

    # Scale data
    df_scaled = scaler.transform(df)

    # Predict
    predictions = model.predict(df_scaled)

    # Output DataFrame
    result_df = df.copy()
    result_df["Predicted_Flag"] = predictions

    return result_df


# --------------------------------------------------
# 4. Main Function (Demo)
# --------------------------------------------------
def main():


    model_path = MODEL_DIR/"predict_flag_invoice.pkl"
    scaler_path = MODEL_DIR / "scaler.pkl"

    # Demo input (3 samples)
    input_data = {
        "invoice_quantity": [10, 5, 2],
        "invoice_dollars": [2000, 300, 50],
        "total_quantity": [12, 6, 3],
        "total_dollars": [2100, 320, 55],
        "average_receiving_delay": [5, 12, 3]
    }

    result = predict_invoice_flag(input_data)

    print("\nInvoice Flag Predictions:\n")
    print(result)


# --------------------------------------------------
if __name__ == "__main__":
    main()
    