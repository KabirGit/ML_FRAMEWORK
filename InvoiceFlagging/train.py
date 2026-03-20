from InvoiceFlagging.data_preprocessing import load_data,apply_labels,split_data,scale_features
from InvoiceFlagging.model_eval import train_random_forest,evaluate_model,train_logistic_regression
import joblib
from config import DATA_PATH, MODEL_DIR
features=['invoice_quantity','invoice_dollars','total_quantity','total_dollars','average_receiving_delay']
target='flag_invoice'

def main():
    df=load_data()
    
    df=apply_labels(df)
    xtrain,xtest,ytrain,ytest=split_data(df,features,target)
    xtrain_scaled,xtest_scaled=scale_features(xtrain,xtest,MODEL_DIR / "scaler.pkl")
    model=train_logistic_regression(xtrain_scaled,ytrain)
    evaluate_model(model, xtest_scaled, ytest,'logisticregression')
    joblib.dump(model, MODEL_DIR / "predict_flag_invoice.pkl")

if __name__=="__main__":
    main()
