import os
import logging
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
from xgboost import XGBClassifier
import numpy as np
from sklearn.metrics import precision_score,accuracy_score,f1_score,recall_score
import json
from dvclive import Live
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger(name='model_evaluation')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_evaluation.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
def load_model(path:str)->XGBClassifier:
    
    """Load Model from the file"""
    try:
        with open(path,'rb') as file:
            model = pickle.load(file)
        logger.debug('Model Loaded successfully from %s',path)
        return model
    except Exception as e:
        logger.error('Unexpected error occurred : %s',e)
        raise
    

def load_data(file_path:str)-> pd.DataFrame:
    """Load Testing Data"""
    try:
        X_test = pd.read_csv(file_path)
        logger.debug('Test Data loaded..')
        return X_test
    
    except Exception as e:
        logger.error('Uexpected Error occured while loading file : %s',e)
        raise
    
def encode(y_test:pd.Series)->np.ndarray:
    """Encode the testing  target labels"""
    try:
        le = LabelEncoder()
    
        y_test_en = le.fit_transform(y_test)
        return y_test_en

    except Exception as e:
        logger.error('Unexpected error occurred while encoding : %s',e)
        raise

def evaluate_model(model:XGBClassifier,X_test:pd.DataFrame,y_test:np.ndarray)->dict:
    """Evaluate the model and return the evaluation metrics."""
    
    try:
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test,y_pred)
        precision=precision_score(y_test,y_pred)
        recall=recall_score(y_test,y_pred)
        f1score = f1_score(y_test,y_pred)
        
        metrics = {
            "accuracy":accuracy,
            "precision":precision,
            "recall":recall,
            "f1_score":f1score   
        }
        logger.debug('Model Evaluation metrics calculated.... ')
        return metrics
    except Exception as e:
        logger.error('Unexpected Error occurred while model evaluation : %s',e)
        raise
    
def save_metrics(metrics:dict,metric_path:str)->None:
    """Save the evaluation metrics to a JSON File."""
    
    try:
        os.makedirs(os.path.dirname(metric_path),exist_ok=True)
        
        with open(metric_path,'w') as file:
            json.dump(metrics,file,indent=4)
        logger.debug('Metrics saved to %s',metric_path)
    except Exception as e:
        logger.error('Error occurred while saving the evaluation metrics : %s',e )
        raise
    
def main():
    
    try:
        file_path = 'data/final/test_final.csv'
        test_data = load_data(file_path=file_path)
        path = './models/model.pkl'
        model = load_model(path=path)
        X_test = test_data.drop(columns='Loan_Approved')
        y_test=test_data['Loan_Approved']
        y_test_en = encode(y_test=y_test)
        
        results = evaluate_model(model=model,X_test=X_test,y_test=y_test_en)
        metric_path='reports/metrics.json'
        with Live(save_dvc_exp=True) as live:
            for metric_name,metric_value in results.items():
                live.log_metric(metric_name,metric_value)
                
        save_metrics(metrics=results,metric_path=metric_path)
        
    
    except Exception as e:
        logger.error('Failed to complete model evaluation : %s',e)
        print(f"Error : {e}")
        


if __name__ == '__main__':
    main()
    
