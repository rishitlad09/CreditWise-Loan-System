import os
import logging
import pandas as pd
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import numpy as np
import yaml
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger(name='model_training')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,'model_training.log')
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

def load_params(path:str)->dict:
    try:
        with open(path,'r') as file:
            params = yaml.safe_load(file)
        logger.debug("parameters loaded successfully")
        return params
    except FileNotFoundError:
        logger.error('File not found : %s',path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error : %s',e)
        raise
    except Exception as e:
        logger.error('Error while loading parameters from path : %s',path)

def load_data(path:str)->pd.DataFrame:
    """Load the training dataset from disk."""
    try:
        train_data = pd.read_csv(os.path.join(path,'train_final.csv'))
        logger.debug('Data loaded successfully from %s',path)
        return train_data
    except Exception as e:
        logger.error('Unexpected error occurred while loading data : %s',e)
        raise
    
    
def encode(y_train:pd.Series)->np.ndarray:
    """Encode the training target labels"""
    try:
        le = LabelEncoder()
    
        y_train_en = le.fit_transform(y_train)
        return y_train_en

    except Exception as e:
        logger.error('Unexpected error occurred while encoding : %s',e)
        raise
    

    
def train_model(X_train:pd.DataFrame,y_train:pd.DataFrame,params:dict)-> XGBClassifier:
    """Training the model using XGBClassifier """
    try:
        logger.info('Model Training started...')
        xgb_cls = XGBClassifier(max_depth = params['max_depth'],n_estimators=params['n_estimators'],random_state=params['random_state'])
        xgb_cls.fit(X_train,y_train)
        logger.info('Model Training finished....')
        return xgb_cls
    
    except Exception as e:
        logger.error('Unexpected error while model training : %s',e)
        raise
    
def save_model(model:XGBClassifier,file_path:str)->None:
    try:
        os.makedirs(os.path.dirname(file_path),exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(model,file)
        logger.debug('Model saved successfully... %s',file_path)
    
    
    except FileNotFoundError as e:
        logger.error('File path not found : %s',e)
        raise
    
    except Exception as e:
        logger.error('Error while saving model : %s',e)
        raise 
    
        
    

def main():
    try:
        path = './data/final/'
        train_data = load_data(path=path)
        X_train = train_data.drop(columns='Loan_Approved',axis=1)
        y_train = train_data['Loan_Approved']
        y_train_en = encode(y_train=y_train)
        param_path = 'params.yaml'
        params = load_params(path=param_path)['model_training']
        model = train_model(X_train,y_train_en,params)
        file_path = 'models/model.pkl'
        save_model(model=model,file_path=file_path)
    except Exception as e:
        logger.error('Unexpected error occured : %s',e)
        print(f'Error {e}')
    
    
    
    
if __name__ == '__main__':
    main()
    