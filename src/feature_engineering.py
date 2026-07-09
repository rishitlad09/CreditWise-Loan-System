import logging
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
import yaml
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger(name='feature_engineering')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,"feature_engineering.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s ')


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
        
        
def encode_data(train_data:pd.DataFrame,test_data:pd.DataFrame)->pd.DataFrame:
    try:

        le=LabelEncoder()
        
        train_data["Education_Level"] = le.fit_transform(
            train_data["Education_Level"]
        )

        test_data["Education_Level"] = le.transform(
            test_data["Education_Level"]
        )
        
        
        logger.debug('Education_Level encoded Successfully using LabelEncoder')
        
        cols=["Employment_Status","Marital_Status","Loan_Purpose","Property_Area","Gender","Employer_Category"]
        ohe=OneHotEncoder(drop="first",sparse_output=False,handle_unknown="ignore")

        
        train_encoded = ohe.fit_transform(train_data[cols])
        
        test_encoded = ohe.transform(test_data[cols])

        train_encoded_df = pd.DataFrame(
            train_encoded,
            columns=ohe.get_feature_names_out(cols),
            index=train_data.index
        )
        
        test_encoded_df = pd.DataFrame(
            test_encoded,
            columns=ohe.get_feature_names_out(cols),
            index=test_data.index
        )

        train_data = pd.concat([
                train_data.drop(columns=cols),
                train_encoded_df
            ],
            axis=1
        )   
        
        
        test_data = pd.concat([
                test_data.drop(columns=cols),
                test_encoded_df
            ],
            axis=1
        )    
        return train_data,test_data
            
    except Exception as e:
        logger.error('Unexpected Error while Encoding the Data : %s',e)
        raise

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,path:str)->None:
    try:
        os.makedirs(path,exist_ok=True)
        
        train_data.to_csv(os.path.join(path,"train_final.csv"),index=False)
        test_data.to_csv(os.path.join(path,"test_final.csv"),index=False)  
        logger.debug('Saved train and test Data to %s',path)
    except Exception as e:
        logger.error('Unexpected error occured while saving : %s',e)
        raise
def main():
    param_path = 'params.yaml'
    test_size = load_params(path=param_path)['feature_engineering']['test_size']
    df = pd.read_csv('./data/processed/processed.csv')
    
    train_data,test_data = train_test_split(df,test_size=test_size,random_state=42)
    train_final,test_final = encode_data(train_data=train_data,test_data=test_data)
    path = './data/final/' 
    save_data(train_data=train_final,test_data=test_final,path=path)
    
    
if __name__ =='__main__':
    main()