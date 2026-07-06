import logging
import os
import pandas as pd
from sklearn.impute import SimpleImputer
log_dir = "logs"
os.makedirs(log_dir,exist_ok=True)

logger = logging.getLogger(name='data_preprocessing')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')

log_file_path = os.path.join(log_dir,"data_preprocessing.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s ')


console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def clean_data(df:pd.DataFrame)->pd.DataFrame:
    """Clean the dataset by removing unnecessary columns and imputing missing values."""
    
    try:
        df = df.drop(columns="Applicant_ID",axis=1)
        logger.info("Dropped columns: %s", ["Applicant_ID"])
        categorical_cols=df.select_dtypes(include=["object"]).columns
        numerical_cols=df.select_dtypes(include=["number"]).columns
        
        num_imp=SimpleImputer(strategy='mean')
        logger.debug("Handling numerical columns with 'MEAN'")
        df[numerical_cols]=num_imp.fit_transform(df[numerical_cols])
        cat_imp=SimpleImputer(strategy='most_frequent')
        logger.debug("Handling categorical columns with 'MOST FREQUENT'")
        df[categorical_cols]=cat_imp.fit_transform(df[categorical_cols])
        logger.debug("Handled null values successfully.")
        return df
    
    except Exception as e:
        logger.error('Unexpected Error occured : %s',e)
        raise
    
    
def save_data(df:pd.DataFrame,path:str)->None:
    try:
        os.makedirs(path, exist_ok=True)
        df.to_csv(os.path.join(path,"processed.csv"),index=False)
        logger.debug('Processed Data saved to : %s',path)
    except Exception as e:
        logger.error("Unexpected Error occured while saving the file : %s",e)
        raise
        
        
def main():
    try:
        df = pd.read_csv("./data/raw/raw.csv")
        df = clean_data(df)
        path = "./data/processed"
        save_data(df,path=path)
    except Exception as e:
        logger.error('Failed to complete Data Preprocessing : %s',e)
        print(f"Error : {e}")
        
        
if __name__ == '__main__':
    main()
        