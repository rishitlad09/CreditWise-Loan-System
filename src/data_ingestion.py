import logging
import os
from sklearn.model_selection import train_test_split
import pandas as pd

# logs directory

log_dir = "logs"

os.makedirs(log_dir,exist_ok=True)

#logging configuration

logger = logging.getLogger(name='data_ingestion')
logger.setLevel('DEBUG')

console_handler = logging.StreamHandler()
console_handler.setLevel('DEBUG')


log_file_path = os.path.join(log_dir,"data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel('DEBUG')

formatter = logging.Formatter('%(asctime)s | %(name)s | %(levelname)s | %(message)s')

file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

def load_data(data_url:str)->pd.DataFrame:
    """Load Data from a CSV File."""
    try:
        df = pd.read_csv(data_url)
        logger.debug('Data Loaded from %s',data_url)
        logger.info("Dataset shape: %s", df.shape)
        return df
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV FILE : %s',e)
        raise
    except Exception as e:
        logger.error('Unexcpected Error Occurred while loading the data : %s',e)
        raise
    

def clean_data(df:pd.DataFrame)->pd.DataFrame:
    """Preprocesss the data"""
    
    try:
        df = df.drop(columns="Applicant_ID",axis=1)
        logger.info("Dropped columns: %s", ["Applicant_ID"])
        return df
    
    except pd.errors.ParserError as e:
        logger.error('Failed to parse the CSV File : %s',e)
        raise
    except Exception as e:
        logger.error('Unexpected Error occured : %s',e)
        raise
    

def save_data(train_data:pd.DataFrame,test_data:pd.DataFrame,data_path:str)->None:
    """Save the train and test Dataset."""
    
    try:
        raw_data_path = os.path.join(data_path,'raw')
        
        os.makedirs(raw_data_path,exist_ok=True)
        train_data.to_csv(os.path.join(raw_data_path,'train.csv'),index=False)
        test_data.to_csv(os.path.join(raw_data_path,'test.csv'),index=False)    
        logger.debug('Train and test data saved to : %s',raw_data_path)
        
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data %s',e)
        raise
    
    
def main():
    try:
        test_size = 0.20
        data_url = "https://raw.githubusercontent.com/rishitlad09/Datasets/refs/heads/main/loan_approval_data.csv"
        
        df = load_data(data_url=data_url)
        final_df = clean_data(df)
        train_data,test_data = train_test_split(final_df,test_size=test_size,random_state=42)
        logger.info("Train shape: %s", train_data.shape)
        logger.info("Test shape: %s", test_data.shape)
        save_data(train_data=train_data,test_data=test_data,data_path="./data")
    except Exception as e:
        logger.error('Failed to complete the data ingestion process : %s',e)
        print(f"Error : {e}")
        
        
if __name__ == "__main__":
    main()

        
    











