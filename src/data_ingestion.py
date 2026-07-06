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

if not logger.handlers:
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
        logger.error('Unexpected Error Occurred while loading the data : %s',e)
        raise
    


    

def save_data(df:pd.DataFrame,data_path:str)->None:
    """Save the Raw Dataset."""
    
    try:
        raw_data_path = os.path.join(data_path,'raw')
        
        os.makedirs(raw_data_path,exist_ok=True)
        df.to_csv(os.path.join(raw_data_path,'raw.csv'),index=False)    
        logger.debug('Raw data saved to : %s',raw_data_path)
        
    except Exception as e:
        logger.error('Unexpected error occurred while saving the data %s',e)
        raise
    
    
def main():
    try:
        data_url = "https://raw.githubusercontent.com/rishitlad09/Datasets/refs/heads/main/loan_approval_data.csv"
        
        df = load_data(data_url=data_url)
        
        
        save_data(df=df,data_path="./data")
    except Exception as e:
        logger.error('Failed to complete the data ingestion process : %s',e)
        print(f"Error : {e}")
        
        
if __name__ == "__main__":
    main()

        
    











