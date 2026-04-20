import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split

from src.exceptions import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformer
from src.components.data_transformation import DataTransformerConfig

from dataclasses import dataclass

@dataclass
class DataIngestionConfig:
    train_data_path : str = os.path.join('artifacts','train.csv')
    test_data_path : str = os.path.join('artifacts','test.csv')
    raw_data_path : str = os.path.join('artifacts','data.csv')

class DataIngestion:

    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_ingestion(self):
        logging.info('started initiating data')
        try:
            df = pd.read_csv('notebook/data/stud.csv')
            logging.info('read the dataset as df')
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            
            train_dataset,test_dataset = train_test_split(df,test_size=0.25,random_state=42)
            logging.info('data is splitted into train and test ')

            train_dataset.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_dataset.to_csv(self.ingestion_config.test_data_path,index = False,header=True)

            logging.info("ingestion data is completed")

            return(self.ingestion_config.train_data_path,
            self.ingestion_config.test_data_path)
        except Exception as err:
            raise CustomException(err,sys)

if __name__ == '__main__':
    obj = DataIngestion()
    train_datapath,test_datapath = obj.initiate_ingestion()

    transformer_obj = DataTransformer()
    transformer_obj.initiate_transformation(train_datapath,test_datapath)