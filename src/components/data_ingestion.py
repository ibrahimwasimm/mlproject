import os
import sys
import pandas as pd
from src.exception import CustomException
from src.logger import logger
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DatatransformationConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

@dataclass
class DataingestionConfig:
    train_path: str=os.path.join('artifacts',"train.csv")
    test_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"data.csv")

class dataingestion:
    def __init__(self):
        self.ingestionConfig=DataingestionConfig()

    def initiate_data_ingestion(self):
        logger.info('Entered the data metbhod or componenet :')

        try:
            df=pd.read_csv('notebook/data/stud.csv')
            logger.info("Reading the data")

            os.makedirs(os.path.dirname(self.ingestionConfig.train_path),exist_ok=True)
            df.to_csv(self.ingestionConfig.raw_data_path,index=False,header=True)

            logger.info('training and testing initiated')

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestionConfig.train_path,index=False,header=True)
            test_set.to_csv(self.ingestionConfig.test_path,index=False,header=True)

            logger.info("ingestion of training and testing completed")
            return(
                self.ingestionConfig.train_path,
                self.ingestionConfig.test_path
            )
        except Exception as e:
            raise CustomException(e,sys)
        
if __name__=="__main__":
    obj=dataingestion()
    train_data,test_data=obj.initiate_data_ingestion()

    data_transformation=DataTransformation()
    train_arr,test_arr,_=data_transformation.initiate_data_transformation(train_data,test_data)

    model_trainer=ModelTrainer()
    print(model_trainer.initiate_model_trainer(train_arr,test_arr))


        
    
    
