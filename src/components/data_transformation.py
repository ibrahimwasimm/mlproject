import os 
from src.exception import CustomException

from src.logger import logger

from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import sys
import pandas as pd
import numpy as np
from src.utils import save_object




class DatatransformationConfig:# this definig a path  only
    processor_obj_file_path=os.path.join('artifacts','processor.pkl')

class DataTransformation:# its actual containing the the code which path holds
    def __init__(self):
        self.data_transformation_config=DatatransformationConfig()

    def get_data_transformer_object(self):
        """ this function is for data transforamtion
        """
        try:
            numeric_features=['writing_score','reading_score']
            cat_features=[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder()),
                    ("scaler", StandardScaler(with_mean=False))
                ]
            )
            logger.info(f"Numerical features: {numeric_features}")
            logger.info(f"Categorical features: {cat_features}")

            preprocessor=ColumnTransformer([
                ("num_pipeline", num_pipeline, numeric_features),
                ("cat_pipeline", cat_pipeline, cat_features)
            ])

            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
    def intiate_datatransformation(self,train_path,test_path):

        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logger.info("reading training and testing the  data")
            logger.info("obtaing the preprocessing data")

            preprocessor_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            numerical_column=["reading_score","writing_score"]

            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]


            logger.info("applying preprocessing the trainning and testing daata")

            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr= np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logger.info("saved processing objects")
            
            save_object(
                file_path=self.data_transformation_config.processor_obj_file_path,
                obj=preprocessor_obj
            )
            return(
                train_arr,
                test_arr,
                self.data_transformation_config.processor_obj_file_path

            )
        except Exception as e:
            raise CustomException(e,sys)
        

        




            
        
