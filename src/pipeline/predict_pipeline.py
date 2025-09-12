import sys
from src.exception import CustomException
from src.logger import logger
import pandas as pd
from src.utils import load_object

class predictpipeline():
    def __init__(self):
        pass
    def predict(self,features):
        try:
            model_path="artifacts/model.pkl"
            processor_path="artifacts/processor.pkl"
            
            # Validate the input data
            categorical_columns = ['gender', 'race_ethnicity', 'parental_level_of_education', 
                                'lunch', 'test_preparation_course']
            numerical_columns = ['reading_score', 'writing_score']
            
            # Check for missing values
            missing_cols = features.columns[features.isnull().any()].tolist()
            if missing_cols:
                raise ValueError(f"Missing values found in columns: {missing_cols}")
            
            # Validate categorical values
            for col in categorical_columns:
                if col not in features.columns:
                    raise ValueError(f"Missing required column: {col}")
                if features[col].isnull().any() or (features[col] == '').any():
                    raise ValueError(f"Empty or null values found in column: {col}")
            
            # Load and apply preprocessing
            model=load_object(file_path=model_path)
            processor=load_object(file_path=processor_path)
            
            # Transform the features
            data_scaled=processor.transform(features)
            prediction=model.predict(data_scaled)

            return prediction


        except Exception as e:
            raise CustomException(e,sys)


class customdata:
    def __init__( self,
        gender:str,
        race_ethnicity:str,
        parental_level_of_education:str,
        lunch: str,
        test_preparation_course: str,
        reading_score: int,
        writing_score: int):

        self.gender=gender

        self.race_ethnicity = race_ethnicity

        self.parental_level_of_education = parental_level_of_education

        self.lunch = lunch

        self.test_preparation_course = test_preparation_course

        self.reading_score = reading_score

        self.writing_score = writing_score

    def get_data_as_frame(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.race_ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score]
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e,sys)
    
        
