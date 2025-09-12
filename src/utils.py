import numpy as np
import pandas as pd
import os
import sys
from src.exception import CustomException
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV



def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e,sys)
    
def evaluate_models(X_train,y_train,X_test,y_test,models,params):

    try:
        reports={}

        for model_name, model in models.items():
            para=params[model_name]

            random_search=RandomizedSearchCV(
                estimator=model,
                param_distributions=para,
                cv=3
            )

            random_search.fit(X_train,y_train)
            model.set_params(**random_search.best_params_)
            model.fit(X_train,y_train)

            y_train_predict=model.predict(X_train)
            y_predicted=model.predict(X_test)
            train_accuracy=r2_score(y_train,y_train_predict)
            test_accuracy=r2_score(y_test,y_predicted)

            reports[model_name]=test_accuracy

        return reports 
    
    except Exception as e:
        raise CustomException(e,sys)
    
def load_object(file_path):

    try:
        with open(file_path,'rb') as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e,sys)    