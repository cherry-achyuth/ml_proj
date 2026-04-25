import os
import sys

import numpy as np 
import pandas as pd
import pickle
from src.exceptions import CustomException
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

def saving_obj(file_path,obj):
    try:
        dir_name = os.path.dirname(file_path)
        os.makedirs(dir_name,exist_ok=True)

        with open(file_path,"wb") as file_obj:
            pickle.dump(obj,file_obj)
    except Exception as e:
        raise CustomException(e,sys)

def evaluate_models(x_train,x_test,y_train,y_test,models,params):
    try:
        report = {}
        for i in range(len(list(models))):

           
            model = list(models.values())[i]
            para = params[list(models.keys())[i]]

            gsc = GridSearchCV(model,para,cv=3)
            gsc.fit(x_train,y_train)

            model.set_params(**gsc.best_params_)
            model.fit(x_train,y_train)
            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] = test_model_score
        return report
    except Exception as e:
        raise CustomException(e)