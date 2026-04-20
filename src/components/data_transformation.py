import pandas as pd
import numpy as np
import sys
import os

from src.exceptions import CustomException
from src.logger import logging

from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from src.utils import saving_obj

@dataclass
class DataTransformerConfig:
    preprocessor_obj_filepath = os.path.join('artifacts','preprocessor.pkl')

class DataTransformer:
    def __init__(self):
        self.data_transformation_obj = DataTransformerConfig()

    def get_data_transformation_obj(self):
        try:
            num_columns = ["writing_score", "reading_score"]
            cat_columns = ["gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",]
            
            num_pipeline = Pipeline(steps = [
                ("imputer",SimpleImputer(strategy='median')),
                ("scaler",StandardScaler())
            ])

            cat_pipeline = Pipeline(steps = [
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ])
            logging.info("divided cat_columns and num_columns ")

            preprocessor = ColumnTransformer([
                ("num_pipeline",num_pipeline,num_columns),
                ("cat_pipeline",cat_pipeline,cat_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)

    def initiate_transformation(self,train_path,test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("reading train and test data")
            logging.info("obtaining preprocessor obj")
            
            preproceessor_obj = self.get_data_transformation_obj()

            train_input_feature = train_data.drop(columns = ['math_score'],axis=1)
            train_target_feature = train_data['math_score']

            test_input_feature = test_data.drop(columns = ['math_score'],axis=1)
            test_target_feature = test_data['math_score']
            logging.info("performing preprocessing")

            train_input_arr = preproceessor_obj.fit_transform(train_input_feature)
            test_input_arr = preproceessor_obj.transform(test_input_feature)

            train_arr = np.c_[
                train_input_arr,np.array(train_target_feature)
            ]
            test_arr = np.c_[
                test_input_arr,np.array(test_target_feature)
            ]

            logging.info("saving preprocessing object")

            saving_obj(
                file_path=self.data_transformation_obj.preprocessor_obj_filepath,
                obj = preproceessor_obj
            )

            return(
                train_arr,
                test_arr,
                self.data_transformation_obj.preprocessor_obj_filepath
            )
        except Exception as e:
            raise CustomException(e,sys)