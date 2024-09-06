import sys
import os
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer 
from src.exception  import CustomException
from src.logger import logging
from src.utils import save_object
from dataclasses import dataclass

@dataclass
class DataTransformationConfig:
    preprocessing_obj_file_path = os.path.join('artifacts','proprecessor.pkl')


class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformationConfig()
    
    def get_data_tranformer_object(self):
        
        """
         This function is responsible for data transformation
        """
        try:
            numrical_columns = ['writing_score','reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education',
                                   'lunch','test_preparation_course',]
            
            num_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')), # imputer, handling missing values
                    ("scalar",StandardScaler())
                    ]
            )
            car_pipeline =Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')), # imputer, handling missing values
                    ("one_hot_encoder",OneHotEncoder()),
                    ('standardsclar',StandardScaler())
                    ]

            )

            
            logging.info('numerical columns standard scaling completed')
            logging.info('categorical columns encoding completed')

            # combing the numerical and categorical pipeline togther using ColumnTransformer
            preprocessor = ColumnTransformer(
                [
                    ('numerical',num_pipeline,numrical_columns),
                    ('categorical',car_pipeline,categorical_columns)
                ]
            )

            return preprocessor

        except Exception as e:
            raise CustomException(e,sys)
        
    def initiate_data_transform(self,train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data is completed")
            
            logging.info("Obaining preprocessing objects")

            preprocessing_obj=self.get_data_tranformer_object()

            target_column_name = "math_score"
            numerical_columns = ["writing_score",'reading_score']

            #dropping the target column from train_df and test_df
            input_feature_train_df = train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info('Applying preprocessing object on training dataframe and test dataframe')
            input_feature_train_array = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_array = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[
                input_feature_train_array,np.array(input_feature_train_df)
            ]

            test_arr = np.c_[
                input_feature_test_array,np.array(input_feature_test_df)
            ]

            logging.info('saved the preprocessing object')
            
            # To save the pickle file
            save_object(file_path=self.data_tranformation_config.preprocessing_obj_file_path,
                        obj =preprocessing_obj            
            )
            return(
                train_arr,
                test_arr,
                self.data_tranformation_config.preprocessing_obj_file_path,

            )
        except Exception as e:
            raise CustomException(e,sys)