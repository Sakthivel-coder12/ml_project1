import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from src.exception import CustomException
from src.logger import logging
from dataclasses import dataclass
from src.utils import save_object
@dataclass
class datatransformationconfig:
    preprocessor_filepath = os.path.join('artifacts',"preprocessor.pkl")


class datatransforamtion:
    def __init__(self):
        self.data_transformation_config = datatransformationconfig()
    def get_data_transformer_object(self):
        try:
            num_features= ['reading_score', 'writing_score']
            cat_features = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline = Pipeline(
                steps=[
                    ('simpleImputer',SimpleImputer(strategy="median")),
                    ('scaler',StandardScaler())
                ]
            )
            


            cat_pipeline = Pipeline(
                steps=[
                    ('simpleImputer',SimpleImputer(strategy="most_frequent")),
                    ('onehotencoder',OneHotEncoder()),
                    ('scaler',StandardScaler(with_mean=False))
                ]
            )   

            logging.info(f"Categorical columns: {cat_features}")
            logging.info(f"Numerical columns: {num_features}")

            preprocessor = ColumnTransformer(
                [
                    ('num_pipeline',num_pipeline,num_features),
                    ('cat_pipeline',cat_pipeline,cat_features)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    


    def initiate_transformaion(self,train_path,test_path):
            try:
                train_df = pd.read_csv(train_path)
                test_df = pd.read_csv(test_path)

                logging.info("Read train and test data completed")

                logging.info("Obtaining preprocessing object")

                preprocessing_obj = self.get_data_transformer_object()

                target_column_name = "math_score"
                numerical_columns = ["writing_score", "reading_score"]

                input_feature_train_df = train_df.drop(columns = [target_column_name],axis=1)
                target_feature_train_df = train_df[target_column_name]


                input_feature_test_df = test_df.drop(columns=[target_column_name],axis=1)
                target_feature_test_df = test_df[target_column_name]

                logging.info(
                f"Applying preprocessing object on training dataframe and testing dataframe."
                )

                input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
                input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

                train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
                test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]


                logging.info(f"Saved preprocessing objects....!")

                save_object(
                    file_path = self.data_transformation_config.preprocessor_filepath,
                    obj = preprocessing_obj
                )

                return(
                    train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_filepath
                )
            except Exception as e:
                 raise CustomException(e,sys)

