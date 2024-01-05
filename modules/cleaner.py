import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from modules.logger import logging
from modules.exception import CustomException
import sys
def clean(data):
    try:
        ''' This function checks for null values and do some basic preprocessing'''
        logging.info("In logger.py")
        logging.info("Logging is started")
        df = pd.read_csv(data)
        
        df.isna().sum()
        numerical = df.select_dtypes(exclude='object')
        categorical = df.select_dtypes(include='object')
        
        logging.info("Doing Preprocessing")
        for i in categorical.columns:
            df[i] = df[i].str.replace(" ","")
            df[i] = df[i].str.replace("?","")
            df[i] = df[i].str.replace("-","")
            df[i] = df[i].str.lower()
        df.columns = df.columns.str.lower()

        if(df[df.columns[-1]].dtype=="object"):
            encoder = LabelEncoder()
            df[df.columns[-1]] = encoder.fit_transform(df[[df.columns[-1]]])
        logging.info("preprocessing done data saved in raw.csv")
        os.makedirs("static/artifacts",exist_ok=True)
        df.to_csv("static/artifacts/raw.csv",index=False)
        logging.info("Success")
        logging.info(df.head())
        return df
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)

if __name__ == "__main__":
    df = clean("train_new.csv")
