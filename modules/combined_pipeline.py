from modules.cat_pipeline import catPipeline
from modules.num_pipeline import numPipeline
from modules.exception import CustomException
from sklearn.compose import ColumnTransformer
from modules.logger import logging
from modules.cleaner import clean
from sklearn.model_selection import train_test_split
import time
import sys
import pickle

def combinedPreProcessor(df,numPipeline,catPipeline):
    try:
        logging.info("In combined_pipeline.py")
        logging.info("Creating Pipeline")
        logging.info("pipeline creation started")
        temp = df.drop(df.columns[-1],axis=1)
        numerical = list(temp.select_dtypes(exclude="object"))
        categorical = list(temp.select_dtypes(include="object"))
        combinedPreProcessor = ColumnTransformer([
            ("num_piepline",numPipeline,numerical),
            ("cat_pipeline",catPipeline,categorical)
        ])
        X,y = df.drop(df.columns[-1],axis=1),df[df.columns[-1]]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        X_train = combinedPreProcessor.fit_transform(X_train)
        X_test = combinedPreProcessor.transform(X_test)
        logging.info("Task Success")
        pickle.dump(combinedPreProcessor,open("static/artifacts/combined_preprocessor.pkl","wb"))
        return combinedPreProcessor,X_train,X_test,y_train,y_test
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)

if __name__ == "__main__":
    df = clean("mushrooms.csv")
    numPipeline = numPipeline()
    catPipeline = catPipeline()
    preprocessor = combinedPreProcessor(df,numPipeline,catPipeline)
