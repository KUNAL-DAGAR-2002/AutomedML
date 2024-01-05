from modules.num_pipeline import numPipeline
from modules.num_pipeline import numProcessor
from modules.cleaner import clean
from modules.logger import logging
from modules.exception import CustomException
from modules.model_selection import regressor
from modules.model_selection import classifier
from sklearn.metrics import r2_score,accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import os
import sys

def regression(regressor,X_train,X_test,y_train,y_test):
    try:
        logging.info("Making Model")
        regressor.fit(X_train,y_train)
        y_pred = regressor.predict(X_test)
        score = r2_score(y_test,y_pred)
        logging.info("Task completed successfully")
        pickle.dump(regressor,open("static/artifacts/model.pkl","wb"))
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)
    

def classification(classifier,X_train,X_test,y_train,y_test):
    try:
        logging.info("Making Model")
        classifier.fit(X_train,y_train)
        y_pred = classifier.predict(X_test)
        score = accuracy_score(y_test,y_pred)
        logging.info("Task completed successfully")
        pickle.dump(classifier,open("static/artifacts/model.pkl","wb"))
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)

if __name__ == "__main__":
    df = clean("housing1.csv")
    numPipeline = numPipeline()
    preprocessor,X_train,X_test,y_train,y_test = numProcessor(df,numPipeline)
    model = regressor(X_train,X_test,y_train,y_test)
    regression(model,X_train,X_test,y_train,y_test)