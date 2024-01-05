from modules.cat_pipeline import catPipeline
from modules.cat_pipeline import catProcessor
from modules.cleaner import clean
from modules.logger import logging
from modules.exception import CustomException
from modules.model_selection import classifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle
import os
import sys

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
    df = clean("mushroom1.csv")
    catPipeline = catPipeline()
    preprocessor,X_train,X_test,y_train,y_test = catProcessor(df,catPipeline)
    model = classifier(X_train,X_test,y_train,y_test)
    classification(model,X_train,X_test,y_train,y_test)