from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from modules.cleaner import clean
from modules.logger import logging
from modules.exception import CustomException
from sklearn.model_selection import train_test_split
import sys
import pickle

def catPipeline():
    try:
        logging.info("In cat_pipleine")
        catPipeline = Pipeline(
            steps=[
                ("impute",SimpleImputer(strategy="most_frequent")),
                ("onehot",OneHotEncoder())
            ]
        )
        logging.info("cat Pipeline created success")
        return catPipeline
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)


def catProcessor(df,catPipeline):
    try:
        logging.info("In cat_pipeline.py")
        logging.info("Creating categorical preprocessor")
        temp = df.drop(df.columns[-1],axis=1)
        categorical = list(temp.select_dtypes(include="object"))
        catPreProcessor = ColumnTransformer([
            ("catPipeline",catPipeline,categorical)
        ])
        pickle.dump(catPreProcessor,open("artifacts/cat_preprocessor.pkl","wb"))
        X,y = df.drop(df.columns[-1],axis=1),df[df.columns[-1]]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        X_train = catPreProcessor.fit_transform(X_train)
        X_test = catPreProcessor.transform(X_test)
        logging.info("Task done Success")
        return catPreProcessor,X_train,X_test,y_train,y_test
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)


if __name__ == "__main__":
    df = clean("mushrooms.csv")
    catPipeline = catPipeline()
    preprocessor = catProcessor(df,catPipeline)

