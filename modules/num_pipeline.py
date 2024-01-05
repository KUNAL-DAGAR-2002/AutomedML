from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from modules.cleaner import clean
from modules.logger import logging
from modules.exception import CustomException
import sys
import pickle
from sklearn.model_selection import train_test_split

def numPipeline():
    try:
        logging.info("In num_pipleine")
        numPipeline = Pipeline(
            steps=[
                ("Impute",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler()),
            ]
        )
        logging.info("Num Pipeline created success")
        return numPipeline
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)


def numProcessor(df,numPipeline):
    try:
        logging.info("In num_pipeline.py")
        logging.info("Creating numerical preprocessor")
        temp = df.drop(df.columns[-1],axis=1)
        numerical = list(temp.select_dtypes(exclude="object"))
        numPreProcessor = ColumnTransformer([
            ("numPipeline",numPipeline,numerical)
        ])
        pickle.dump(numPreProcessor,open("static/artifacts/num_preprocessor.pkl","wb"))
        X,y = df.drop(df.columns[-1],axis=1),df[df.columns[-1]]
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=42)
        X_train = numPreProcessor.fit_transform(X_train)
        X_test = numPreProcessor.transform(X_test)
        logging.info("Task done Success")
        return numPreProcessor,X_train,X_test,y_train,y_test
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)


if __name__ == "__main__":
    df = clean("train_new.csv")
    numPipeline = numPipeline()
    preprocessor = numProcessor(df,numPipeline)

