from modules.cat_pipeline import catPipeline
from modules.cat_pipeline import catProcessor
from modules.num_pipeline import numPipeline
from modules.num_pipeline import numProcessor
from modules.combined_train import combinedPreProcessor
from modules.cleaner import clean
from modules.visulaization import visualization
from modules.model_selection import regressor
from modules.model_selection import classifier
from modules import train_num
from modules import train_cat
from modules import combined_train
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pickle
from modules.exception import CustomException
from modules.logger import logging
import sys
import os

def combined_classification(data):
    df = clean(data)
    visualization(df)
    num_pipeline = numPipeline()
    cat_pipeline = catPipeline()
    preprocessor,X_train,X_test,y_train,y_test = combinedPreProcessor(df,num_pipeline, cat_pipeline)
    model = classifier(X_train,X_test,y_train,y_test)
    combined_train.classification(model,X_train,X_test,y_train,y_test)

def combined_regressor(data):
    df = clean(data)
    visualization(df)
    num_pipeline = numPipeline()
    cat_pipeline = catPipeline()
    preprocessor,X_train,X_test,y_train,y_test = combinedPreProcessor(df,num_pipeline, cat_pipeline)
    model = classifier(X_train,X_test,y_train,y_test)
    combined_train.regression(model,X_train,X_test,y_train,y_test)

def cat_classification(data):
    df = clean(data)
    visualization(df)
    cat_Pipeline = catPipeline()
    preprocessor,X_train,X_test,y_train,y_test = catProcessor(df,cat_Pipeline)
    model = classifier(X_train,X_test,y_train,y_test)
    train_cat.classification(model,X_train,X_test,y_train,y_test)

def num_regressor(data):
    df = clean(data)
    visualization(df)
    num_Pipeline = numPipeline()
    preprocessor,X_train,X_test,y_train,y_test = numProcessor(df,num_Pipeline)
    model = regressor(X_train,X_test,y_train,y_test)
    train_num.regression(model,X_train,X_test,y_train,y_test)

def num_classification(data):
    df = clean(data)
    visualization(df)
    num_Pipeline = numPipeline()
    preprocessor,X_train,X_test,y_train,y_test = numProcessor(df,num_Pipeline)
    model = classifier(X_train,X_test,y_train,y_test)
    train_num.classification(model,X_train,X_test,y_train,y_test)

if __name__ == "__main__":
    num_regressor("housing1.csv")
    
