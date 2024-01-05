from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from modules.exception import CustomException
from modules.logger import logging

def regressor(X_train,X_test,y_train,y_test):
    try:
        logging.info("In model_selection.py")
        models = {
            "LinearRegression":LinearRegression(),
            "DecisionTree":DecisionTreeRegressor(),
            "RandomForest":RandomForestRegressor(),
            "SVM":SVR(),
            "Lasso":Lasso(),
            "ElasticNet":ElasticNet()
        }

        result = []
        result_name = []
        for i in range(len(list(models.keys()))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            result.append(r2_score(y_test,y_pred))
            result_name.append(list(models.values())[i])
        logging.info("Model selected Proceeing for model creation")
        logging.info(f"{result_name[result.index(max(result))]} is selected as the model")
        return result_name[result.index(max(result))]
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)


def classifier(X_train,X_test,y_train,y_test):
    try:
        logging.info("In model_selection.py")
        models = {
            "RandomForest":RandomForestClassifier(),
            "DecisionTree":DecisionTreeClassifier(),
            "SVC":SVC(),
            "LogisticRegression":LogisticRegression()
        }
        result = []
        result_name = []
        for i in range(len(list(models.keys()))):
            model_name = list(models.keys())[i]
            model = list(models.values())[i]
            model.fit(X_train,y_train)
            y_pred = model.predict(X_test)
            result.append(accuracy_score(y_test,y_pred))
            result_name.append(list(models.values())[i])
        logging.info("Model selected proceeding for model creating")
        logging.info(f"{result_name[result.index(max(result))]} is selected as the model")
        return result_name[result.index(max(result))]

    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)