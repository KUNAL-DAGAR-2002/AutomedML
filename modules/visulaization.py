import matplotlib.pyplot as plt
import seaborn as sns
from modules.cleaner import clean
from modules.exception import CustomException
from modules.logger import logging
import sys
import os
import pandas as pd
import warnings
import shutil
def visualization(df):
    try:
        warnings.filterwarnings("ignore")
        ''' This is for generating visualizations '''
        
        if os.path.exists("static/artifacts/visualization"):
            shutil.rmtree("static/artifacts/visualization")
        logging.info("In Visualization.py fetched the data")
        os.makedirs("static/artifacts/visualization")
        checklist = list(df.dtypes)
        num = 0
        cat = 0
        for i in checklist:
            if(i=='int64'):
                num = num + 1
            if(i=='object'):
                cat = cat + 1
        if(num>0):
            numerical = df.select_dtypes(exclude="object")
        if(cat>0):
            categorical = df.select_dtypes(include="object")
        #Plotting heatmap
        logging.info("Plotting Visualizations")
        if(num>0):
            sns.heatmap(numerical.corr(),annot=True)
            plt.savefig("static/artifacts/visualization/heatmap.jpg")
            plt.close()
        #Categorical Plot
        if(cat>0):
            for i in categorical.columns:
                plt.figure(figsize=(15,5))
                sns.countplot(x=i, data=categorical, hue=df[df.columns[-1]]).set_title('Count of '+str(i))
                plt.savefig(f"static/artifacts/visualization/{i}.jpg")
                plt.close()
        if(num>0):
            for i in numerical.columns:
                plt.figure()
                sns.histplot(x=i,data=numerical,hue=df[df.columns[-1]],kde=True,bins=30)
                plt.savefig(f"static/artifacts/visualization/{i}.jpg")
                plt.close()
            for i in numerical.columns:
                plt.figure()
                sns.barplot(x=i,data=numerical,y=df[df.columns[-1]])
                plt.savefig(f"static/artifacts/visualization/{i}_barplot.jpg")
                plt.close()
        #Pair plot
        sns.pairplot(df)
        plt.savefig("static/artifacts/visualization/pairplot.jpg")
        plt.close()
        logging.info("Task completed")
        logging.info("Visualizations stored in visualization")
    except Exception as e:
        logging.info(CustomException(e,sys))
        raise CustomException(e,sys)

if __name__ == "__main__":
    df = clean("mushrooms.csv")
    visualization(df)
