# Import necessary modules
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from lazypredict.Supervised import LazyRegressor


# CSV_FILE_PATH = "datasets/vegetable_price.csv"

def regressor(CSV_FILE_PATH):

    label   = preprocessing.LabelEncoder() 

    df = pd.read_csv(CSV_FILE_PATH)
    missing = df.isnull().values.any()

    if missing == True:
        df = df.ffill().bfill()
        print("null values filled")
    else:
        print("no null values encountered")
    
    object_col = df.select_dtypes(include=['object']).columns

    for name in object_col:
        df[name] = label.fit_transform(df[name]) 

    df.drop_duplicates(inplace=True)

    X = df.iloc[:,:-1]
    y = df[df.columns[-1]]

    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=.3,random_state =123)
    reg = LazyRegressor(verbose=0,ignore_warnings=True, custom_metric=None)

    models,predictions = reg.fit(X_train, X_test, y_train, y_test)
    
    print(models)

    return models

def startpy():
    
    CSV_FILE_PATH = "../datasets/vegetable_price.csv"
    regressor(CSV_FILE_PATH)


if __name__ == '__main__':
    startpy()