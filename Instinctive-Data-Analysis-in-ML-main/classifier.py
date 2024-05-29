# Import necessary modules
import pandas as pd
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split
from sklearn import preprocessing 
from lazypredict.Supervised import LazyClassifier


# CSV_FILE_PATH = "datasets/vegetable_price.csv"

def classifier(CSV_FILE_PATH):

    label   = preprocessing.LabelEncoder() 

    df      =  pd.read_csv(CSV_FILE_PATH)
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
    Y = df[df.columns[-1]]

    X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=.3,random_state =23)
    classi=LazyClassifier(verbose=0,predictions=True)

    models_c, predictions_c = classi.fit(X_train, X_test, Y_train, Y_test)

    print(models_c)

    return models_c

def startpy():
    
    CSV_FILE_PATH = "../datasets/Iris.csv"
    classifier(CSV_FILE_PATH)


if __name__ == '__main__':
    startpy()
