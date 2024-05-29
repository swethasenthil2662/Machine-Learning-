import streamlit as st  
import pandas as pd
import warnings  #avoid warnings
warnings.filterwarnings('ignore')
from sklearn.model_selection import train_test_split  #spliting training and test
from sklearn import preprocessing 
 #str to num converting (its is a module working with its label encoder)
from lazypredict.Supervised import LazyClassifier #27classifier module import
from lazypredict.Supervised import LazyRegressor# 42 regression module import
import matplotlib.pyplot as plt #final result img by graph

label   = preprocessing.LabelEncoder() #preprocessing fun

def main():
    st.title("Instinctive Data Analysis in Machine Learning")
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)#converting csv to dataframe
        st.write(df)#show df in website
        col1, col2, col3 = st.columns([1,1,1])
        with col2:
            if st.button('Classifier'):
                missing = df.isnull().values.any()#checking null values
                if missing == True:
                    df = df.ffill().bfill()#illing values on null
                    print("null values filled")
                else:
                    print("no null values encountered")
                
                object_col = df.select_dtypes(include=['object']).columns # using preprocessing store only str values to num
 
                for name in object_col:
                    df[name] = label.fit_transform(df[name]) #tranforming num process

                df.drop_duplicates(inplace=True)

                X = df.iloc[:,:-1]  #descriptive var (y based on max x val)
                Y = df[df.columns[-1]] #to store target variable

                X_train, X_test, Y_train, Y_test =train_test_split(X,Y,test_size=.3,random_state =23)#xtrain,y train - ip , xtest-for op,y test -backup to comoare to find accuracy and error
                classi=LazyClassifier(verbose=0,predictions=True)#verbose-log (details).predicting process (instruct)

                models_c, predictions_c = classi.fit(X_train, X_test, Y_train, Y_test) #actual running phase

                st.write(models_c)#printing result

                plot_df = pd.DataFrame(models_c)#show df

                plot_df = plot_df.reset_index()#ordervise printing

                print(plot_df)#printing it terminal

                plt.rcParams["figure.figsize"] =[16,9]
                plot_df.plot(x="Model",y="Accuracy",color="Red",kind="line",marker='o',markersize=12)
                plt.xlabel("Model")
                plt.ylabel('Accuracy')
                plt.grid()
                plt.show()
                st.set_option('deprecation.showPyplotGlobalUse', False)#ignore error

                st.pyplot(x="Model",y="Accuracy")

                return models_c

            
        with col3:
            if st.button('Regressor'):
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
                
                st.write(models,predictions)

                plot_data = pd.DataFrame(models)
                plot_data = plot_data.reset_index()

                plt.rcParams["figure.figsize"] =[16,9]
                plot_data.plot(x="Model",y="RMSE",color="Red",kind="line",marker='o',markersize=12)
                plt.xlabel("Model")
                plt.ylabel('RMSE')
                plt.grid()
                plt.show()

                st.set_option('deprecation.showPyplotGlobalUse', False)

                st.pyplot(x="Model",y="RMSE")

                return models



if __name__ == "__main__":
    main()