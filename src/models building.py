import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import  cross_val_score

def two_digit(df,modified_value):
    '''
    This function is to reduce the zip_code to 2 digits to prevent the model score from being negative.
    
    '''

    for index, item in df[modified_value].items():
    
        df[modified_value].loc[index]=item[:2]

    return df

df=pd.read_csv('data\cleaned_1.csv')
modified_value='zip_code'
two_digit(df,modified_value)    



def data_formatting(df,categorical_values):
    ''' 
    This function is to format the data in order to fit into the Machine learning model.
    '''
    
    #Convert categorical values to numeric values
    dummies_df = pd.get_dummies(df[categorical_values],prefix=categorical_values,dtype=float)
    merged_df = pd.concat([df, dummies_df], axis=1).drop(categorical_columns,axis = 1)
    return merged_df

    
df=two_digit(df,modified_value) 
categorical_columns=['region','province','property_type','property_subtype','building_state','kitchen','zip_code']

data_formatting(df,categorical_columns)


'''Global variables for following 2 models'''

df=data_formatting(df,categorical_columns)
target='price'
#Convert dataframe to numpy array
X=df.drop(target,axis=1).to_numpy()
y=df[target].to_numpy().reshape(-1,1)
      
#Split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1,test_size=0.2)

def Linearmodel_score():
    '''
    This function is to show the train and test score of Linearregression model
    '''
    #initialize the model
    regressor=LinearRegression()
    #fit model with data
    regressor.fit(X_train, y_train)
    #show train and test score
    score_train=regressor.score(X_train, y_train)
    score_test=regressor.score(X_test, y_test)  
    
    print('Train_score:',score_train, 'Test score:',score_test)

Linearmodel_score()


'''Global variables for following 2 models'''

df=data_formatting(df,categorical_columns)
target='price'
#Convert dataframe to numpy array
X=df.drop(target,axis=1).to_numpy()
y=df[target].to_numpy().reshape(-1,1)
      
#Split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1,test_size=0.2)

def Linearmodel_score():
    '''
    This function is to show the train and test score of Linearregression model
    '''
    #initialize the model
    regressor=LinearRegression()
    #fit model with data
    regressor.fit(X_train, y_train)
    #show train and test score
    score_train=regressor.score(X_train, y_train)
    score_test=regressor.score(X_test, y_test)  
    
    print('Train_score:',score_train, 'Test score:',score_test)

Linearmodel_score()


def cross_value():
    '''
    This function is to show the actual acurracy of Decisiontreeregression model by using cross value method
    '''
    regressor=DecisionTreeRegressor(random_state=1)
    scores = cross_val_score(regressor, X, y, cv=5) # cv is the number of folds (k)
    print(scores)
    print("Accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean() * 100, scores.std() * 100))
    
cross_value()   
