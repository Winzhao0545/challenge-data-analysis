#Import neccessary liberies 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression 
from sklearn.model_selection import  cross_val_score

# read data
file=r'C:\Users\frede_0021xgx\OneDrive\Documents\GitHub\challenge-data-analysis\data\dataframe_cleaned.csv'
immo_info=pd.read_csv(file)
#drop the irrelevent features,locality has conflits with zipcode,so delete it 
immo_info=immo_info.drop(['Unnamed: 0.1','Unnamed: 0','url','locality'],axis=1)

''' Data cleaning'''
#1.drop duplicates
immo_info=immo_info.drop_duplicates()
#2.No NaNs(replace NaNs with 0)
immo_info=immo_info.fillna(0)
#3.No text data(replace 'UNKNOWN' value with zero)
immo_info=immo_info.replace('UNKNOWN',0)
#4.subset the data positive value  
    #price  
immo_info=immo_info[immo_info['price']>0.0] 
#5.turn surface_land to type float 
immo_info['surface_land']=immo_info['surface_land'].astype(float)
# keep first 2 number of the zipcode to avoid the negative model score
for index, item in immo_info['zip_code'].items():
    
    immo_info['zip_code'].loc[index]=item[:2]

#6.Devide dataset into two parts
text_value=['region','province','property_type','property_subtype','building_state','kitchen','zip_code']
number_value=['number_rooms','living_area','terrace_area','garden_area','surface_land','number_facades']

#7.remove outliers from numeber_value
for col in immo_info[number_value]:
    mean = immo_info[col].mean()
    std_dev = immo_info[col].std()
    upper_limit = mean+3*std_dev
    lower_limit = mean-3*std_dev
    print(immo_info[col].max(),immo_info[col].min())
    immo_info[col] = np.where(immo_info[col] >upper_limit,upper_limit,immo_info[col])
    immo_info[col] = np.where(immo_info[col] <lower_limit,lower_limit,immo_info[col])
    print(immo_info[col].max(),immo_info[col].min())

''' Data formating'''
#Convert text_value to numeric value
dummies_df = pd.get_dummies(immo_info[text_value],prefix=text_value,dtype=float)
result = pd.concat([immo_info, dummies_df], axis=1).drop(text_value,axis = 1)
result.head()

#Convert dataframe to numpy array
X=result.drop('price',axis=1).to_numpy()
y=result['price'].to_numpy().reshape(-1,1)

X.shape,y.shape

#Split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=1,test_size=0.2)

'''Model training'''
#initialize the model
regressor=LinearRegression()
#fit model with data
regressor.fit(X_train, y_train)
#display score of model
score_train=regressor.score(X_train, y_train)
#See how the prediction match the y test
score_test=regressor.score(X_test, y_test)
print('Train_score:',score_train)
print('Test score:',score_test)

'''Model evaluating'''
#See the accuracy of the model by cross-validation
regressor=LinearRegression()
scores = cross_val_score(regressor, X, y, cv = 5)
print("Cross Validation Scores: ", scores)
print("Accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean() * 100, scores.std() * 100))
