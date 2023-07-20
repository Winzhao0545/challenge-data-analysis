#Import neccessary liberies 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.tree import DecisionTreeRegressor

# read data
file=r'C:\Users\frede_0021xgx\OneDrive\Documents\GitHub\challenge-data-analysis\data\dataframe_cleaned.csv'
immo_info=pd.read_csv(file)
#drop the irrelevent features,locality has conflits with zipcode,so delete it 
immo_info=immo_info.drop(['Unnamed: 0.1','Unnamed: 0','url','zip_code'],axis=1)

'''Data cleaning'''
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

#6.Devide dataset into two parts
text_value=['region','province','property_type','property_subtype','building_state','kitchen','locality']
number_value=['number_rooms','living_area','terrace_area','garden_area','surface_land','number_facades']

#7.remove outliers
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
immo_info_merge = pd.concat([immo_info, dummies_df], axis=1).drop(text_value,axis = 1)
immo_info_merge.head()

#Convert dataframe to numpy array
X=immo_info_merge.drop('price',axis=1).to_numpy()
y=immo_info_merge['price'].to_numpy().reshape(-1,1)

X.shape,y.shape

'''Data training'''
#Split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=10,test_size=0.2)
#initialize the DecisionTreeRegressor model
d_reg=DecisionTreeRegressor()
#fit data in the model
d_reg.fit(X_train, y_train)
#display score of model
print('Train_score:',d_reg.score(X_train, y_train))
#See how the prediction match the y test
print('Test_score:',d_reg.score(X_test, y_test))



#Avoid overfitting of the model by cross-validation
d_reg=DecisionTreeRegressor(random_state=1)
scores = cross_val_score(d_reg, X, y, cv=5) # cv is the number of folds (k)
print(scores)
print("Accuracy: {:.2f}% (+/- {:.2f})".format(scores.mean() * 100, scores.std() * 100))