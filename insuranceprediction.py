#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


# In[63]:


insurance=pd.read_csv('C:/Users/hitix/OneDrive/Documents/dataset/insurance.csv')


# In[64]:


insurance.columns


# In[65]:


insurance.head()


# In[66]:


insurance.isna().sum()


# In[67]:


insurance.info()


# In[68]:


insurance.describe()


# In[69]:


insurance['sex'].value_counts()


# In[70]:


insurance['smoker'].value_counts()


# In[71]:


insurance['region'].value_counts()


# In[72]:


plt.figure()
smoker_bp=sns.countplot(x='region',data=insurance)
smoker_bp.set_xlabel("region", fontsize=8)
smoker_bp.set_ylabel("count", fontsize=8)


# In[73]:


plt.figure()
sns.countplot(x=insurance['children'])


# In[74]:


plt.figure()
smoker_bp=sns.countplot(x='smoker',data=insurance)
smoker_bp.set_xlabel("smokers", fontsize=8)
smoker_bp.set_ylabel("count", fontsize=8)


# In[75]:


plt.figure()
plt.title("Smoker Vs Expenses",fontsize=12)
bp3=sns.barplot(x='expenses',y='smoker',data=insurance)
bp3.set_xlabel("expenses", fontsize=8)
bp3.set_ylabel("smokers", fontsize=8)


# In[76]:


plt.figure()
plt.title("Children Vs Expenses",fontsize=12)
bp3=sns.barplot(x='children',y='expenses',data=insurance)
bp3.set_ylabel("expenses", fontsize=8)
bp3.set_xlabel("children", fontsize=8)
plt.show()


# In[77]:


plt.figure()
plt.title("region Vs Expenses",fontsize=12)
bp3=sns.barplot(x='region',y='expenses',data=insurance)
bp3.set_xlabel("Region", fontsize=8)
bp3.set_ylabel("Expenses", fontsize=8)


# In[78]:


plt.figure()
plt.title("Gender Vs Expenses",fontsize=12)
bp3=sns.barplot(x='sex',y='expenses',data=insurance)
bp3.set_xlabel('Gender', fontsize=8)
bp3.set_ylabel("Expenses", fontsize=8)


# In[79]:


plt.figure(figsize=(15,7))
plt.title("Distribution Of BMI",fontsize=25)
g=sns.distplot(insurance['bmi'])
g.set_xlabel("bmi", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)


# In[80]:


plt.figure(figsize=(15,7))
plt.title("Distribution Of Age",fontsize=25)
bp1=sns.histplot(x='age',data=insurance,bins=15,color='orange')
bp1.set_xlabel("Age", fontsize=16)
bp1.set_ylabel("Count", fontsize=16)


# In[81]:


plt.figure(figsize=(15,7))
plt.title("Distribution Of Age",fontsize=25)
bp1=sns.histplot(x='bmi',data=insurance,bins=25,color='orange')
bp1.set_xlabel("Age", fontsize=16)
bp1.set_ylabel("Count", fontsize=16)


# In[82]:


plt.figure()
plt.title("Distribution Of BMI",fontsize=25)
g=sns.distplot(insurance['expenses'])
g.set_xlabel("bmi", fontsize=16)
g.set_ylabel("Frequency", fontsize=16)


# In[83]:


plt.figure(figsize=(12,5))
sns.boxplot(x='bmi',data=insurance)


# In[84]:


plt.figure(figsize=(15,6))
plt.title("Age vs Expenses",fontsize=25)
bp7=sns.regplot(x='age',y='expenses',data=insurance)
bp7.set_xlabel("Age", fontsize=16)
bp7.set_ylabel("Expenses", fontsize=16)


# In[85]:


plt.figure(figsize=(15,6))
plt.title("BMI vs Expenses",fontsize=25)
bp7=sns.regplot(x='bmi',y='expenses',data=insurance)
bp7.set_xlabel("BMI", fontsize=16)
bp7.set_ylabel("Expenses", fontsize=16)


# In[86]:


plt.figure(figsize=(15,6))
bp6=sns.countplot(x=insurance.smoker,hue=insurance.sex)
bp6.set_xlabel("smoker", fontsize=16)
bp6.set_ylabel("count", fontsize=16)


# In[87]:


#smokers bmi vs expenses
plt.figure(figsize=(15,6))
plt.title("Smokers BMI vs Expenses",fontsize=25)
smokers_bmi=insurance[insurance['smoker']=='yes']
bp9=sns.regplot(x='bmi',y='expenses',data=smokers_bmi)
bp9.set_xlabel("BMI", fontsize=16)
bp9.set_ylabel("expenses", fontsize=16)


# In[26]:


smoker=(insurance['smoker']=='yes')
nonsmoker=(insurance['smoker']=='no')
insurance.loc[smoker,'smoker']=1
insurance.loc[nonsmoker,'smoker']=0


# In[27]:
#Converting object data to interger 

SER=(insurance['region']=='southeast')
SWR=(insurance['region']=='southwest')
NER=(insurance['region']=='northeast')
NWR=(insurance['region']=='northwest')
insurance.loc[SER,'region']=1
insurance.loc[SWR,'region']=2
insurance.loc[NER,'region']=3
insurance.loc[NWR,'region']=4


# In[28]:


men=(insurance['sex']=='male')
woman=(insurance['sex']=='female')
insurance.loc[men,'sex']=1
insurance.loc[woman,'sex']=0


# In[29]:


convert_dict={'sex':float,'smoker':float,'region':float}
insurance=insurance.astype(convert_dict)


# In[30]:


insurance.expenses.corr(insurance.smoker)


# In[31]:


cor=insurance.corr()
cor


# In[32]:



sns.heatmap(cor,fmt='.1g',cmap='Reds')


# In[33]:


sns.pairplot(data=insurance,hue='smoker')


# In[34]:


X=insurance.drop(['expenses'],axis=1)
Y=insurance['expenses']


# In[51]:


from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score,cross_validate
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.35,random_state=True)


# In[52]:


from sklearn.metrics import accuracy_score,confusion_matrix,r2_score,mean_squared_error
from math import sqrt


# In[53]:


#defining function for model summary
def model_summary(model,model_name,cvn=20):
    print(model_name)
    Y_pred_model_train = model.predict(X_train)
    Y_pred_model_test = model.predict(X_test)
    R2Score_model_train = r2_score(Y_train, Y_pred_model_train)
    print("Training R2 Score: ", R2Score_model_train)
    R2Score_model_test = r2_score(Y_test, Y_pred_model_test)
    print("Testing R2 Score: ",  R2Score_model_test)
    RMSE_model_train = sqrt(mean_squared_error(Y_train, Y_pred_model_train))
    print("RMSE for Training Data: ", RMSE_model_train)
    RMSE_model_test = sqrt(mean_squared_error(Y_test, Y_pred_model_test))
    print("RMSE for Testing Data: ", RMSE_model_test)
    Y_pred_cv_model = cross_val_predict(model, X, Y, cv=cvn)
    accuracy_cv_model = r2_score(Y, Y_pred_cv_model)
    print("Accuracy for", cvn,"- Fold Cross Predicted: ", accuracy_cv_model)


# In[54]:


#linaer Regration
from sklearn.linear_model import LinearRegression 
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
model_summary(regressor,'MultilinearRegression')


# In[55]:


#DecisionTree Regression
from sklearn.tree import DecisionTreeRegressor
decision_tree_reg = DecisionTreeRegressor(max_depth=5, random_state=13)  
decision_tree_reg.fit(X_train, Y_train) 
model_summary(decision_tree_reg, "Decision_Tree_Regression")


# In[56]:


# Randomforest Regression
from sklearn.ensemble import RandomForestRegressor
random_forest_reg=RandomForestRegressor()
random_forest_reg.fit(X_train,Y_train)
model_summary(random_forest_reg,"Random_Forest_Regressor")


# In[57]:


pip install xgboost


# In[58]:


import xgboost as xgb 
from xgboost import XGBRegressor
xgb_r = xgb.XGBRegressor(objective ='reg:linear',
                  n_estimators = 10, seed = 123,verbosity=0)

xgb_r.fit(X_train,Y_train)
model_summary(xgb_r,'XGBoost')


# In[ ]:




