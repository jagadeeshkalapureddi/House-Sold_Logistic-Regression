
# # --------------------------------@ HOUSE_PRICE DATA ANALYSIS @---------------------------------------
# # ------------------------------------! LOG REGRESSION !----------------------------------------------

# # IMPORT THE REQUIRED PACKAGES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns

# ### READ THE DATASET

HP = pd.read_csv('House-Price.csv')

# ### DATA UNDERSTANDING

HP.head()

HP.shape

HP.info()

HP.isna().sum()

# #### There are presence of Null values.

# ## Individual Variable Understanding.

# ## Dependent Variable.  :  Sold

# Categorical Variable.
print('Column_name : ' ,HP.iloc[:,18].name)
print('Type : ',HP.iloc[:,18].dtype)

print('Null_value_count: ',HP.iloc[:,18].isna().sum())

HP.iloc[:,18].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'

HP.iloc[:,18].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,18].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,18].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)

# ## Independent Variables.

print('Column_name : ' ,HP.iloc[:,0].name)
print('Type : ',HP.iloc[:,0].dtype)
print('Null_value_count: ',HP.iloc[:,0].isna().sum())

print('Skewness: ', HP.iloc[:,0].skew())
HP.iloc[:,0].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,0], color = 'green')
plt.xlabel(HP.iloc[:,0].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,0].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,0], color = 'orange')
plt.xlabel(HP.iloc[:,0].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,0].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# #### No Outliers presence.

print('Column_name : ' ,HP.iloc[:,1].name)
print('Type : ',HP.iloc[:,1].dtype)
print('Null_value_count: ',HP.iloc[:,1].isna().sum())

print('Skewness: ', HP.iloc[:,1].skew())
HP.iloc[:,1].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,1], color = 'green')
plt.xlabel(HP.iloc[:,1].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,1].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,1], color = 'orange')
plt.xlabel(HP.iloc[:,1].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,1].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ### Combined plot

## Cut the window in 2 parts
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Add a graph in each part
plt.suptitle('Box_Dist_Plot ' + HP.iloc[:,1].name, fontsize = 15)
sns.boxplot(HP.iloc[:,1], ax=ax_box, color = 'blue')
sns.distplot(HP.iloc[:,1], ax=ax_hist, color = 'orange')
plt.show()

# ## No Outliers presence in resid_area variable.

## pairplot
sns.pairplot(data = HP, x_vars = 'resid_area', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('Type : ',HP.iloc[:,2].dtype)
print('Column_name : ' ,HP.iloc[:,2].name)

print('Null_value_count: ',HP.iloc[:,2].isna().sum())

print('Skewness: ', HP.iloc[:,1].skew())
HP.iloc[:,2].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,2], color = 'green')
plt.xlabel(HP.iloc[:,2].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,2].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,2], color = 'orange')
plt.xlabel(HP.iloc[:,2].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,2].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in air_qual variable.

##pairplot
sns.pairplot(data = HP, x_vars = 'air_qual', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('Type : ',HP.iloc[:,3].dtype)
print('Column_name : ' ,HP.iloc[:,3].name)

print('Null_value_count: ',HP.iloc[:,3].isna().sum())

print('Skewness: ', HP.iloc[:,3].skew())
HP.iloc[:,3].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,3], color = 'green')
plt.xlabel(HP.iloc[:,3].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,3].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,3], color = 'orange')
plt.xlabel(HP.iloc[:,3].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,3].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in room_num variable.

# # pairplot

sns.pairplot(data = HP, x_vars = 'room_num', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('Type : ',HP.iloc[:,4].dtype)
print('Column_name : ' ,HP.iloc[:,4].name)

print('Null_value_count: ',HP.iloc[:,4].isna().sum())

print('Skewness: ', HP.iloc[:,4].skew())
HP.iloc[:,4].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,4], color = 'green')
plt.xlabel(HP.iloc[:,4].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,4].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,4], color = 'orange')
plt.xlabel(HP.iloc[:,4].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,4].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in age variable.

# # pairplot

sns.pairplot(data = HP, x_vars = 'age', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# #  5,6,7,8  here we have four variables with relevant data so merge them into one variable.

print('Type : ',HP.iloc[:,5].dtype)
print('Column_name : ' ,HP.iloc[:,5].name)

print('Null_value_count: ',HP.iloc[:,5:9].isna().sum())

HP["avg_dist"]=HP[["dist1","dist2","dist3","dist4"]].mean(axis=1)
HP.drop(["dist1","dist2","dist3","dist4"], axis=1, inplace=True)

print('Column_Index_Number :',HP.columns.get_loc('avg_dist'))

print('Skewness: ', HP.iloc[:,15].skew())
HP.iloc[:,15].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,15], color = 'green')
plt.xlabel(HP.iloc[:,15].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,15].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,15], color = 'orange')
plt.xlabel(HP.iloc[:,15].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,15].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in avg_dist variable.

##pairplot
sns.pairplot(data = HP, x_vars = 'avg_dist', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('Type : ',HP.iloc[:,5].dtype)
print('Column_name : ' ,HP.iloc[:,5].name)

print('Null_value_count: ',HP.iloc[:,5].isna().sum())

print('Skewness: ', HP.iloc[:,5].skew())
HP.iloc[:,5].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,5], color = 'green')
plt.xlabel(HP.iloc[:,5].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,5].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,5], color = 'orange')
plt.xlabel(HP.iloc[:,5].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,5].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in teachers variable.

##pairplot
sns.pairplot(data = HP, x_vars = 'teachers', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('Type : ',HP.iloc[:,6].dtype)
print('Column_name : ' ,HP.iloc[:,6].name)

print('Null_value_count: ',HP.iloc[:,6].isna().sum())

print('Skewness: ', HP.iloc[:,6].skew())
HP.iloc[:,6].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,6], color = 'green')
plt.xlabel(HP.iloc[:,6].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,6].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,6], color = 'orange')
plt.xlabel(HP.iloc[:,6].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,6].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in poor_prop variable.

# # pairplot

sns.pairplot(data = HP, x_vars = 'poor_prop', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('Type : ',HP.iloc[:,7].dtype)
print('Column_name : ' ,HP.iloc[:,7].name)

print('Null_value_count: ',HP.iloc[:,7].isna().sum())

HP.iloc[:,7].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'

HP.iloc[:,7].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,7].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,7].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)

# ## Convert the string to integer.

HP.iloc[:,7] = np.where(HP.iloc[:,7] == 'YES', 1, 0)
HP.iloc[:,7].head()

print('Type : ',HP.iloc[:,8].dtype)
print('Column_name : ' ,HP.iloc[:,8].name)

print('Null_value_count: ',HP.iloc[:,8].isna().sum())

# ## Having Null Values.  So replace with its mean value.

HP.iloc[:,8] = np.where(HP.iloc[:,8].isnull() == True, np.mean(HP.iloc[:,8]), HP.iloc[:,8])
HP.iloc[:,8].isna().sum()

print('Skewness: ', HP.iloc[:,8].skew())
HP.iloc[:,8].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,8], color = 'green')
plt.xlabel(HP.iloc[:,8].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,8].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,8], color = 'orange')
plt.xlabel(HP.iloc[:,8].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,8].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in n_hos_beds variable.

##pairplot

sns.pairplot(data = HP, x_vars = 'n_hos_beds', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('Type : ',HP.iloc[:,9].dtype)
print('Column_name : ' ,HP.iloc[:,9].name)

print('Null_value_count: ',HP.iloc[:,9].isna().sum())

print('Skewness: ', HP.iloc[:,9].skew())
HP.iloc[:,9].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,9], color = 'green')
plt.xlabel(HP.iloc[:,9].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,9].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,9], color = 'orange')
plt.xlabel(HP.iloc[:,9].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,9].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ### Combined plot

## Cut the window in 2 parts
f, (ax_box, ax_hist) = plt.subplots(2, sharex=True, gridspec_kw={"height_ratios": (.15, .85)})
# Add a graph in each part
plt.suptitle('Box_Dist_Plot ' + HP.iloc[:,9].name, fontsize = 15)
sns.boxplot(HP.iloc[:,9], ax=ax_box, color = 'blue')
sns.distplot(HP.iloc[:,9], ax=ax_hist, color = 'orange')
plt.show()

# ## It is having outliers, then check for quantile ranges.

print('5% :', HP.iloc[:,9].quantile(0.05), '\n','95% :', HP.iloc[:,9].quantile(0.95))

import numpy as np
HP.iloc[:,9] = np.where(HP.iloc[:,9] > HP.iloc[:,9].quantile(0.95), np.median(HP.iloc[:,9]), HP.iloc[:,9])
HP.iloc[:,9].describe()

print('Skewness :' ,HP.iloc[:,9].skew())
sns.boxplot(HP.iloc[:,9], color = 'green')
plt.xlabel(HP.iloc[:,9].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,9].name, fontsize = 25)

plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in n_hot_rooms variable.

##pairplot
sns.pairplot(data = HP, x_vars = 'n_hot_rooms', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('Type : ',HP.iloc[:,10].dtype)
print('Column_name : ' ,HP.iloc[:,10].name)

print('Null_value_count: ',HP.iloc[:,10].isna().sum())

HP.iloc[:,10].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'

HP.iloc[:,10].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,10].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,10].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)

# ## We have four categorical elements in the waterbody so we are going for dummification to seperate those four elements into four variables.

status = pd.get_dummies(HP.iloc[:,10], drop_first = True)
HP = pd.concat([HP,status],axis=1)
HP.head()

HP.drop('waterbody', axis=1, inplace = True)
HP.head()

print('Type : ',HP.iloc[:,10].dtype)
print('Column_name : ' ,HP.iloc[:,10].name)

print('Null_value_count: ',HP.iloc[:,10].isna().sum())

print('Skewness: ', HP.iloc[:,10].skew())
HP.iloc[:,10].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,10], color = 'green')
plt.xlabel(HP.iloc[:,10].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,10].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,10], color = 'orange')
plt.xlabel(HP.iloc[:,10].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,10].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in rainfall variable.

##pairplot
sns.pairplot(data = HP, x_vars = 'rainfall', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

print('Type : ',HP.iloc[:,11].dtype)
print('Column_name : ' ,HP.iloc[:,11].name)

print('Null_value_count: ',HP.iloc[:,11].isna().sum())

HP.iloc[:,11].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'

HP.iloc[:,11].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,11].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,11].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)

# ## It is having only one categorical element so, it wont affect the model.

HP.drop('bus_ter', axis = 1, inplace = True)

print('Type : ',HP.iloc[:,11].dtype)
print('Column_name : ' ,HP.iloc[:,11].name)

print('Null_value_count: ',HP.iloc[:,11].isna().sum())

print('Skewness: ', HP.iloc[:,11].skew())
HP.iloc[:,11].describe()

plt.subplot(1,2,1)
sns.boxplot(HP.iloc[:,11], color = 'green')
plt.xlabel(HP.iloc[:,11].name, fontsize = 20)
plt.title('Boxplot_ '+ HP.iloc[:,11].name, fontsize = 15)
plt.subplot(1,2,2)
plt.hist(HP.iloc[:,11], color = 'orange')
plt.xlabel(HP.iloc[:,11].name, fontsize = 20)
plt.title('Histogram_ '+ HP.iloc[:,11].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# ## No Outliers presence in parks variable.

##pairplot
sns.pairplot(data = HP, x_vars = 'parks', y_vars = 'price', height = 5, aspect = 0.7, kind = 'scatter')
plt.subplots_adjust(left=0.4, bottom=0.1, right=2.2, top=1.2)

# # 14,15,16  ---- 12 - Sold & 13 - avg_dist done already

print('Type : ',HP.iloc[:,14].dtype)
print('Column_name : ' ,HP.iloc[:,14].name)

print('Null_value_count: ',HP.iloc[:,14].isna().sum())

HP.iloc[:,14].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'

HP.iloc[:,14].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,14].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,14].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)

print('Type : ',HP.iloc[:,15].dtype)
print('Column_name : ' ,HP.iloc[:,15].name)

print('Null_value_count: ',HP.iloc[:,15].isna().sum())

HP.iloc[:,15].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'

HP.iloc[:,15].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,15].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,15].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)

print('Type : ',HP.iloc[:,16].dtype)
print('Column_name : ' ,HP.iloc[:,16].name)

print('Null_value_count: ',HP.iloc[:,16].isna().sum())

HP.iloc[:,16].value_counts(normalize = True).mul(100).round(1).astype(str) + '%'

HP.iloc[:,16].value_counts().plot(kind = 'bar', color = 'blue')
plt.xlabel(HP.iloc[:,16].name, fontsize = 20)
plt.title('Barplot_ '+ HP.iloc[:,16].name, fontsize = 15)
plt.subplots_adjust(left=0.4, bottom=0.1, right=1.1, top=1.2)

HP['Sold1'] = HP['Sold']
HP.drop('Sold', axis = 1, inplace = True)
HP.info()

# # Spliting the data.

x_final = HP.iloc[:,0:16]

y = HP.iloc[:,16]

# ### For Building Model:
# 
# **x_final** is the x variables.
# **y** is the y variable.
# 
# ### Split the data into x and y for training and testing :

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x_final, y, train_size = 0.8, random_state = 100)
print('x_train shape :', x_train.shape)
print('x_test shape :', x_test.shape)
print('y_train shape :', y_train.shape)
print('y_test shape :', y_test.shape)

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm = LogisticRegression()
glm = glm.fit(x_train, y_train)
predicted = glm.predict(x_test)


# ## Model Evaluation :
# 
# ### Full Model:

# ### RFE

import warnings
warnings.filterwarnings("ignore")

from sklearn.feature_selection import RFE
rfe = RFE(glm, 1)
rfe = rfe.fit(x_train, y_train)
print(rfe.support_)
print(rfe.ranking_)

# ### VIF 

from statsmodels.stats.outliers_influence import variance_inflation_factor

vif_data = pd.DataFrame() 
vif_data["feature"] = x_final.columns

vif_data["VIF"] = [variance_inflation_factor(x_final.values, i) 
                          for i in range(len(x_final.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x_train_sm = sm.Logit(y_train, x_train)
lm = x_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm.summary())

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

def plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle=':')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()
    
def get_summary(y_test,predicted):
    # Confusion Matrix
    conf_mat = confusion_matrix(y_test,predicted)
    TP = conf_mat[0,0:1]
    FP = conf_mat[0,1:2]
    FN = conf_mat[1,0:1]
    TN = conf_mat[1,1:2]
    
    accuracy = (TP+TN)/((FN+FP)+(TP+TN))
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    precision = TP/(TP+FP)
    recall =  TP / (TP + FN)
    fScore = (2 * recall * precision) / (recall + precision)
    auc = roc_auc_score(y_test, predicted)

    print("Confusion Matrix:\n",conf_mat)
    print("Accuracy:",accuracy)
    print("Sensitivity :",sensitivity)
    print("Specificity :",specificity)
    print("Precision:",precision)
    print("Recall:",recall)
    print("F-score:",fScore)
    print("AUC:",auc)
    print('\n')
    print("ROC curve:")
    fpr, tpr, thresholds = roc_curve(y_test, predicted)
    plot_roc_curve(fpr, tpr)

get_summary(y_test,predicted)


# ### Model-1 (Backward Elimination) --- age

# ### Train and Test the data.

x1_train = x_train[['price', 'resid_area', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms', 'rainfall',
       'parks', 'avg_dist', 'Lake and River', 'None', 'River']]

x1_test = x_test[['price', 'resid_area', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms', 'rainfall',
       'parks', 'avg_dist', 'Lake and River', 'None', 'River']]


# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x1_train.columns

vif_data["VIF"] = [variance_inflation_factor(x1_train.values, i) 
                          for i in range(len(x1_train.columns))] 
  
print(vif_data)


# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x1_train_sm = sm.Logit(y_train, x1_train)
lm1 = x1_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm1.summary())


# ### Fit the train model and predict the test set

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm1 = LogisticRegression()
glm1 = glm1.fit(x1_train, y_train)
predicted1 = glm1.predict(x1_test)


# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score


# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

get_summary(y_test,predicted1)

# ### Model-2 (Backward Elimination) -- Lake & River

# ### Train and Test the data.

x2_train = x_train[['price', 'resid_area', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms', 'rainfall',
       'parks', 'avg_dist', 'None', 'River']]

x2_test = x_test[['price', 'resid_area', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms', 'rainfall',
       'parks', 'avg_dist', 'None', 'River']]


# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x2_train.columns

vif_data["VIF"] = [variance_inflation_factor(x2_train.values, i) 
                          for i in range(len(x2_train.columns))] 
  
print(vif_data)


# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x2_train_sm = sm.Logit(y_train, x2_train)
lm2 = x2_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm2.summary())


# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm2 = LogisticRegression()
glm2 = glm2.fit(x2_train, y_train)
predicted2 = glm2.predict(x2_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

get_summary(y_test,predicted2)

# ### Model-3 (Backward Elimination) -- resid_area

# ### Train and Test the data.

x3_train = x_train[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms', 'rainfall',
       'parks', 'avg_dist', 'None', 'River']]

x3_test = x_test[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms', 'rainfall',
       'parks', 'avg_dist', 'None', 'River']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x3_train.columns

vif_data["VIF"] = [variance_inflation_factor(x3_train.values, i) 
                          for i in range(len(x3_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x3_train_sm = sm.Logit(y_train, x3_train)
lm3 = x3_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm3.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm3 = LogisticRegression()
glm3 = glm3.fit(x3_train, y_train)
predicted3 = glm3.predict(x3_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

get_summary(y_test,predicted3)

# ### Model-4 (Backward Elimination) -- rainfall

# ### Train and Test the data.

x4_train = x_train[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms',
       'parks', 'avg_dist', 'None', 'River']]

x4_test = x_test[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms',
       'parks', 'avg_dist', 'None', 'River']]


# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x4_train.columns

vif_data["VIF"] = [variance_inflation_factor(x4_train.values, i) 
                          for i in range(len(x4_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x4_train_sm = sm.Logit(y_train, x4_train)
lm4 = x4_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm4.summary())


# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm4 = LogisticRegression()
glm4 = glm4.fit(x4_train, y_train)
predicted4 = glm4.predict(x4_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

get_summary(y_test,predicted4)


# ### Model-5 (Backward Elimination) -- None

# ### Train and Test the data.

x5_train = x_train[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms',
       'parks', 'avg_dist', 'River']]

x5_test = x_test[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'airport', 'n_hos_beds', 'n_hot_rooms',
       'parks', 'avg_dist', 'River']]


# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x5_train.columns

vif_data["VIF"] = [variance_inflation_factor(x5_train.values, i) 
                          for i in range(len(x5_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x5_train_sm = sm.Logit(y_train, x5_train)
lm5 = x5_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm5.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm5 = LogisticRegression()
glm5 = glm5.fit(x5_train, y_train)
predicted5 = glm5.predict(x5_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

get_summary(y_test,predicted5)

# ### Model-6 (Backward Elimination) -- airport

# ### Train and Test the data.

x6_train = x_train[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'n_hos_beds', 'n_hot_rooms',
       'parks', 'avg_dist', 'River']]

x6_test = x_test[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'n_hos_beds', 'n_hot_rooms',
       'parks', 'avg_dist', 'River']]


# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x6_train.columns

vif_data["VIF"] = [variance_inflation_factor(x6_train.values, i) 
                          for i in range(len(x6_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x6_train_sm = sm.Logit(y_train, x6_train)
lm6 = x6_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm6.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm6 = LogisticRegression()
glm6 = glm6.fit(x6_train, y_train)
predicted6 = glm6.predict(x6_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

get_summary(y_test,predicted6)

# ### Model-7 (Backward Elimination) -- parks

# ### Train and Test the data.

x7_train = x_train[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'n_hos_beds', 'n_hot_rooms',
       'avg_dist', 'River']]

x7_test = x_test[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'n_hos_beds', 'n_hot_rooms',
       'avg_dist', 'River']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x7_train.columns

vif_data["VIF"] = [variance_inflation_factor(x7_train.values, i) 
                          for i in range(len(x7_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x7_train_sm = sm.Logit(y_train, x7_train)
lm7 = x7_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm7.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm7 = LogisticRegression()
glm7 = glm7.fit(x7_train, y_train)
predicted7 = glm7.predict(x7_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

get_summary(y_test,predicted7)

# ### Model-8 (Backward Elimination) -- hot_rooms

# ### Train and Test the data.

x8_train = x_train[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'n_hos_beds',
       'avg_dist', 'River']]

x8_test = x_test[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'n_hos_beds',
       'avg_dist', 'River']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x8_train.columns

vif_data["VIF"] = [variance_inflation_factor(x8_train.values, i) 
                          for i in range(len(x8_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x8_train_sm = sm.Logit(y_train, x8_train)
lm8 = x8_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm8.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm8 = LogisticRegression()
glm8 = glm8.fit(x8_train, y_train)
predicted8 = glm8.predict(x8_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

print(get_summary(y_test,predicted8))

# ### Model-9 (Backward Elimination) -- River

# ### Train and Test the data.

x9_train = x_train[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'n_hos_beds',
       'avg_dist']]

x9_test = x_test[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'n_hos_beds',
       'avg_dist']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x9_train.columns

vif_data["VIF"] = [variance_inflation_factor(x9_train.values, i) 
                          for i in range(len(x9_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x9_train_sm = sm.Logit(y_train, x9_train)
lm9 = x9_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm9.summary())

# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm9 = LogisticRegression()
glm9 = glm9.fit(x9_train, y_train)
predicted9 = glm9.predict(x9_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

print(get_summary(y_test,predicted9))

# ### Model-10 (Backward Elimination) -- no_beds

# ### Train and Test the data.

x10_train = x_train[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'avg_dist']]

x10_test = x_test[['price', 'air_qual', 'room_num', 'teachers',
       'poor_prop', 'avg_dist']]

# ### VIF

from statsmodels.stats.outliers_influence import variance_inflation_factor 

vif_data = pd.DataFrame() 
vif_data["feature"] = x10_train.columns

vif_data["VIF"] = [variance_inflation_factor(x10_train.values, i) 
                          for i in range(len(x10_train.columns))] 
  
print(vif_data)

# ### LOGIT OLS SUMMARY

import statsmodels.api as sm
x10_train_sm = sm.Logit(y_train, x10_train)
lm10 = x10_train_sm.fit(method = 'newton')

# Check for OLS Summary
print(lm10.summary())


# ### Fit the train model and predict the test set.

import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
glm10 = LogisticRegression()
glm10 = glm10.fit(x10_train, y_train)
predicted10 = glm10.predict(x10_test)

# ### IMPORT THE REQUIRED PACKAGES FOR CONFUSION_MATRIX AND ROC-AUC

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, roc_auc_score

# #### Lets prepare a user defined function to make the confusion_matrix and ROC-AUC curve.

# Function For Logistic Regression Create Summary For Logistic Regression

print(get_summary(y_test,predicted10))

# ## Final Selection:

conf_mat1 = confusion_matrix(y_test,predicted1)
TP1 = conf_mat1[0,0:1]
FP1 = conf_mat1[0,1:2]
FN1 = conf_mat1[1,0:1]
TN1 = conf_mat1[1,1:2]
accuracy1 = (TP1+TN1)/((FN1+FP1)+(TP1+TN1))

conf_mat2 = confusion_matrix(y_test,predicted2)
TP2 = conf_mat2[0,0:1]
FP2 = conf_mat2[0,1:2]
FN2 = conf_mat2[1,0:1]
TN2 = conf_mat2[1,1:2]
accuracy2 = (TP2+TN2)/((FN2+FP2)+(TP2+TN2))

conf_mat3 = confusion_matrix(y_test,predicted3)
TP3 = conf_mat3[0,0:1]
FP3 = conf_mat3[0,1:2]
FN3 = conf_mat3[1,0:1]
TN3 = conf_mat3[1,1:2]
accuracy3 = (TP3+TN3)/((FN3+FP3)+(TP3+TN3))

conf_mat4 = confusion_matrix(y_test,predicted4)
TP4 = conf_mat4[0,0:1]
FP4 = conf_mat4[0,1:2]
FN4 = conf_mat4[1,0:1]
TN4 = conf_mat4[1,1:2]
accuracy4 = (TP4+TN4)/((FN4+FP4)+(TP4+TN4))

conf_mat5 = confusion_matrix(y_test,predicted5)
TP5 = conf_mat5[0,0:1]
FP5 = conf_mat5[0,1:2]
FN5 = conf_mat5[1,0:1]
TN5 = conf_mat5[1,1:2]
accuracy5 = (TP5+TN5)/((FN5+FP5)+(TP5+TN5))

conf_mat6 = confusion_matrix(y_test,predicted6)
TP6 = conf_mat6[0,0:1]
FP6 = conf_mat6[0,1:2]
FN6 = conf_mat6[1,0:1]
TN6 = conf_mat6[1,1:2]
accuracy6 = (TP6+TN6)/((FN6+FP6)+(TP6+TN6))

conf_mat7 = confusion_matrix(y_test,predicted7)
TP7 = conf_mat7[0,0:1]
FP7 = conf_mat7[0,1:2]
FN7 = conf_mat7[1,0:1]
TN7 = conf_mat7[1,1:2]
accuracy7 = (TP7+TN7)/((FN7+FP7)+(TP7+TN7))

conf_mat8 = confusion_matrix(y_test,predicted8)
TP8 = conf_mat8[0,0:1]
FP8 = conf_mat8[0,1:2]
FN8 = conf_mat8[1,0:1]
TN8 = conf_mat8[1,1:2]
accuracy8 = (TP8+TN8)/((FN8+FP8)+(TP8+TN8))

conf_mat9 = confusion_matrix(y_test,predicted9)
TP9 = conf_mat9[0,0:1]
FP9 = conf_mat9[0,1:2]
FN9 = conf_mat9[1,0:1]
TN9 = conf_mat9[1,1:2]
accuracy9 = (TP9+TN9)/((FN9+FP9)+(TP9+TN9))

conf_mat10 = confusion_matrix(y_test,predicted10)
TP10 = conf_mat10[0,0:1]
FP10 = conf_mat10[0,1:2]
FN10 = conf_mat10[1,0:1]
TN10 = conf_mat10[1,1:2]
accuracy10 = (TP10+TN10)/((FN10+FP10)+(TP10+TN10))

# ### Print all accuracies for best selection.

print('accuracy1 : ', accuracy1)
print('\n')
print('Model-1 : ', x1_test.columns)
print('Variable_Count-1 : ', len(x1_test.columns))

print('\n')
print('accuracy2 : ', accuracy2)
print('\n')
print('Model-2 : ', x2_test.columns)
print('Variable_Count-2 : ', len(x2_test.columns))

print('\n')
print('accuracy3 : ', accuracy3)
print('\n')
print('Model-3 : ', x3_test.columns)
print('Variable_Count-3 : ', len(x3_test.columns))

print('\n')
print('accuracy4 : ', accuracy4)
print('\n')
print('Model-4 : ', x4_test.columns)
print('Variable_Count-4 : ', len(x4_test.columns))

print('\n')
print('accuracy5 : ', accuracy5)
print('\n')
print('Model-5 : ', x5_test.columns)
print('Variable_Count-5 : ', len(x5_test.columns))

print('\n')
print('accuracy6 : ', accuracy6)
print('\n')
print('Model-6 : ', x6_test.columns)
print('Variable_Count-6 : ', len(x6_test.columns))

print('\n')
print('accuracy7 : ', accuracy7)
print('\n')
print('Model-7 : ', x7_test.columns)
print('Variable_Count-7 : ', len(x7_test.columns))

print('\n')
print('accuracy8 : ', accuracy8)
print('\n')
print('Model-8 : ', x8_test.columns)
print('Variable_Count-8 : ', len(x8_test.columns))

print('\n')
print('accuracy9 : ', accuracy9)
print('\n')
print('Model-9 : ', x9_test.columns)
print('Variable_Count-9 : ', len(x9_test.columns))

print('\n')
print('accuracy10 : ', accuracy10)
print('\n')
print('Model-10 : ', x10_test.columns)
print('Variable_Count-10 : ', len(x10_test.columns))


# ### Final Model Summary Details of Tested and predicted data.

print(get_summary(y_test,predicted9))


# ### Finally we got the result :  Model-09
# 
# #### After building  a models we choosen the model 09 because of:
# 
# #### As per the model seen Accuracy - 66 % that the model get fitted with following Independent attributes -   
# 'price', 'air_qual', 'room_num', 'teachers', 'poor_prop', 'avg_dist'

# ---------------------------------------------------------------------------------------------------------------------**@** **Analysis done by,**
# ------------------------------------------------------------------------------------------------------------------------ **@**   **Jagadeesh K**

# #### Note : Don't run individual cells in between, Please run from starting onwards, Because its done on column Indexing based. 
