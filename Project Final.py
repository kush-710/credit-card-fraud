#!/usr/bin/env python
# coding: utf-8

# # CSE3013- ARTIFICIAL INTELLIGENCE
# # TOPIC- CREDIT CARD FRAUD DETECTION
# ***

# ## Team Members-
# ### Ananya Pantvaidya (20BCE0678)
# ### Arsh Ansari (20BCE0371)
# ### Kushagra Srivastava (20BCE2060)
# ### Priadarshni Muthukumar (20BCE2510)
# ***

# ### Importing the necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')


# ### Importing the dataset
# ##### We have downloaded our dataset from Kaggle

# In[2]:


data=pd.read_csv("creditcard.csv")


# ### Taking a look at the features of the dataset

# In[3]:


print(data.columns)


# ##### V1-V28 are the result of a PCA dimensionality reduction 

# In[4]:


data.head()


# In[5]:


data.tail()


# ##### In the following cell we can see the datatypes of the features

# In[6]:


data.info()


# ##### Now we check for null values

# In[7]:


data.isnull().sum()


# ##### We see that in this dataset, there are no null values anywhere.

# In[8]:


data=data.sample(frac=0.1, random_state=1)

print(data.shape)


# ##### We will be using about 10% of the dataset
# ### Classifying the transactions into safe and fraudulent ones
# ##### Now we will check the column "Class" to see how many safe and fraudulent transactions there are in the dataset

# In[9]:


data["Class"].value_counts()


# In[10]:


safe=data[data.Class==0]
fraud=data[data.Class==1]
print('No. of safe cases= {}'.format(len(safe)))
print('No. of fraud cases= {}'.format(len(fraud)))

frac=len(fraud)/len(safe)
print('Fraction of fraud cases= {}'.format(frac))


# ##### We can see that the number of fraud transactions is very less compared to the number of safe transactions (only about 0.17%)

# In[11]:


print(safe.shape)


# ##### We have printed the shape of the safe transactions part of the dataset
# ##### Now we will see the shape of the fraud transactions part of the dataset

# In[12]:


print(fraud.shape)


# In[13]:


safe.Amount.describe()


# In[14]:


fraud.Amount.describe()


# ### Visualising the dataset

# In[15]:


data.hist(figsize=(20,20))
plt.show()


# ##### Here we have printed the histograms for the dataset we are using. We can see in the histogram for the "Class" column that the number of fraudulent transactions is negligible compared to the safe transactions
# 
# ##### Next we look at a relational plot

# In[16]:


sns.relplot(x='Amount', y='Time', hue='Class', data=data)
plt.show()


# ##### Now, we make a correlation matrix, and then a heatmap based on that

# In[17]:


cormat=data.corr()
fig=plt.figure(figsize=(12,8))

sns.heatmap(cormat, square=True)
plt.show()


# ### Importing the libraries for processing our dataset

# In[18]:


from sklearn import linear_model
from sklearn.model_selection import train_test_split


# ### Segregating the data into features and target

# In[19]:


X=data.iloc[:,:-1]
y=data["Class"]


# In[20]:


X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2)


# ##### We have taken our test size as 20% of the dataset

# In[21]:


print(X.shape)
print(y.shape)


# ##### Here we can see that the dataset has been split into the features and target
# ### Importing our classifiers
# ##### We will now import the IsolationForest classifier, and the LocalOutlierFactor classifier. They are used for anomaly detection.
# ##### The LocalOutlierFactor calculates the anomaly score of each sample. It measures the local deviation of density of a given sample with respect to its neighbours.
# ##### The IsolationForest isolates the observations by randomly selecting a feature and then randomly selecting a value between the minimum and maximum values of that select feature. It returns an anomaly score.

# In[22]:


from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor


# ##### Defining a random state

# In[23]:


state=1


# ### Defining the outlier detection methods

# In[24]:


classifiers={
    "Isolation Forest": IsolationForest(max_samples=len(X),
                                       contamination=frac,
                                       random_state=state),
    "Local Outlier Factor": LocalOutlierFactor(n_neighbors=20,
                                               contamination=frac),
    "Random Forest": RandomForestClassifier(max_samples=len(X),
                                            random_state=state)
}


# ### Fitting the model

# In[25]:


n_outliers=len(fraud)

for i, (clf_name, clf) in enumerate(classifiers.items()):
    #fit data and tag outliers
    if clf_name=='Local Outlier Factor':
        y_pred=clf.fit_predict(X)
        score_pred=clf.negative_outlier_factor_
    elif clf_name=='Isolation Forest':
        clf.fit(X)
        scores_pred=clf.decision_function(X)
        y_pred=clf.fit_predict(X)
    else:
        clf.fit(X, y)
        y_pred=clf.predict(X)
        
    #reshape pred vals to 0 for valid, 1 for fraud
    y_pred[y_pred==1] = 0
    y_pred[y_pred==-1]= 1
    
    n_errors=(y_pred != y).sum()
    
    #run classification metrics
    print("{}:{}".format(clf_name, n_errors))
    print(accuracy_score(y, y_pred))
    print(classification_report(y, y_pred))


# ***
