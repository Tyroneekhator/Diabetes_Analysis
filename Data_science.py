#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd#importing pandas package
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from sklearn.metrics import f1_score, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler


import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


# In[2]:


data = pd.read_csv(r"C:\Users\ekhat\Downloads\diabetes.csv") #making data frame from csv file


# In[3]:


df = pd.DataFrame(data)


# In[4]:


print(df)


# In[5]:


print(df.dtypes)


# In[6]:


print(df['BMI'].dtypes)


# In[7]:


print(df.shape)


# In[8]:


print(df.index)


# In[9]:


print(df.columns)


# In[10]:


print(df.info())


# In[30]:


print(df.count())


# In[31]:


print(df.sum())


# In[32]:


print(df.cumsum())


# In[33]:


print(df.min())


# In[34]:


print(df.max())


# In[35]:


print(df.idxmin())


# In[36]:


print(df.idxmax())


# In[37]:


print(df.describe())


# In[38]:


print(df.mean())


# In[39]:


print(df.median())


# In[40]:


# making data frame from csv file
data = pd.read_csv(r"C:\Users\ekhat\Downloads\diabetes.csv")
 
# creating bool series True for NaN values
bool_series = pd.isnull(data["BMI"])
 
# filtering data
# displaying data only with team = NaN
data[bool_series]


# In[41]:


# making data frame from csv file
data = pd.read_csv(r"C:\Users\ekhat\Downloads\diabetes.csv")
 
# creating bool series True for NaN values
bool_series = pd.notnull(data["BMI"])
 
# filtering data
# displaying data only with team = NaN
data[bool_series]


# In[42]:


data.nunique()#checking for number of unique values in each column


# In[43]:


data['BMI'].unique()


# In[10]:


#Boxplot to check for outliers

fig, ax = plt.subplots(1, 7, figsize=(10, 6))

# draw boxplots - for one column in each subplot
df.boxplot('Glucose', ax=ax[0])
df.boxplot('BloodPressure', ax=ax[1])
df.boxplot('SkinThickness', ax=ax[2])
df.boxplot('Insulin', ax=ax[3])
df.boxplot('BMI', ax=ax[4])
df.boxplot('DiabetesPedigreeFunction', ax=ax[5])
df.boxplot('Age', ax=ax[6])

plt.subplots_adjust(wspace=0.8) 


# In[11]:


#removing outliers
def outliers(df,ft):
    q1 = df[ft].quantile(0.25)
    q3 = df[ft].quantile(0.75)
    IQR = q3-q1
    min= q1 - 1.5*IQR
    max= q3 + 1.5*IQR
    
    ls = df.index[(df[ft] < min) | (df[ft] > max)]
    return ls


# In[12]:


index_list=[]
for outlier in ['Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']:
    index_list.extend(outliers(df,outlier))


# In[13]:


def remove(df,ls):
    ls=sorted(set(ls))
    df=df.drop(ls)
    return df


# In[18]:


df2 = remove(df,index_list)
print('old dataset shape',df.shape)
print()
print('new shape--------',df2.shape)


# In[15]:


fig, ax = plt.subplots(1, 7, figsize=(10, 6))

# draw boxplots - for one column in each subplot
df2.boxplot('Glucose', ax=ax[0])
df2.boxplot('BloodPressure', ax=ax[1])
df2.boxplot('SkinThickness', ax=ax[2])
df2.boxplot('Insulin', ax=ax[3])
df2.boxplot('BMI', ax=ax[4])
df2.boxplot('DiabetesPedigreeFunction', ax=ax[5])
df2.boxplot('Age', ax=ax[6])

plt.subplots_adjust(wspace=0.7) 

plt.show()


# In[238]:


df2.sample(100) #random sample of new dataset.


# In[239]:


plt.hist(df2['Outcome'])


# In[16]:


colors = ['red', 'purple']
labels = ['non-diabetic','diabetic']
values = df2['Outcome'].value_counts()/df2['Outcome'].shape[0]

# Use `hole` to create a donut-like pie chart
fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
fig.update_traces(hoverinfo='label+percent', textinfo='percent', textfont_size=20,
                  marker=dict(colors=colors, line=dict(color='#000000', width=2)))
fig.update_layout(
    title_text="Outcome")
fig.show()


# In[241]:


df2.describe()


# In[242]:


#Simple boxplot using pandas
df2['BMI'].plot(kind='box');


# In[17]:


corr = df2.corr()
corr


# In[244]:


sns.heatmap(corr,cmap ='RdBu',vmin=-1, vmax=1, annot = True)


# In[274]:


plt.figure(figsize=(12,4),dpi=200)
sns.scatterplot(y='BMI', x ='SkinThickness', data=df2, hue='Outcome')


# In[273]:


# Pairplot 
sns.pairplot(data = df2, hue = 'Outcome')
plt.show()


# In[24]:


#define the predictor variable and the response variable
x = df2['Glucose']
y = df2['Outcome']


#plot logistic regression curve
sns.regplot(x=x, y=y, data=df2, logistic=True, ci=None)


# In[ ]:




