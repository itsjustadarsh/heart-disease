#!/usr/bin/env python
# coding: utf-8

# In[7]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[8]:


# heart_df = pd.read_csv("heart.csv")  # Uncomment if heart.csv is available
heart_df


# In[9]:


heart_df.sample(5)


# In[10]:


heart_df.info()


# In[11]:


heart_df.describe()


# In[12]:


heart_df.describe(include="all")


# In[13]:


heart_df.isnull().sum()


# In[14]:


heart_df.duplicated().sum()


# In[15]:


heart_df.nunique()


# In[16]:


cat_col=heart_df.select_dtypes(include='object').columns


# In[17]:


heart_df['ChestPainType'].unique()


# In[18]:


range(heart_df['ChestPainType'].nunique())


# In[19]:


for col in cat_col:
    print(col)
    print((heart_df[col].unique()), list(range(heart_df[col].nunique())))
    heart_df[col].replace((heart_df[col].unique()), range(heart_df[col].nunique()), inplace=True)
    print('*'*90)
    print()


# In[20]:


heart_df


# In[21]:


heart_df['Cholesterol'].value_counts()


# In[22]:


heart_df['Cholesterol'].replace(0,np.nan, inplace=True)


# In[23]:


from sklearn.impute import KNNImputer
imputer= KNNImputer(n_neighbors=3)
after_impute=imputer.fit_transform(heart_df)
heart_df=pd.DataFrame(after_impute, columns=heart_df.columns)


# In[24]:


heart_df['Cholesterol'].isna().sum()


# In[25]:


count=0
for i in heart_df['Cholesterol']:
  if i==0:
    count+=1
print(count)


# In[26]:


heart_df['RestingBP'][heart_df['RestingBP']==0]


# In[27]:


from sklearn.impute import KNNImputer
heart_df['RestingBP'].replace(0, np.nan, inplace=True)
imputer = KNNImputer(n_neighbors=3)
after_impute = imputer.fit_transform(heart_df)
heart_df = pd.DataFrame(after_impute, columns=heart_df.columns)


# In[28]:


heart_df['RestingBP'].unique()


# In[29]:


heart_df['RestingBP'].isnull().sum()


# In[30]:


withoutOldPeak=heart_df.columns
withoutOldPeak=withoutOldPeak.drop('Oldpeak')
heart_df[withoutOldPeak]=heart_df[withoutOldPeak].astype('int32')


# In[31]:


heart_df.info()


# In[32]:


# pip install plotly  # Install plotly separately if needed


# In[33]:


heart_df.sample()


# In[34]:


heart_df.corr()['HeartDisease'][:-1].sort_values()


# In[35]:


import plotly.express as px


# In[36]:


px.line(heart_df.corr()['HeartDisease'][:-1].sort_values())


# In[37]:


px.sunburst(heart_df, path=['HeartDisease','Age'])


# In[38]:


px.histogram(heart_df,x='Age',color='HeartDisease')


# In[39]:


px.pie(heart_df,names='HeartDisease',title='Percentage of HeartDisease classes distribution')


# In[40]:


px.histogram(heart_df,x='Sex',color='HeartDisease')


# In[41]:


px.histogram(heart_df, x='ChestPainType',color='HeartDisease')


# In[42]:


heart_df['RestingBP'].unique()


# In[43]:


px.sunburst(heart_df,path=['HeartDisease','RestingBP'])


# In[44]:


px.histogram(heart_df,x='FastingBS',color='HeartDisease')


# In[45]:


px.sunburst(heart_df,path=['HeartDisease','MaxHR'])


# In[46]:


px.violin(heart_df, x='HeartDisease', y='MaxHR', color='HeartDisease')


# In[47]:


px.violin(heart_df, x='HeartDisease', y='Oldpeak', color='HeartDisease')


# In[48]:


px.histogram(heart_df, x='ST_Slope', color='HeartDisease')


# In[49]:


px.histogram(heart_df, x='ExerciseAngina', color='HeartDisease')


# In[50]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    heart_df.drop('HeartDisease', axis=1),
    heart_df['HeartDisease'],
    test_size=0.2,
    random_state=42,
    stratify=heart_df['HeartDisease']
)


# In[51]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

solver = ['lbfgs', 'liblinear', 'newton-cg', 'newton-cholesky', 'sag', 'saga']
best_solver = ''
test_score = np.zeros(6)
for i, n in enumerate(solver):
    lr = LogisticRegression(solver=n).fit(X_train, y_train)
    test_score[i] = lr.score(X_test, y_test)
    if lr.score(X_test, y_test) == test_score.max():
        best_solver = n

print(best_solver)
lr = LogisticRegression(solver=best_solver)
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
print(f'LogisticRegression Score: {accuracy_score(y_test, lr_pred)}')


# In[52]:


import pickle
file=open('LogisticR.pkl','wb')
pickle.dump(lr,file)


# In[53]:


from sklearn.svm import SVC
from sklearn.metrics import f1_score

kernels = {'linear':0, 'poly':0, 'rbf':0, 'sigmoid':0}
best = ''
for i in kernels:
    svm = SVC(kernel=i)
    svm.fit(X_train, y_train)
    yhat = svm.predict(X_test)
    kernels[i] = f1_score(y_test, yhat, average="weighted")
    if kernels[i] == max(kernels.values()):
        best = i

print(best)
svm = SVC(kernel=best)
svm.fit(X_train, y_train)
svm_pred = svm.predict(X_test)
print(f'SVM f1_score kernel({best}): {f1_score(y_test, svm_pred, average="weighted")}')


# In[54]:


import pickle
file = open('SVM_Model.pkl', 'wb')
pickle.dump(svm, file)


# In[55]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

dtree = DecisionTreeClassifier(class_weight='balanced')
param_grid = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_samples_split': [2, 3, 4],
    'min_samples_leaf': [1, 2, 3, 4],
    'random_state': [0, 42]
}

grid_search = GridSearchCV(dtree, param_grid, cv=5)
grid_search.fit(X_train, y_train)

Ctree = DecisionTreeClassifier(**grid_search.best_params_, class_weight='balanced')
Ctree.fit(X_train, y_train)
dtc_pred = Ctree.predict(X_test)
print("DecisionTree's Accuracy: ", accuracy_score(y_test, dtc_pred))


# In[56]:


import pickle
file = open('DecisionTree.pkl', 'wb')
pickle.dump(Ctree, file)


# In[57]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

rfc = RandomForestClassifier()
param_grid = {
    'n_estimators': [50, 100, 150, 500],
    'max_features': ['sqrt', 'log2', None],
    'max_depth': [3, 6, 9, 19],
    'max_leaf_nodes': [3, 6, 9]
}

grid_search = GridSearchCV(rfc, param_grid, cv=5)
grid_search.fit(X_train, y_train)

rfctree = RandomForestClassifier(**grid_search.best_params_)
rfctree.fit(X_train, y_train)
rfc_pred = rfctree.predict(X_test)
print("RandomForestClassifier's Accuracy: ", accuracy_score(y_test, rfc_pred))


# In[58]:


import pickle
file = open('RandomForest.pkl', 'wb')
pickle.dump(rfctree, file)


# In[ ]:


# Model files saved successfully:
# - RandomForest.pkl
# - DecisionTree.pkl  
# - SVM_Model.pkl
# - LogisticR.pkl

