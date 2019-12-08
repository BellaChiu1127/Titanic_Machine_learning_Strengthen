#!/usr/bin/env python
# coding: utf-8

# In[49]:

import csv
import numpy as np
import pandas as pd
#可視化
#from IPython import get_ipython
import seaborn as sns
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
#model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
import scipy
warnings.filterwarnings('ignore')


# In[8]:


train = pd.read_csv('/Users/bella/titanic/train.csv')
test = pd.read_csv('/Users/bella/titanic/test.csv')
combine = pd.concat([train,test])


# In[9]:


#train.groupby(train.Name.apply(lambda x: len(x)))['Survived'].mean()


# In[10]:





# In[11]:


#combine.groupby(combine['Name'].apply(lambda x : x.split(',')[1]).apply(lambda x: x.split('.')[0]))['Survived'].mean()


# In[12]:


combine['Title'] = combine['Name'].apply(lambda x : x.split(', ')[1]).apply(lambda x : x.split('.')[0])
combine['Title'] = combine['Title'].replace(['Don','Dona', 'Major', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir','Dr'],'Mr')
combine['Title'] = combine['Title'].replace(['Mlle','Ms'], 'Miss')
combine['Title'] = combine['Title'].replace(['the Countess','Mme','Lady','Dr'], 'Mrs')
df = pd.get_dummies(combine['Title'],prefix='Title')
combine = pd.concat([combine, df], axis=1)

combine['Name_Len'] = combine['Name'].apply(lambda x :len(x))
combine['Name_Len'] = pd.qcut(combine['Name_Len'], 5)

# In[13]:


combine['Fname'] = combine['Name'].apply(lambda x:x.split(',')[0])

dead_female_Fname = list(set(combine[(combine.Sex=='female') & (combine.Age>=12)
                        & (combine.Survived==0) & ((combine.Parch>0) | (combine.SibSp>0))]['Fname'].values))
alive_male_Fname = list(set(combine[(combine.Sex=='male') & (combine.Age>=12) 
                        & (combine.Survived==1) & ((combine.Parch>0) | (combine.SibSp>0))]['Fname'].values))
combine['Female_Dead_Family'] = np.where(combine['Fname'].isin(dead_female_Fname),1,0)
combine['Male_Alive_Family'] = np.where(combine['Fname'].isin(alive_male_Fname),1,0)
combine = combine.drop(['Name', 'Fname'], axis=1)


# In[14]:


group = combine.groupby(['Title', 'Pclass'])['Age']
combine['Age'] = group.transform(lambda x : x.fillna(x.median()))
combine = combine.drop('Title', axis=1)

combine['Child'] = np.where(combine['Age'] <=12 ,1,0)
combine['Age'] = pd.cut(combine['Age'], 5)
combine = combine.drop('Age', axis=1)


# In[15]:

combine['FamilySize'] = np.where(combine['SibSp'] + combine['Parch'] == 0 ,'Solo', np.where(combine['SibSp'] + combine['Parch'] <= 3, 'Small' , 'Big'))
df = pd.get_dummies(combine['FamilySize'],prefix='FamilySize')
combine = pd.concat([combine ,df],axis=1).drop(['SibSp', 'Parch', 'FamilySize'],axis=1)


# In[16]:


combine['Ticket_Lett'] = combine['Ticket'].apply(lambda x : str(x)[0])
combine['Ticket_Lett'] = combine['Ticket_Lett'].apply(lambda x :str(x))
combine['High_Survival_Ticket'] = np.where(combine['Ticket_Lett'].isin(['1', '2', 'P']),1,0)
combine['Low_Survival_Ticket'] = np.where(combine['Ticket_Lett'].isin(['A', 'W', '3', '7']),1,0)
combine = combine.drop(['Ticket', 'Ticket_Lett'], axis=1)


# In[17]:


combine.Embarked = combine.Embarked.fillna('S')
df = pd.get_dummies(combine['Embarked'],prefix='Embarked')
combine = pd.concat([combine, df],axis=1)
combine = combine.drop('Embarked', axis=1)


# In[18]:


combine['Cabin_isNull'] = np.where(combine['Cabin'].isnull(),0,1)
combine = combine.drop('Cabin', axis=1)


# In[19]:


df = pd.get_dummies(combine['Pclass'],prefix='Pclass')
combine = pd.concat([combine, df], axis=1).drop('Pclass',axis=1)


# In[20]:


df = pd.get_dummies(combine['Sex'], prefix='Pclass')
combine = pd.concat([combine, df], axis=1).drop('Sex',axis=1)


# In[21]:
combine['Fare'].fillna(combine['Fare'].dropna().median(), inplace=True)
combine['Low_Fare'] = np.where(combine['Fare']<=8.662,1,0)
combine['High_Fare'] = np.where(combine['Fare']>=26,1,0)
combine = combine.drop('Fare',axis=1)
# In[22]:


features = combine.drop(['PassengerId', 'Survived'],axis=1).columns
le = LabelEncoder()
for feature in features:
    le = le.fit(combine[feature])
    combine[feature] = le.transform(combine[feature])


# In[23]:


X_all = combine.iloc[:891, :].drop(['PassengerId', 'Survived'], axis=1)
Y_all = combine.iloc[:891, :]['Survived']
X_test = combine.iloc[891:, :].drop(['PassengerId', 'Survived'], axis=1)

# In[44]:


lg = LogisticRegression()
svc = SVC()
knn = KNeighborsClassifier(n_neighbors = 3)
dt = DecisionTreeClassifier()
rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=4, class_weight={0:0.745 ,1:0.255})
gbdt = GradientBoostingClassifier(n_estimators=500, learning_rate=0.06, max_depth=3)
xgb = XGBClassifier(max_depth=3, n_estimators=300, learning_rate=0.03)
lgb = LGBMClassifier(max_depth=3, n_estimators=500, learning_rate=0.04)
clfs = [lg, svc, gbdt, xgb, lgb]

kfold = 10
cv_results = []
for classifier in clfs :
    cv_results.append(cross_val_score(classifier, X_all, y = Y_all, scoring= "accuracy", cv = kfold , n_jobs=7))

 
x = ["LG","SVC","GBDT","XGB","LGB"]
cv_means = []
cv_stds = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_stds.append(cv_result.std())

#plt.bar(x, cv_means, label=x)
#plt.ylim(0.85,0.95)
#plt.show()
#print(cv_means)

#=====================集成框架=======================
class Ensemble(object):
    def __init__(self, estimators):
        self.estimators_names = []
        self.estimators = []
        for i in estimators:
            self.estimators_names.append(i[0])
            self.estimators.append(i[1])

        self.clf = LogisticRegression()

    def fit(self, train_x, train_y):
        for i in self.estimators:
            i.fit(train_x, train_y)
        x = np.array([i.predict(train_x) for i in self.estimators]).T
        y = train_y
        self.clf.fit(x, y)

    def predict(self, x):
        x = np.array([i.predict(x) for i in self.estimators]).T
        print(x)
        return self.clf.predict(x)

    def score(self, x, y):
        s = precision_score(y, self.predict(x))
        return s

#===================================================

bag = Ensemble([ ('lgb',lgb), ('xgb', xgb),('gbdt',gbdt),('lg',lg) ])
score = 0
for i in range(0,10):
    num_test = 0.20
    X_train, X_cv, Y_train, Y_cv = train_test_split(X_all, Y_all, test_size=num_test)
    bag.fit(X_train, Y_train)
    #Y_test = bag.predict(X_test)
    acc_xgb = round(bag.score(X_cv, Y_cv)*100, 2)
    score+=acc_xgb
bag.fit(X_all.values, Y_all.values) 
Y_test = bag.predict(X_test.values).astype(int)
submission = pd.DataFrame({'PassengerId': test['PassengerId'],'Survived':Y_test})
submission.to_csv(r'submi.csv', index=False)
print(score/10)
