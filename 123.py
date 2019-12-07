import numpy as np
import pandas as pd
#可視化
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
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
import warnings
warnings.filterwarnings('ignore')


train = pd.read_csv('/Users/bella/titanic/train.csv')
test = pd.read_csv('/Users/bella/titanic/test.csv')
combine = pd.concat([train,test])

train.groupby(train.Name.apply(lambda x: len(x)))['Survived'].mean()
combine['Name_Len'] = combine['Name'].apply(lambda x :len(x))
combine['Name_Len'] = pd.qcut(combine['Name_Len'], 5)
combine.groupby(combine['Name'].apply(lambda x : x.split(',')[1]).apply(lambda x: x.split('.')[0]))['Survived'].mean()

combine['Title'] = combine['Name'].apply(lambda x : x.split(',')[1]).apply(lambda x : x.split('.')[0])
combine['Title'] = combine['Name'].replace(['Don','Dona' ,'Majoy', 'Capt', 'Jonkheer', 'Rev', 'Col','Sir'], 'Mr')
combine['Title'] = combine['Name'].replace(['Mlle','Ms'], 'Miss')
combine['Title'] = combine['Name'].replace(['the Countess','Mme','Lady','Dr'], 'Mrs')
df = pd.get_dummies(combine['Title'],prefix='Title')
combine = pd.concat([combine, df], axis=1)

combine['Fname'] = combine['Name'].apply(lambda x:x.split(',')[0])
combine['FamilySize'] = combine['SibSp'] + combine['Parch']
dead_female_Fname = list(set(combine[(combine.Sex=='female') & (combine.Age>=12)
                        & (combine.Survived==0) & (combine.FamilySize>1)]['Fname'].values))
alive_male_Fname = list(set(combine[(combine.Sex=='male') & (combine.Age>=12) 
                        & (combine.Survived==1) & (combine.FamilySize>1)]['Fname'].values))
combine['Female_Dead_Family'] = np.where(combine['Fname'].isin(dead_female_Fname),1,0)
combine['Male_Alive_Family'] = np.where(combine['Fname'].isin(alive_male_Fname),1,0)
combine = combine.drop(['Name', 'Fname'], axis=1)

group = combine.groupby(['Title', 'Pclass'])['Age']
combine['Age'] = group.transform(lambda x : x.fillna(x.median()))
combine = combine.drop('Title', axis=1)

combine['Child'] = np.where(combine['Age'] <=12 ,1,0)
combine['Age'] = pd.cut(combine['Age'], 5)
combine = combine.drop('Age', axis=1)

df = pd.get_dummies(combine['FamilySize'],prefix='FamilySize')
combine = pd.concat([combine ,df],axis=1)
combine = combine.drop(['SibSp', 'Parch', 'FamilySize'],axis=1)

combine['Ticket_Lett'] = combine['Ticket'].apply(lambda x : str(x)[0])
combine['Ticket_Lett'] = combine['Ticket_Lett'].apply(lambda x :str(x))
combine['High_Survival_Ticket'] = np.where(combine['Ticket_Lett'].isin(['1', '2', 'P']),1,0)
combine['Low_Survival_Ticket'] = np.where(combine['Ticket_Lett'].isin(['A', 'W', '3', '7']),1,0)
combine = combine.drop(['Ticket', 'Ticket_Lett'], axis=1)