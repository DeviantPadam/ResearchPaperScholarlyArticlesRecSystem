# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load


 # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Inumpyut data files are available in the read-only "../inumpyut/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the inumpyut directory


# You can write up to 5GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

from sklearn.model_selection import train_test_split
from sklearn.base import TransformerMixin,BaseEstimator,RegressorMixin
from sklearn.preprocessing import MinMaxScaler
from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
import category_encoders as ce
import cloudpickle
import pickle


train = pd.read_csv('data/titanic/train.csv',index_col='PassengerId')
y = train.pop('Survived')

class FeatureSelector(BaseEstimator, TransformerMixin):
    def fit(self,X,y=None):
        return self
    def transform(self,X,y=None):
        train = X.copy()
        train['HasCabin'] = train['Cabin'].where(~train['Cabin'].isna(),0)
        train['HasCabin'] = train['HasCabin'].where(train['Cabin'].isna(),1)
        train['IsAlone'] = ((train['SibSp']+train['Parch'])==0).astype('int')
        train['FamilySize'] = train['SibSp']+train['Parch']
        train['FamilySize'] = train['FamilySize'].astype('object')
        train['Pclass'] = train['Pclass'].astype('object')
        train['Embarked'].fillna('S',inplace=True)
        train['Title'] = pd.DataFrame(((pd.DataFrame((train['Name'].str.split(', ')).tolist(),index=train.index)[1]).str.split(' ')).tolist(),index=train.index)[0]
        train['Title'] = train['Title'].where(train['Title'].isin(['Mr.', 'Mrs.', 'Miss.', 'Master.']),'Others')
        train['AgeGroup'] = pd.cut(train['Age'],[-1,3,16,30,60,float('inf')],labels=['Babies','Children','Young Adults','Middle Aged','Senior Citizen'])
        train['FareGroup'] = pd.cut(train['Fare'],[-1,10,50,100,200,1000],labels=['Low','Mid','UpperMid','High','VeryHigh'])
        train['SexCode']=train['Sex'].where(train['Sex']=='male',0)
        train['SexCode']=train['SexCode'].where(train['Sex']=='female',1)
        final = train[['Pclass','Embarked','HasCabin','IsAlone','Title','AgeGroup','FareGroup','SexCode','FamilySize']]
        return final
    
lgbm = LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=0.7,
                importance_type='split', learning_rate=0.1,
               max_depth=2, min_child_samples=20, min_child_weight=16,
               min_split_gain=0.0, missing=-999, n_estimators=500,n_jobs=1,
               num_leaves=31, objective=None, random_state=64, reg_alpha=0.0,
               reg_lambda=0.0, silent=1, subsample=0.8,
               subsample_for_bin=200000, subsample_freq=0)

pipe = Pipeline([('transformer',FeatureSelector()),('encoder',ce.WOEEncoder()),
                 ('scaler',MinMaxScaler()),('classifier',lgbm)])

pipe.fit(train,y)

cloudpickle.dump(pipe, open('titanicModel.pkl','wb'),protocol=pickle.HIGHEST_PROTOCOL)

model = pickle.load(open('titanicModel.pkl','rb'))


col_dict = {i:[] for i in train.columns}

col_dict['Pclass'].append(3)
col_dict['Name'].append('asda, Mr. Ram')
col_dict['Sex'].append('male')
col_dict['Age'].append(37)
col_dict['SibSp'].append(50)
col_dict['Parch'].append(1231)
col_dict['Ticket'].append('as')
col_dict['Fare'].append(56)
col_dict['Cabin'].append('C97')
col_dict['Embarked'].append('S')

pd.DataFrame(col_dict)

model.predict(pd.DataFrame(col_dict))
print(model.score(train,y))
