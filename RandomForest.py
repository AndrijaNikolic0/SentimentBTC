import pandas as pd
import numpy as np
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn import ensemble
from sklearn import tree
from sklearn import preprocessing
from sklearn import utils
from sklearn.preprocessing import LabelEncoder
from sklearn.base import ClassifierMixin
from sklearn.base import RegressorMixin
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

#prvo skipljamo podatke iz datascraping(main), zatim iz Ursina(vader) vadimo sentiment, i sve spajamo u DataScraping(spajanje) pa, dodaje ruske tu par kolona i od
#toga nastavlje ovde sa primenom RandomForest modela.
df = pd.read_csv(r'C:\Users\Andrija\Downloads\sentimentIBtcFinalno.csv', header=0)

cols_to_drop = ['Date','open_btc','high_btc','low_btc','close_btc','open_sol','high_sol','low_sol','close_sol','open_doge','high_doge','low_doge','close_doge','Broj redova','Polarity','Avg','Prazna kolona']
df = df.drop(cols_to_drop, axis=1)

sb.heatmap(df.isnull())
df['Mavg'] = df['Mavg'].interpolate()


df = df.dropna()

labelencoder = LabelEncoder()

df['Buduci Trend'] = labelencoder.fit_transform(df['Buduci Trend'])

X = df.values
y = df['Buduci Trend'].values

X = np.delete(X,2,axis=1)


X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=0)




rf_clf = ensemble.RandomForestClassifier(n_estimators=100)
rf_clf.fit(X_train, y_train)
print(rf_clf.score(X_test, y_test))
