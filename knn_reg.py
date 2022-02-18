import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split 
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error,mean_absolute_error,r2_score


df = pd.read_csv("/data/Housing.csv")
dum_df = pd.get_dummies(df, drop_first=True)

X = dum_df.iloc[:,:-1]
y = pd.DataFrame(dum_df.iloc[:,-1])
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.3, 
                                                    random_state=2021)

scalerX = StandardScaler()
Xscaled = scalerX.fit_transform(X_train)
Xscaled = pd.DataFrame(Xscaled,columns=X_train.columns)

scalery = StandardScaler()
yscaled = scalery.fit_transform(y_train)

Xscaled_test = scalerX.transform(X_test)
yscaled_test = scalery.transform(y_test)

knn = KNeighborsRegressor(n_neighbors=5)
knn.fit( Xscaled , yscaled )
y_pred = knn.predict(Xscaled_test)
r2_score(yscaled_test, y_pred)

y_pred = pd.DataFrame(y_pred)
y_test = pd.DataFrame(yscaled_test, columns = ['true'])
y_test['predication'] = y_pred

y_test.to_csv('/output/Result.csv',index = False)

