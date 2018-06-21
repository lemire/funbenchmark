import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn import metrics

file="ani_21june2018_csv.txt"
maindataset=pd.read_csv(file)

dataset=maindataset
predictor = ["sel1","sel2","sel3","sel4","sel5","t1","t2","t3","t4","t5","b1","b2","b3","b4","b5"]
X=dataset[predictor]
y = dataset[["y"]].iloc[:,0]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor =  LinearRegression(normalize=True)
regressor.fit(X_train, y_train)
print(regressor.coef_)

y_pred = regressor.predict(X_test)

df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(df[['Actual','Predicted']])
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
print('relative Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)/y_test.mean())
f = plt.figure()
plt.scatter(y_test, y_pred)
plt.scatter(y_test, y_test)
plt.ylim([y.min(), y.max()])
plt.xlim([y.min(), y.max()])
plt.ylabel('predicted selectivity')
plt.xlabel('actual selectivity')
filename = file+".pdf"
if(os.path.exists(filename)):
  os.remove(filename)
f.savefig(filename, bbox_inches='tight')
print(filename)
filename = file+".png"
if(os.path.exists(filename)):
  os.remove(filename)
f.savefig(filename, bbox_inches='tight')
print(filename)
plt.close()

f = plt.figure()

y_pos = np.arange(len(predictor))

plt.bar(y_pos, regressor.coef_, align='center', alpha=0.9)
plt.xticks(y_pos, predictor, rotation='vertical')
plt.ylabel('Weigth')
plt.title('Feature weights')
f.savefig(file+"weight.pdf", bbox_inches='tight')

f.savefig(file+"weight.png", bbox_inches='tight')
plt.close()
