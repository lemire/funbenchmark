import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics

file="ani_twocolumn_21june2018.txt"
maindataset=pd.read_csv(file)

dataset=maindataset
predictor=["t11","t12","t13","t14","t15","t21","t22","t23","t24","t25","b11","b12","b13","b14","b15","b21","b22","b23","b24","b25"] #"c1","c2","c3","sel11","sel12","sel13","sel14","sel15","sel21","sel22","sel23","sel24","sel25",


dataset["w11"]=(dataset["sel11"]*(dataset["b11"]+dataset["b12"]+dataset["b13"]+dataset["b14"]+dataset["b15"]))*(dataset["t12"]+dataset["t13"])
dataset["w12"]=(dataset["sel12"]*(dataset["b12"]+dataset["b13"]+dataset["b14"]+dataset["b15"]))*(dataset["t12"]+dataset["t13"])
dataset["w13"]=(dataset["sel13"]*(dataset["b13"]+dataset["b14"]+dataset["b15"]))*(dataset["t12"]+dataset["t13"])
dataset["w14"]=(dataset["sel14"]*(dataset["b14"]+dataset["b15"]))*(dataset["t12"]+dataset["t13"])
dataset["w15"]=(dataset["sel15"]*dataset["b15"])*(dataset["t12"]+dataset["t13"])

dataset["w11"]+=dataset["sel11"]*(dataset["b11"]) *(dataset["t11"] + dataset["t14"])
dataset["w12"]+=dataset["sel12"]*(dataset["b11"]+dataset["b12"]) *(dataset["t11"] + dataset["t14"])
dataset["w13"]+=dataset["sel13"]*(dataset["b11"]+dataset["b12"]+dataset["b13"]) *(dataset["t11"] + dataset["t14"])
dataset["w14"]+=dataset["sel14"]*(dataset["b11"]+dataset["b12"]+dataset["b13"]+dataset["b14"]) *(dataset["t11"] + dataset["t14"])
dataset["w15"]+=dataset["sel15"]*(dataset["b11"]+dataset["b12"]+dataset["b13"]+dataset["b14"]+dataset["b15"]) *(dataset["t11"] + dataset["t14"])


dataset["w21"]=dataset["sel21"]*(dataset["b21"]+dataset["b22"]+dataset["b23"]+dataset["b24"]+dataset["b25"])*(dataset["t22"]+ dataset["t23"])
dataset["w22"]=dataset["sel22"]*(dataset["b22"]+dataset["b23"]+dataset["b24"]+dataset["b25"])*(dataset["t22"]+ dataset["t23"])
dataset["w23"]=dataset["sel23"]*(dataset["b23"]+dataset["b24"]+dataset["b25"])*(dataset["t22"]+ dataset["t23"])
dataset["w24"]=dataset["sel24"]*(dataset["b24"]+dataset["b25"])*(dataset["t22"]+ dataset["t23"])
dataset["w25"]=dataset["sel25"]*dataset["b25"]*(dataset["t22"]+ dataset["t23"])

dataset["w21"]+=dataset["sel21"]*(dataset["b21"])*(dataset["t21"]+dataset["t24"])
dataset["w22"]+=dataset["sel22"]*(dataset["b21"]+dataset["b22"])*(dataset["t21"]+dataset["t24"])
dataset["w23"]+=dataset["sel23"]*(dataset["b21"]+dataset["b22"]+dataset["b23"])*(dataset["t21"]+dataset["t24"])
dataset["w24"]+=dataset["sel24"]*(dataset["b21"]+dataset["b22"]+dataset["b23"]+dataset["b24"])*(dataset["t21"]+dataset["t24"])
dataset["w25"]+=dataset["sel25"]*(dataset["b21"]+dataset["b22"]+dataset["b23"]+dataset["b24"]+dataset["b25"])*(dataset["t21"]+dataset["t24"])


predictor=["t11","t12","t13","t14","t15","t21","t22","t23","t24","t25","w11","w12","w13","w14","w15","w21","w22","w23","w24","w25"]
lp = len(predictor)
for i in range(len(predictor)):
    for j in range(i + 1):
        dataset[predictor[i]+predictor[j]]=dataset[predictor[i]]*dataset[predictor[j]]
        predictor.append(predictor[i]+predictor[j])
X=dataset[predictor]
y = dataset[["y"]].iloc[:,0]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

regressor = LinearRegression()
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
