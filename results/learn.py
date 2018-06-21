import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.preprocessing import normalize

from sklearn import metrics

datafiles = ["skylake/vecbench.txt", "armA57/vecbench.txt"]



blah=[]
predictor = ["U", "U_2", "N1", "N1_2", "N2", "N2_2", "N1N2"]

for file in datafiles:
  print("file: ", file)
  dataset = pd.read_csv(file, delimiter=" ")
  dataset.columns=["U","N1", "N2", "card", "time", "garbage"]
  dataset["N1_2"]=dataset["N1"]**2
  dataset["N2_2"]=dataset["N2"]**2
  dataset["U_2"]=dataset["U"]**2
  dataset["N1N2"]=dataset["N1"]*dataset["N2"]
  #X=dataset.iloc[:,0:3]
  X=dataset[predictor]
  y=dataset.iloc[:,4]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  regressor =  LinearRegression(normalize=True)

  regressor.fit(X_train, y_train)
  y_pred = regressor.predict(X_test)
  print("intercept: ", regressor.intercept_, " coefs ", regressor.coef_)
  df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('relative Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)/y_test.mean())
  f = plt.figure()
  plt.title(file+" (quadratic)")
  plt.scatter(y_test, y_pred)
  plt.scatter(y_test, y_test)
  plt.ylim([y.min(), y.max()])
  plt.xlim([y.min(), y.max()])
  plt.ylabel('predicted running time')
  plt.xlabel('actual running time')
  filename = file+"quadratic.pdf"
  if(os.path.exists(filename)):
    os.remove(filename)
  f.savefig(filename, bbox_inches='tight')
  print(filename)
  filename = file+"quadratic.png"
  if(os.path.exists(filename)):
    os.remove(filename)
  f.savefig(filename, bbox_inches='tight')
  print(filename)

  print()
  print()
  plt.close()

  f = plt.figure()

  y_pos = np.arange(len(predictor))
  norm1 = regressor.coef_ / np.linalg.norm(regressor.coef_)
  blah.append(norm1)
  print(norm1)

  plt.bar(y_pos, norm1, align='center', alpha=0.9)
  plt.xticks(y_pos, predictor, rotation='vertical')
  plt.ylabel('Weigth')
  plt.title('Feature weights')


  f.savefig(file+"quadraticweight.pdf", bbox_inches='tight')

  f.savefig(file+"quadraticweight.png", bbox_inches='tight')
  plt.close()

f = plt.figure()

y_pos = np.arange(len(predictor))
n = blah[1]/blah[0]

plt.bar(y_pos, n, align='center', alpha=0.9)
plt.xticks(y_pos, predictor, rotation='vertical')
plt.ylabel('ratios')
plt.title('Weigth ratios (ARM/Skylake)')


f.savefig("quadraticweightratios.pdf", bbox_inches='tight')

f.savefig("quadraticweightratios.png", bbox_inches='tight')
plt.close()
