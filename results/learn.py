import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

for file in ["skylake/vecbench.txt", "skylake/hashbench.txt", "armA57/vecbench.txt", "armA57/hashbench.txt"]:
  print("file: ", file)
  dataset = pd.read_csv(file, delimiter=" ")
  X=dataset.iloc[:,1:4]
  y=dataset.iloc[:,4]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  regressor = LinearRegression()
  regressor.fit(X_train, y_train)

  print("intercept: ", regressor.intercept_, " coefs ", regressor.coef_)

  y_pred = regressor.predict(X_test)
  df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('relative Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)/y_test.mean())

  #dataset.plot(x='Hours', y='Scores', style='o')
  plt.title(file)
  f = plt.figure()
  plt.ylim([y.min(), y.max()])
  plt.xlim([y.min(), y.max()])
  plt.scatter(y_test, y_pred)
  plt.scatter(y_test, y_test)

  plt.ylabel('predicted running time')
  plt.xlabel('actual running time')
  filename = file+".pdf"
  if(os.path.exists(filename)):
    os.remove(filename)
  f.savefig(filename, bbox_inches='tight')
  print(filename)
  print()


for file in ["skylake/vecbench.txt", "skylake/hashbench.txt", "armA57/vecbench.txt", "armA57/hashbench.txt"]:
  print("file: ", file)
  dataset = pd.read_csv(file, delimiter=" ")
  X=dataset.iloc[:,2:4]
  y=dataset.iloc[:,4]
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

  regressor = LinearRegression()
  regressor.fit(X_train, y_train)

  print("intercept: ", regressor.intercept_, " coefs ", regressor.coef_)

  y_pred = regressor.predict(X_test)
  df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
  print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))
  print('relative Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred)/y_test.mean())
  f = plt.figure()
  plt.title(file + " (No Max)")
  plt.scatter(y_test, y_pred)
  plt.scatter(y_test, y_test)
  plt.ylim([y.min(), y.max()])
  plt.xlim([y.min(), y.max()])
  plt.ylabel('predicted running time')
  plt.xlabel('actual running time')
  filename = file+"NoMax.pdf"
  if(os.path.exists(filename)):
    os.remove(filename)
  f.savefig(filename, bbox_inches='tight')
  print(filename)
  print()
  print()
