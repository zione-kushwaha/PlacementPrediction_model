#machine learning first project training the model
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# load the data
data = pd.read_csv('placement.csv');
print(data);

# split the data into independent 'X' and dependent 'Y' variables
X = data.iloc[:,1 :-1].values
Y = data.iloc[:, -1].values

# split the data into 90% training and 10% testing

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=0);

# training the model using sklearn logistic regration

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression();

clf.fit(X_train,Y_train);

u=clf.predict(X_test);

print(u);
print(Y_test)
from sklearn.metrics import accuracy_score

print(accuracy_score(Y_test,u));


# way for seeing the decision boundary of the model

from mlxtend.plotting import plot_decision_regions;
plot_decision_regions(X_train,Y_train,clf=clf,legend=2);
plt.show();


# for creating file for this trained model 
import pickle
pickle.dump(clf,open('model.pkl','wb'));