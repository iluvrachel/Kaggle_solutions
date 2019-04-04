import pandas as pd
import numpy as np 
import random 
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train_df = train_df.drop(['Name','Ticket','Cabin','SibSp','Parch','Embarked','Age','PassengerId'],axis = 1)
test_df = test_df.drop(['Name','Ticket','Cabin','SibSp','Parch','Embarked','Age'],axis = 1)

# Fill the loss Fare value in the test data with median
test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

combine = [train_df,test_df]

for dataset in combine:
    dataset.loc[ dataset['Fare'] <= 15, 'Fare'] = 0
    dataset.loc[ dataset['Fare'] > 15, 'Fare'] = 1
    dataset['Fare'] = dataset['Fare'].astype(int)
    dataset['Sex'] = dataset['Sex'].map({'female':1,'male':0}).astype(int)

combine = [train_df, test_df]

X_train = train_df.drop('Survived', axis=1).values
Y_train = train_df['Survived'].values
X_test  = test_df.drop('PassengerId', axis=1).copy()

X_train, Y_train = shuffle(X_train, Y_train)

rf = RandomForestClassifier()
rf_params = {'n_estimators': [100, 200, 300, 400, 500, 1000]}
rf_grid = GridSearchCV(rf, rf_params, scoring='neg_log_loss', refit=True)
rf_grid.fit(X_train, Y_train)

Y_pred = rf_grid.predict(X_test)
rf_grid.score(X_train, Y_train)
acc_random_forest = round(rf_grid.score(X_train, Y_train) * 100, 2)
print(acc_random_forest)

submission = pd.DataFrame({
        "PassengerId": test_df["PassengerId"],
        "Survived": Y_pred
    })

submission.to_csv('submission.csv', index=False)
