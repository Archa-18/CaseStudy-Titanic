import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

data = pd.read_csv('tested.csv')

data = data.drop(['PassengerId', 'Cabin', 'Name', 'Ticket'], axis=1)

data['Fare'].fillna(data['Fare'].mean(), inplace=True)

le = LabelEncoder()
le.fit(data['Sex'])
data['Sex'] = le.transform(data['Sex'])

le = LabelEncoder()
le.fit(data['Embarked'])
data['Embarked'] = le.transform(data['Embarked'])
data['Age'].fillna(data['Age'].median(), inplace=True)
data1 = data.drop('Survived', axis=1)
model = ExtraTreesClassifier()
model.fit(data1, data['Survived'])

feature_importance = pd.Series(model.feature_importances_, index=data1.columns)
feature_importance.nlargest(8).plot(kind='barh')
#plt.show()

data=data.drop(['Age', 'Embarked', 'Pclass'], axis=1)

x = data.drop('Survived', axis=1)
y = data['Survived']

data['SibSp'] = pd.cut(data['SibSp'], 2, labels=[0,1])

nb=MultinomialNB()

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, train_size=0.2)

nb.fit(x_train, y_train)
y_pred = nb.predict(x_test)
print('NaiveBayes', accuracy_score(y_test, y_pred))

#NaiveBayes 0.8059701492537313