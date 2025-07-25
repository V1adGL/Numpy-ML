import pandas as pd
import numpy as np
data = pd.read_csv('./spam.csv')

print(data)

data.info()

data['spam'] = data['Category'].apply(lambda x: 1 if x == 'spam' else 0)
# print(data.head(3))

from sklearn.feature_extraction.text import CountVectorizer

vect = CountVectorizer()
X = vect.fit_transform(data['Message'])

w = vect.get_feature_names_out()

# print(X[:, 1000])


from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import GaussianNB

from sklearn.pipeline import Pipeline

model = Pipeline([('vect', CountVectorizer()), ('NB', MultinomialNB())])
# model = Pipeline([('vect', CountVectorizer()), ('NB', GaussianNB())])


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    data['Message'],
    data['spam'],
    test_size=0.3
)

model.fit(X_train, y_train)
# model.fit(X_train, y_train)

y_predict = model.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_predict, y_test))

msg = [
    'Hi! How are you?',
    'Free subscription',
    'You won the lottery. Call us'
]

print(model.predict(msg))