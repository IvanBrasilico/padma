"""Treinar e avaliar o modelo a definir em models/peso/peso.py.

"""
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

from treinador_pesagem import load_histograms, evaluate

params = {'n_estimators': 500, 'max_depth': 5, 'min_samples_split': 5,
          'learning_rate': 0.02, 'loss': 'ls'}

grid = [
    ('Linear Ridge', linear_model.RidgeCV(alphas=[0.1, 1.0, 10.0])),
    ('Linear ', linear_model.LinearRegression()),
    ('Random Forest', RandomForestRegressor()),
    ('Gradient boosting',  GradientBoostingRegressor(**params))
]

if __name__ == '__main__':
    for sufix in ['', 'refined']:
        histograms, labels = load_histograms(sufix)
        print()
        print('Média, min, max')
        print(labels.mean(), labels.min(), labels.max())
        cont = len(histograms)
        print('número de exemplos', cont)
        train = (cont // 5 + 1) * 4
        test = (cont // 5 + 1)
        print('train', train, 'test', test)
        for tuple in grid:
            print(tuple[0])
            reg = tuple[1]
            labels_train = labels[:train]
            labels_test = labels[-test:]
            reg.fit(histograms[:train], labels[:train])
            labels_predicted_train = reg.predict(histograms[:train])
            labels_predicted = reg.predict(histograms[-test:])
            evaluate(labels_train, labels_predicted_train,
                     labels_test, labels_predicted)
