# text_classifier.py

'''
The MIT License (MIT)
Copyright (C) 2022 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import sys
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import ShuffleSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline

from calc_class import calc, my_cpu_count

def create_dataset(filename):
    calc_all = calc(filename, subset='all')

    X_id  = np.asarray( [txt for txt in calc_all.df.ID] )
    X_voc = np.asarray( [txt.strip() for txt in calc_all.df.VOC_MORPHEME_S] )
    X_base = np.asarray( [item for item in zip( X_id, X_voc)] )
    X = X_base

    y_class = np.asarray(calc_all.df.CORRECT)
    lb = LabelEncoder()
    lb.fit(y_class)
    y = lb.fit_transform(y_class)
 
    return (X, y)

class create_classifier():
    def __init__(self, train_file):
        (X, y) = create_dataset(train_file)

        rs = ShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
        for train_index, test_index in rs.split(X):
            train_X, train_y = X[train_index, 1], y[train_index]
            test_X , test_y  = X[test_index , 1], y[test_index]

        pipeline = Pipeline(steps=[
            ('tfidf', TfidfVectorizer(use_idf=True, ngram_range=(1, 1))),
            ('best' , TruncatedSVD(n_components=200)),
            ('svc'  , SVC(kernel='rbf', C=0.8, gamma='scale', random_state=42)),
        ], verbose=True, )

        self.pipeline = pipeline

        pipeline.fit(train_X, train_y)

        pred_y = pipeline.predict(test_X)
        print('correct: test_y=\n{}'.format(test_y))
        print('predict: pred_y=\n{}'.format(pred_y))

        accuracy = np.mean(pred_y == test_y)
        print('accuracy={0}\n'.format(accuracy))
        print('Classification report:\n\n{}'.format(classification_report(test_y, pred_y)))

    def predict(self, X):
        pred_y = self.pipeline.predict(X)
        return pred_y

train_file='dataset_calc-review.xlsx'
cls = create_classifier(train_file)

def predict_target(classifier, target_file):
    (target_X, target_y) = create_dataset(target_file)

    target_X_str = target_X[:, 1]
    print('      target_X_str=\n{0}\n{1}'.format(target_X_str, target_X_str.shape))

    pred_y = classifier.predict(target_X_str)
    print('correct: target_y=\n{0} {1}'.format(target_y, target_y.shape))
    print('predict: pred_y=\n{0} {1}'.format(pred_y, pred_y.shape))

    accuracy = np.mean(pred_y == target_y)
    print('accuracy={0}\n'.format(accuracy))
    print('Classification report:\n\n{}'.format(classification_report(target_y, pred_y)))

    def write_result(filename, target_X, target_y, pred_y):
        target_X_y = np.concatenate((target_X, np.reshape(target_y,(-1,1))), axis=1)

        target_X_y_pred_y = np.concatenate((target_X, np.reshape(target_y,(-1,1)), np.reshape(pred_y,(-1,1)),), axis=1)
        target_X_y_pred_y_df = pd.DataFrame(data = target_X_y_pred_y, columns=['ID', 'VOC_MORPHEME_S', 'TARGET', 'PRED'])
        target_X_y_pred_y_df.drop(columns=['VOC_MORPHEME_S'], axis='columns', inplace=True)
        target_X_y_pred_y_df.to_excel(filename, sheet_name='Sheet1')

    write_result(
        os.path.splitext(target_file)[0] + '-predict.xlsx',
        target_X,
        target_y,
        pred_y,
    )

target_file='dataset_calc-target.xlsx'
result = predict_target(cls, target_file)

sys.exit()

print(pipeline.get_params())

from sklearn.model_selection import GridSearchCV

parameters = {
    'tfidf__use_idf': [True, ],
    'tfidf__ngram_range': [(1, 1), (1, 2)],
    'best__n_components': [50, 75, 100, 125, 150, 200, ],
    'svc__C': [ 0.8, 0.9, 1.0, 1.1, 1.2, ],
    'svc__gamma': [ 'scale', ],
    'svc__kernel': ['linear', 'rbf', ],
    'svc__class_weight': [None, ],
}

gs_clf = GridSearchCV(pipeline, parameters, cv=5, n_jobs=my_cpu_count()-1, verbose=2)
gs_clf = gs_clf.fit(X, y)

print('Best score: {0}'.format(gs_clf.best_score_))
print("Best parameters set:")
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))
