# text_classifier.py

'''
The MIT License (MIT)
Copyright (C) 2022 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

'''
ipython
%run text_classifier4.py
'''

# ========================================================================
# 異種データソースを扱うColumn Transformer(列変換器)
# ========================================================================
# https://scikit-learn.org/stable/auto_examples/compose/plot_column_transformer.html

# データセットには、異なる特徴抽出や処理パイプラインを必要とするコンポーネントが含まれることがよくある。
# このようなシナリオは、以下のような場合に発生する可能性がある。
# 1. データセットが異種データ型（ラスター画像とテキストキャプションなど）から構成されている場合。
# 2. データセットがpandas.DataFrameに格納されており、異なる列は異なる処理パイプラインを必要とする場合。
# 
# この例では、異なるタイプの特徴を含むデータセットに対してColumnTransformerを使用する方法を説明する。
# 特徴量の選択は特に有用ではありませんが、テクニックを説明するのに役立つ。

import numpy as np
import pandas as pd
import re

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from sklearn.preprocessing import OneHotEncoder
# from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.svm import LinearSVC

import io
def info_to_str(df):
    '''pd.DataFrameの構造の文字列を返す'''
    buf = io.StringIO()
    df.info(buf=buf)
    s = buf.getvalue()
    s = re.sub('^<class .+>\n', '', s)
    s = re.sub('^Int64Index:.+\n', '', s)
    s = re.sub('^Data columns.+\n', '', s)
    s = re.sub('dtypes:.+\n', '', s)
    s = re.sub('memory usage:.+\n', '', s)
    s = re.sub('\n$', '', s)
    return s

# --------------------------------
# データセット作成
# --------------------------------
print('データセット作成')

from calc_class import calc, get_self_sufficient_word
csvfile='dataset_calc.xlsx'

calc_train = calc(csvfile, subset='train')  # calc_class.calc

X_train = calc_train.df         # pd.DataFrame
print('X_train.info()=\n{}'.format(info_to_str(X_train)))

y_train = calc_train.target     # array of a char ['T','F']
print('y_train.info()=\n{}'.format(info_to_str(y_train)))

# サイズ縮小
# X_train = X_train[0:10]
# y_train = y_train[0:10]

print('X_train=\n{0} ({1})'.format(X_train, type(X_train)))
print('y_train=\n{0} ({1})'.format(y_train, type(y_train)))
print('X_train.iloc[0]=\n{0} ({1})'.format(X_train.iloc[0], type(X_train.iloc[0])))
print('y_train.iloc[0]=\n{0} ({1})'.format(y_train.iloc[0], type(y_train.iloc[0])))

assert isinstance(X_train, pd.DataFrame) and isinstance(X_train.iloc[0], pd.Series)
assert isinstance(y_train, pd.DataFrame) and isinstance(y_train.iloc[0], pd.Series)

calc_test = calc(csvfile, subset='test')  # calc_class.calc

X_test = calc_test.df           # pd.DataFrame
print('X_test.info()=\n{}'.format(info_to_str(X_test)))

y_test = calc_test.target       # array of a char ['T','F']
print('y_test.info()=\n{}'.format(info_to_str(y_test)))


print('X_test=\n{0} ({1})'.format(X_test, type(X_test)))
print('y_test=\n{0} ({1})'.format(y_test, type(y_test)))
print('X_test.iloc[0]=\n{0} ({1})'.format(X_test.iloc[0], type(X_test.iloc[0])))
print('y_test.iloc[0]=\n{0} ({1})'.format(y_test.iloc[0], type(y_test.iloc[0])))

# assert len(X_train) + len(X_test) == 118

# --------------------------------
# トランスフォーマー作成
# --------------------------------
print('トランスフォーマー作成')

# まず、各投稿の件名と本文を抽出するトランスフォーマーが必要です。
# これはステートレス変換（学習データの状態情報を必要としない）なので、データ変換を行う関数を定義し、FunctionTransformerを使って scikit-learn トランスフォーマーを作成することができます。

def subject_body_extractor(posts):
    if isinstance(posts, pd.DataFrame):
        posts_cp = posts.copy()
        # 時間のかかる処理
        posts_cp['VOC_MORPHEME']   = posts_cp['VOC'].map(get_self_sufficient_word)
        posts_cp['VOC_MORPHEME_S'] = posts_cp['VOC'].apply(get_self_sufficient_word, result_type='str')
        posts_cp['SUBJECT_MORPHEME']   = posts_cp['SUBJECT'].map(get_self_sufficient_word)
        posts_cp['SUBJECT_MORPHEME_S'] = posts_cp['SUBJECT'].apply(get_self_sufficient_word, result_type='str')
    else:
        assert False, 'this must not be reached: type(posts_cp)={}'.format(type(posts_cp))
    return posts_cp

'''
print( info_to_str(calc_test.df) )

 #   Column      Non-Null Count  Dtype
---  ------      --------------  -----
 0   ID          30 non-null     int64
 1   PRODUCT     30 non-null     object
 2   CUSTOMER    30 non-null     object
 3   SUBMITDATE  30 non-null     datetime64[ns]
 4   SUPPORT     30 non-null     int64
 5   VOC         30 non-null     object
 6   SUBJECT     30 non-null     object

subject_body_extractor(calc_test.df)

        ID  ...             SUBJECT_MORPHEME_S
index       ...
0        1  ...                 モノ 良い 液晶 画面 ゴミ
4        5  ...                     hp 製 遜色 ある
10      11  ...                       本家 品質 良い
11      12  ...                          素晴らしい
 :      :    : 
109    110  ...          英語 取る 説 流す 読める 使う 切れる
111    112  ...                進化 コンピュータ 機能 満載
116    117  ...                           格好いい
[30 rows x 11 columns]

print( info_to_str(subject_body_extractor(calc_test.df)) )

 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   ID                  30 non-null     int64
 1   PRODUCT             30 non-null     object
 2   CUSTOMER            30 non-null     object
 3   SUBMITDATE          30 non-null     datetime64[ns]
 4   SUPPORT             30 non-null     int64
 5   VOC                 30 non-null     object
 6   SUBJECT             30 non-null     object
 7   VOC_MORPHEME        30 non-null     object
 8   VOC_MORPHEME_S      30 non-null     object
 9   SUBJECT_MORPHEME    30 non-null     object
 10  SUBJECT_MORPHEME_S  30 non-null     object
'''

subject_body_transformer = FunctionTransformer(subject_body_extractor)

# また、文章の長さと文の数を抽出する変換器を作成します。
def text_stats(posts):
    # return [{"length": len(text), "num_sentences": text.count(".")} for text in posts]
    if isinstance(posts, pd.DataFrame):
        # print('text_stats(): posts=\n{0} ({1})'.format(posts, type(posts)))
        result = []
        for index, rest in posts.iterrows():
            text = rest['VOC']
            length = len(text)
            num_sentences = re.sub('\n+', '\n', text.replace('。', '\n')).count('\n')
            result.append({
                'length': length,
                'num_sentences': num_sentences
                })
        # print('text_stats(): result=\n{0} ({1})'.format(result, type(result)))
        return result
    elif isinstance(posts, pd.Series):
        # print('text_stats(): posts=\n{0} ({1})'.format(posts, type(posts)))
        result = []
        for text in posts:
            length = len(text)
            num_sentences = re.sub('\n+', '\n', text.replace('。', '\n')).count('\n')
            result.append({
                'length': length,
                'num_sentences': num_sentences
                })
        # print('text_stats(): result=\n{0} ({1})'.format(result, type(result)))
        return result
    else:
        assert False, 'this must not be reached: type(posts)={}'.format(type(posts))

'''
text_stats(X_train)

[{'length': 293, 'num_sentences': 8},
 {'length': 695, 'num_sentences': 18},
 {'length': 344, 'num_sentences': 8},
    :
 {'length': 391, 'num_sentences': 11},
 {'length': 264, 'num_sentences': 8}]
'''

text_stats_transformer = FunctionTransformer(text_stats)


# --------------------------------
# 分類のパイプライン
# --------------------------------

# 以下のパイプラインは、SubjectBodyExtractor を用いて各投稿から件名と本文を抽出し、(n_samples, 2) の配列を生成している。
# この配列を使って、ColumnTransformer を使って、件名と本文の標準的な単語袋素性と、本文の長さと文の数を計算する。
# そして、それらを重み付きで結合し、結合された特徴量に対して分類器を学習する。

'''
'SUBJECT_MORPHEME_S', 'VOC_MORPHEME_S' の列番号を得る。

print( info_to_str(subject_body_extractor(calc_test.df)) )

 #   Column              Non-Null Count  Dtype
---  ------              --------------  -----
 0   ID                  30 non-null     int64
 1   PRODUCT             30 non-null     object
 2   CUSTOMER            30 non-null     object
 3   SUBMITDATE          30 non-null     datetime64[ns]
 4   SUPPORT             30 non-null     int64
 5   VOC                 30 non-null     object
 6   SUBJECT             30 non-null     object
 7   VOC_MORPHEME        30 non-null     object
 8   VOC_MORPHEME_S      30 non-null     object
 9   SUBJECT_MORPHEME    30 non-null     object
 10  SUBJECT_MORPHEME_S  30 non-null     object
'''

pipeline = Pipeline(
    [
        # Extract subject & body
        ('subjectbody', subject_body_transformer),
        # Use ColumnTransformer to combine the subject and body features
        (
            'union',
            ColumnTransformer(
                [
                    # bag-of-words for subject (col 9)
                    ('subject', TfidfVectorizer(use_idf=True), 10),                       # 'SUBJECT_MORPHEME_S' の列番号
                    # bag-of-words with decomposition for body (col 1)
                    (
                        'body_bow',
                        Pipeline(
                            [
                                ('tfidf', TfidfVectorizer()),
                                ('best', TruncatedSVD(n_components=10)),      # 特異値分解による次元削減
                            ]
                        ),
                        8,                                                    # 'VOC_MORPHEME_S' の列番号
                    ),
                    # Pipeline for pulling text stats from post's body
                    (
                        'body_stats',
                        Pipeline(
                            [
                                (
                                    'stats',
                                    text_stats_transformer,
                                ),  # returns a list of dicts
                                (
                                    'vect',
                                    DictVectorizer(),
                                ),  # list of dicts -> feature matrix
                            ]
                        ),
                        5,                                                    # 'VOC' の列番号
                    ),
                    (
                        'support_scaler',
                        # MinMaxScaler(),
                        StandardScaler(),
                        [4],                                                  # 'SUPPORT' の列番号
                    ),
                    (
                        'product_category',
                        # OneHotEncoder(dtype='int'),
                        CountVectorizer(),
                        1,                                                    # 'PRODUCT' の列番号
                    ),
                ],
                # weight above ColumnTransformer features
                transformer_weights={
                    'subject': 2.0,
                    'body_bow': 1.0,
                    'body_stats': 1.0,
                    'support_scaler': 1.0,
                    'product_category': 1.0,
                },
            ),
        ),
        # Use a SVC classifier on the combined features
        ('svc', LinearSVC(dual=True, C=0.05, max_iter=2000)),
    ],
    verbose=True,
)

# 最後に、学習データに対してパイプラインを適用し、X_testのトピックを予測するために使用する。
# そして、我々のパイプラインのパフォーマンスメトリクスを出力する。
pipeline.fit(X_train, y_train.values.ravel())

y_pred = pipeline.predict(X_test)
print('予測: y_pred=\n{}'.format(y_pred))
print('正解: y_test.values.ravel()=\n{}'.format(y_test.values.ravel()))

# 精度 accuracy
accuracy = np.mean(y_pred == y_test.values.ravel())
print('accuracy={0}\n'.format(accuracy))

print("Classification report:\n\n{}".format(classification_report(y_test, y_pred)))

# --------------------------------
# Excel保存
# --------------------------------

calc_train.shared_df['TRAIN_OR_TEST'] = ''
calc_train.shared_df['y_target'] = ''
calc_train.shared_df['y_predict'] = ''

for index, rest in X_train.join(y_train).iterrows():
    id = rest['ID']
    y  = rest['y']
    calc_train.shared_df.loc[ calc_train.shared_df.ID == id, 'TRAIN_OR_TEST'] = 'train'
    calc_train.shared_df.loc[ calc_train.shared_df.ID == id, 'y_target'] = y

for index, rest in X_test.join(y_test).iterrows():
    id = rest['ID']
    y  = rest['y']
    calc_train.shared_df.loc[ calc_train.shared_df.ID == id, 'TRAIN_OR_TEST'] = 'test'
    calc_train.shared_df.loc[ calc_train.shared_df.ID == id, 'y_target'] = y

for index, rest in X_test.join( pd.DataFrame(data=y_pred,columns=['y_pred'],index=X_test.index) ).iterrows():
    id = rest['ID']
    y  = rest['y_pred']
    calc_train.shared_df.loc[ calc_train.shared_df.ID == id, 'TRAIN_OR_TEST'] = 'test'
    calc_train.shared_df.loc[ calc_train.shared_df.ID == id, 'y_predict'] = y

'''
pandas.DataFrame.to_excel
https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html
'''

calc_train.shared_df.to_excel('output.xlsx', sheet_name='Sheet1')


# --------------------------------
# グリッドサーチによるパラメータチューニング
# --------------------------------

# 3.2. Tuning the hyper-parameters of an estimator
# https://scikit-learn.org/stable/modules/grid_search.html#grid-search

# Sample pipeline for text feature extraction and evaluation
# https://scikit-learn.org/stable/auto_examples/model_selection/grid_search_text_feature_extraction.html

# 最適化アルゴリズムが集束しない場合 (ConvergenceWarning: Liblinear failed to converge, increase the number of iterations.)
# https://stackoverflow.com/questions/52670012/convergencewarning-liblinear-failed-to-converge-increase-the-number-of-iterati

from sklearn.model_selection import GridSearchCV

# ある推定量に対する全パラメータの名前と現在の値を求めるには pipeline.get_params() を実行する。

parameters = {
    # 'classifier__scaler__selected_model': pipeline.named_steps['classifier'].generate({
    #     # 'svc__C': [0.01, 0.1, 1.0],
    #     'sgd__alpha': [0.0001,],
    # }),
    # 'svc': [LinearSVC(), SGDClassifier()],
    # 'union__subject__use_idf': [True],
    # 'union__body_bow__best__n_components': [5, 10, 20],
    # 'svc__max_iter': [1000, 2000, 3000],
    'svc__max_iter': [2000, ],
    # 'svc__dual': [True, False],
    # 'svc__C': [0.02, 0.03, 0.05, ],
    'svc__C': [ 0.05, ],
    'union__transformer_weights': [
        None,
        {'subject': 1.0, 'body_bow': 1.5, 'body_stats': 0.8, 'support_scaler': 0.5, 'product_category': 1.0},
        {'subject': 1.0, 'body_bow': 1.0, 'body_stats': 1.0, 'support_scaler': 1.0, 'product_category': 1.0},
        {'subject': 2.0, 'body_bow': 1.0, 'body_stats': 1.0, 'support_scaler': 1.0, 'product_category': 1.0},
        {'subject': 1.0, 'body_bow': 1.0, 'body_stats': 2.0, 'support_scaler': 1.0, 'product_category': 1.0},
        {'subject': 1.0, 'body_bow': 1.0, 'body_stats': 1.0, 'support_scaler': 2.0, 'product_category': 1.0},
        {'subject': 1.0, 'body_bow': 2.0, 'body_stats': 1.0, 'support_scaler': 1.0, 'product_category': 2.0},

        {'subject': 2.0, 'body_bow': 2.0, 'body_stats': 1.0, 'support_scaler': 1.0, 'product_category': 1.0},
        {'subject': 2.0, 'body_bow': 1.0, 'body_stats': 2.0, 'support_scaler': 1.0, 'product_category': 1.0},
        {'subject': 2.0, 'body_bow': 1.0, 'body_stats': 1.0, 'support_scaler': 2.0, 'product_category': 1.0},
        {'subject': 2.0, 'body_bow': 1.0, 'body_stats': 1.0, 'support_scaler': 1.0, 'product_category': 2.0},

        {'subject': 2.0, 'body_bow': 2.0, 'body_stats': 2.0, 'support_scaler': 1.0, 'product_category': 1.0},
        {'subject': 2.0, 'body_bow': 2.0, 'body_stats': 1.0, 'support_scaler': 2.0, 'product_category': 1.0},
        {'subject': 2.0, 'body_bow': 2.0, 'body_stats': 1.0, 'support_scaler': 1.0, 'product_category': 2.0},

        {'subject': 2.0, 'body_bow': 2.0, 'body_stats': 1.0, 'support_scaler': 2.0, 'product_category': 1.0},
        {'subject': 2.0, 'body_bow': 1.0, 'body_stats': 2.0, 'support_scaler': 2.0, 'product_category': 1.0},
        {'subject': 2.0, 'body_bow': 1.0, 'body_stats': 1.0, 'support_scaler': 2.0, 'product_category': 2.0},

    ],
    # 'union__body_bow__tfidf__ngram_range': [(1, 1), (1, 2)],  # unigrams or bigrams
}
gs_clf = GridSearchCV(pipeline, parameters, cv=2, n_jobs=3)
gs_clf = gs_clf.fit(X_train, y_train.values.ravel())

print('Best score: {0}'.format(gs_clf.best_score_))
print("Best parameters set:")
for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

