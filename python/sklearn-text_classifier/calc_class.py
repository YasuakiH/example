# calc_class.py

'''
The MIT License (MIT)
Copyright (C) 2022 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

'''
ipython
%run calc_class.py
'''

# ------------------------------------------------------------
# 自立語を抽出して分かち書きにする
# ------------------------------------------------------------

def split_str(s):
    '''連接する文字どうしを分離'''
    s = s.replace('。)','。 )')
    return s

import os
from janome.tokenizer import Tokenizer
import unicodedata
t = None
stoplist = ('の','する','-','+','/','[',']','.','(',')',',',':',';','=','~','%','、') # 無視する語
def get_self_sufficient_word(s, result_type = 'list'):
    '''strから自立語(名詞、形容詞、動詞)を抽出してlistまたは分かち書きしたstrとして返す'''
    global t
    # ユーザー定義辞書を使う <https://mocobeta.github.io/janome/>
    # 単語の追加方法 - MeCab <https://taku910.github.io/mecab/dic.html>
    userdic = 'userdic-utf_8.csv'
    if t is None:
        if os.path.exists(userdic):
            t = Tokenizer(userdic, udic_enc="utf8")
        else:
            t = Tokenizer()
    result = []
    for token in t.tokenize(split_str(str(s).lower()).replace('\n','。')):
        base_form = unicodedata.normalize("NFKC", token.base_form)
        if base_form in stoplist:
            continue
        part_of_speech = token.part_of_speech.split(',')[0]
        if part_of_speech in ['名詞','形容詞','動詞']:
            result.append(base_form)
    if result_type == 'list':
        return result
    elif result_type == 'str':
        return ' '.join(result)
    else:
        assert False, 'unexpected: result_type={}'.format(str(result_type))
# end-of def get_self_sufficient_word(s, result_type = 'list'):

# ------------------------------------------------------------
# DataFrame作成
# ------------------------------------------------------------

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

global_shared_df = None

class calc():
    def __init__(self, filename, subset='train', test_size = 0.25, random_state = 42, return_X_y = False):
        global global_shared_df

        # 必要な属性: data, target, filenames, DESCR, target_names
        # https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_20newsgroups.html
        self.DESCR = 'HP Calculator user reviews on Amazon'
        self.filenames = None
        self.shared_df = global_shared_df

        update_df = False

        if self.shared_df is None:
            update_df = True

            self.shared_df = pd.read_excel(
                filename,
                # index_col=0,
                usecols=[
                    'ID',
                    'PRODUCT',
                    'CUSTOMER',
                    'SCORE',
                    'SUBJECT',
                    'SUBMITDATE',
                    'VOICEOFCUSTOMER',
                    'SUPPORT',
                ])
            self.shared_df.rename({'VOICEOFCUSTOMER':'VOC'}, inplace=True, axis='columns')
            self.shared_df.index.name ='index'

            self.shared_df.loc[ self.shared_df['SCORE'] > 3, 'LIKE' ] = 'T'
            self.shared_df.loc[ self.shared_df['SCORE'] <= 3, 'LIKE' ] = 'F'

            # テキスト列を分かち書きする。(オリジナル列は'列名_ORIG'として保存、分かち書きは'列名_STR'に保存)
            if 'VOC_ORIG' not in self.shared_df.columns.to_list():
                self.shared_df['VOC_ORIG'] = self.shared_df['VOC']
                do_text_analysis = True
            if 'SUBJECT_ORIG' not in self.shared_df.columns.to_list():
                self.shared_df['SUBJECT_ORIG'] = self.shared_df['SUBJECT']
                do_text_analysis = True

            # 時間のかかる処理
            # self.shared_df['VOC_MORPHEME']     = self.shared_df['VOC_ORIG'].map(get_self_sufficient_word)
            # self.shared_df['VOC_STR']          = self.shared_df['VOC_ORIG'].apply(get_self_sufficient_word, result_type='str')
            # 
            # self.shared_df['SUBJECT_MORPHEME'] = self.shared_df['SUBJECT_ORIG'].map(get_self_sufficient_word)
            # self.shared_df['SUBJECT_STR']      = self.shared_df['SUBJECT_ORIG'].apply(get_self_sufficient_word, result_type='str')

            # VOC_STR: PRODUCT, VOC_STR, SUBJECT_STR を連接し、1列に格納
            # self.shared_df['VOC_STR'] = self.shared_df['PRODUCT'] + ' ' + self.shared_df['VOC_STR'] + ' ' + self.shared_df['SUBJECT_STR']

            global_shared_df = self.shared_df
            # print('global_shared_df.columns={}'.format(global_shared_df.columns.tolist()))
        # end-of if self.shared_df is None

        # X_df = self.shared_df[['PRODUCT','CUSTOMER', 'VOC_STR']].values.tolist()
        # X_df = self.shared_df['VOC_STR'].values.tolist()
        X_df = self.shared_df.index.tolist()

        # y_s  = self.shared_df['LIKE'].values.tolist()
        y_s = []
        for item in self.shared_df['LIKE'].values.tolist():
            if item == 'T':
                y_s.append(1)
            elif item == 'F':
                y_s.append(0)
            else:
                assert False, 'this must not be reache: item={0} ({1})'.format(item, type(item))
        y_s = pd.DataFrame(data=y_s, columns=['y'])
        y_s.index.name='index'

        X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
            X_df,
            y_s,
            test_size=test_size,
            random_state=random_state,
        )
        usecols = [
            'ID',
            'PRODUCT',
            'CUSTOMER',
            # 'SCORE',
            # 'SUBJECT_MORPHEME',
            'SUBMITDATE',
            # 'VOC_MORPHEME',
            'SUPPORT',
            # 'LIKE',
            'VOC',
            'SUBJECT',
            # 'VOC_STR',
            # 'SUBJECT_STR',
            ]
        if subset == 'train':
            self.df     = self.shared_df[usecols].loc[ self.shared_df.index.isin(X_train_df) ]
            self.data   = self.df
            self.target = y_train_s
        elif subset == 'test':
            self.df     = self.shared_df[usecols].loc[ self.shared_df.index.isin(X_test_df) ]
            self.data   = self.df
            self.target = y_test_s
        elif subset == 'all':
            self.data   = self.shared_df[usecols]
            self.target = y_s
        else:
            assert False, 'subset must be in [train, test, all]'

        self.target_names = sorted(list(set(self.target)))
    # end-of def __init__(self, ...)

    # def get_x_y(): これは動かない
    #     if return_X_y:
    #         return (self.data, self.target)
    #     else:
    #         result = {}
    #         result['data'] = self.data
    #         result['target'] = self.target
    #         result['filenames'] = self.filenames
    #         result['filenames'] = self.filenames
    #         result['DESCR'] = self.DESCR
    #         result['df'] = self.shared_df
    #         result['target_names'] = self.target_names
    #         return result
    # end-of def get_x_y()

# end-of class calc()

'''
サンプル操作

csvfile='dataset_calc.xlsx'
calc_train = calc(csvfile, subset='train')
calc_test = calc(csvfile, subset='test')

print('calc_train.target_names={}'.format(calc_train.target_names))
print('calc_test.target_names ={}'.format(calc_test.target_names))

print('calc_train.data[0]  ={}'.format( calc_train.data[0] ))
print('calc_train.target[0]={}'.format (calc_train.target[0] ))


# scikit-learnでテキストをトークン化する
# 6.2.特徴抽出
# https://scikit-learn.org/stable/modules/feature_extraction.html

# テキストの前処理、トークン化、ストップワードのフィルタリングはすべてCountVectorizerに含まれており、特徴量の辞書を構築し、ドキュメントを特徴ベクトルに変換する。

from sklearn.feature_extraction.text import CountVectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(calc_train.data)
X_train_counts.shape

# CountVectorizerは、単語や連続した文字のN-gramのカウントをサポートしています。ベクトライザは、特徴量インデックスの辞書を構築する。

print(count_vect.vocabulary_.get(u'スイッチ'))
print(count_vect.vocabulary_.get(u'hp'))

# 発生頻度から頻度へ
# 出現回数は良いスタートだが、同じトピックについて話していても、長い文書の方が短い文書よりも平均出現回数が高くなるという問題がある。
# 
# このような潜在的な矛盾を避けるためには、文書中の各単語の出現回数を文書中の総単語数で割ればよい。この新しい特徴は、Term Frequenciesの略でtfと呼ばれている。
# 
# tfのもう一つの改良点は、コーパスの多くの文書に出現し、コーパスのごく一部にしか出現しない単語よりも情報量が少ない単語の重みを小さくすることである。
# 
# このダウンスケールは、「Term Frequency times Inverse Document Frequency」の略で、tf-idfと呼ばれている。
# 
# tfと tf-idfは、TfidfTransformerを使用して以下のように計算することができます。

from sklearn.feature_extraction.text import TfidfTransformer
tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)
X_train_tf.shape

# 上記のサンプルコードでは、まずfit(...)メソッドで推定量をデータにフィットさせ、次にtransform(...)メソッドでカウント行列をtf-idf表現に変換しています。 この2つの手順を組み合わせれば、冗長な処理を省いて同じ最終結果をより速く達成することが可能です。これは、以下のようにfit_transform(...)メソッドを使うことによって行われます。

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

# 分類器の学習
# 特徴量が得られたので、分類器を学習して投稿のカテゴリを予測することができます。scikit-learnには ナイーブベイズ分類器があり、単語数に最も適したものは多項式分類器です。

from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train_tfidf, calc_train.target)


# 新しい文書に対する結果を予測するためには、先ほどとほぼ同じ特徴抽出チェーンを使って特徴を抽出する必要があります。 違いは、変換器はすでに学習セットにフィットしているので、fit_transformの代わりにtransformを呼んでいることです。

# docs_new = ['God is love', 'OpenGL on the GPU is fast']
docs_new = [
    '計算 機 使う 続ける 頭 中 スタック できる 計算 進める 後戻り 簡単 できる すごい スムーズ 計算 できる 普通 電卓 使える なる rpn 便利 意味  スタック 頭 中 ある 計算 間違える しまう 意味',
    '欧米 高校生 ターゲット 教育 用 電卓 hp 38 シリーズ',
    'hp rpn calc',
    'apple microsoft google', 
    get_self_sufficient_word('サポートベクターマシン（SVM）は、分類、回帰、外れ値検出などに用いられる教師あり学習法の一種である。', result_type = 'str'),
]
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    # print('%r --> %s' % (doc, calc_train.target_names[category]))
    print('%r => %s' % (doc, category))

# パイプラインの構築
# vectorizer => transformer => classifier をより簡単に扱うために、scikit-learnは複合分類器のように振る舞うPipelineクラスを提供します。

from sklearn.pipeline import Pipeline
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', MultinomialNB()),
])


# vect, tfidf, clf(classifier)という名前は任意ですが、後ほど適切なハイパーパラメータをグリッド検索するために使用します。 これで1つのコマンドでモデルを学習することができます。

text_clf.fit(calc_train.data, calc_train.target)

# テストセットでの性能評価
# モデルの予測精度の評価も同様に簡単である。

import numpy as np
# calc_test = fetch_20newsgroups(subset='test',
#     categories=categories, shuffle=True, random_state=42)
docs_test = calc_test.data
predicted = text_clf.predict(docs_test)
print('predicted={}'.format(predicted))
np.mean(predicted == calc_test.target)  # 0.7142857142857143

# 83.5%の精度を達成しました。線形サポートベクターマシン(SVM)を使って、これより良い結果が得られるか見てみましょう。学習器の変更は、別の分類器オブジェクトをパイプラインに差し込むだけで可能です。

from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([
    ('vect', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('clf', SGDClassifier(loss='hinge', penalty='l2',
                          alpha=1e-3, random_state=42,
                          max_iter=5, tol=None)),
])

text_clf.fit(calc_train.data, calc_train.target)

predicted = text_clf.predict(docs_test)
print('predicted={}'.format(predicted))
np.mean(predicted == calc_test.target)  # 0.7619047619047619

# scikit-learnは、結果をより詳細に性能分析するためのユーティリティをさらに提供しています。

from sklearn import metrics
print(metrics.classification_report(calc_test.target, predicted,
    target_names=calc_test.target_names))

#               precision    recall  f1-score   support
# 
#            T       0.67      0.33      0.44         6
#            F       0.78      0.93      0.85        15
# 
#     accuracy                           0.76        21
#    macro avg       0.72      0.63      0.65        21
# weighted avg       0.75      0.76      0.73        21

print(metrics.confusion_matrix(calc_test.target, predicted))

# [[ 2  4]
#  [ 1 14]]

# 予測が外れたケースを一覧

for doc, category in zip(docs_test, predicted):
    if category == 'F':
        print('%r => %s' % (doc, category))

'HP 12C hp 12 c 普通 モデル 比べる ボタン タッチ 気持ち いい 感触' => F

'HP Prime texas instruments 社 ti nspire cas 強い 意識 グラフ 電卓 色々 メニュー  アイコン 化 選択 せる 仕様 ため ti nspire 様 階層 深い メニュー 構成 選択 楽 アル ファベット キー 表 出す 一方 sin log 等 関数 キー メニュー 仕舞う 込む ti nspire  逆 仕様 使い方 どちら 選ぶ 分 れる 事 思う 入力 方式 電卓 特徴 rpn 選択 時 数式 囲む 書式 エラー なる 使う づらい 5 0 g 深い メニュー 階層 苦労 取っ掛り 易い グラフ 電卓 思う グラフ 電卓 表示 位置 指 タップ 変える られる ズームイン アウト +- キー 簡単 出来る 非常 気に入る 全て 機能 使う 倒す 訳 ある 気に入る いる 楽しみ 便利' => F

'HP 12C 商品 自体 速やか 届く パッケージ 開ける 電源 入れる みる 液晶 セグメント  欠け 数字 読める 写真 1 2 3 4 5 6 7 8 9 0 入力 ところ メイド イン チャイナ 古い もの 液晶 はんだ 付け 甘い もの ある それ あたる しまう 残念 返品 注文 直す 中 残念 良品' => F

'HP 12C 購入 前 他 レビュー 見る 普通 12 c 覚悟 購入 結果 的 想定 中 商品 pkg 裏面 貼る てる hdpma 123 e 02 mwc 表記 シール 剥がす hp product # 表記 f 2231 aa # 12 届く 物 電池 数 cr 2032 2 ネット 上 入手 情報 対照 古い f 2231 a 25 th anniversary edition 電池 数 cr 2032 1 異なる 事 分かる hp サポート ページ sn 入力 見る 結果 2018 年 4 月 製造 もの フィリピン 製 ある こと 間違い rpn 電卓 実機 使う 目的 完遂 本来 文句 言う ある 25 周年 記念 モデル 宣伝 不満 f 2231 aa # 12 モデル amazon 8 000 割 販売 所 ある 久々 レビュー 投稿 25 周年 記念 モデル ある 満足' => F

'HP 12C world collections 販売 amazon co jp 発送 元祖 12 c 注文 platinum 到着 画像 違う もの 到着 開 梱 しまう 後 気がつく amazon 経由 クレーム 入れる 返品 できる world collections 返品 受け入れ 理由 商品 発送 遅れ 現品 相違 認める 納品 者 評価  呆け 対応 返品 mitsukansha 販売 amazon co jp 発送 注文 元祖 12 c ボタン いる 目ざ わり rpn 表示 無い ゴールド 色 高級 感 あり 現品 相違 world collections 注意' => F

'''
