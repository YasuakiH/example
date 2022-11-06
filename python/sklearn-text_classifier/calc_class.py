# calc_class.py

'''
The MIT License (MIT)
Copyright (C) 2022 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import sys
import os

import multiprocessing
def my_cpu_count():
    return multiprocessing.cpu_count()

import io
def info_to_str(df):
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

from janome.tokenizer import Tokenizer
import unicodedata
t = None
def get_self_sufficient_word(s, result_type = 'list'):
    def split_str(s):
        s_orig = s
        s = s.replace('。)', '。 )')
        s = s.replace('>。', '> 。)')
        s = s.replace('。<', '。 <)')
        s = s.replace('。。', '。 。')
        s = s.replace('>。。<', '\n')
        s = s.replace('<HTML>', '\n')
        s = s.replace('<XML>', '\n')
        s = s.replace('<SGML>', '\n')
        return s

    global t
    stoplist = ('<', '>', '。', 'の','する','-','+','/','[',']','.','(',')',',',':',';','=','~','%','、')

    # <https://mocobeta.github.io/janome/>
    # <https://taku910.github.io/mecab/dic.html>
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

import os
import io
import numpy as np
import pandas as pd
import getpass
from sklearn.model_selection import train_test_split

class calc():
    def __init__(self, filename, subset='train', test_size = 0.25, random_state = 42, return_X_y = False):
        def read_excel(file, sheet_name=None):
            assert os.path.isfile(file), 'a file must be exists: {0}'.format(file)
            if sheet_name is None:
                sheet_name = 'Sheet1'
            try:
                df = pd.read_excel(file, sheet_name=sheet_name)
            except Exception as e1:
                _ = str(e1).index("Can't find workbook in OLE2 compound document")
                import msoffcrypto
                decrypted = io.BytesIO()
                with open(file, "rb") as f:
                    file = msoffcrypto.OfficeFile(f)
                    file.load_key(password=getpass.getpass(prompt='Password: ', stream=None))
                    file.decrypt(decrypted)
                df = pd.read_excel(decrypted, sheet_name=sheet_name)
            return df

        self.DESCR = 'rpn calculator user reviews'
        self.filenames = None

        work_df = read_excel(filename, 'Sheet1')
        usecols = [
            'ID',
            'PRODUCT',
            'CUSTOMER',
            'SUBJECT',
            'SUBMITDATE',
            'VOC',
            'SUPPLEMENT',
            'FOLLOWUP',
            'CLASS',
            'LV',
            'CORRECT',
        ]
        self.df = work_df[usecols].copy()
        del work_df
        self.df.index.name ='index'

        self.df['ID'] = self.df['ID'].astype('string')

        self.df['CUSTOMER'] = self.df['CUSTOMER'].astype('string')
        self.df.loc[self.df.CUSTOMER.isna(), 'CUSTOMER'] = 'UNKNOWN'

        self.df.loc[ self.df.VOC.isna(), 'VOC' ] = ''

        self.df.loc[ self.df.SUPPLEMENT.isna(), 'SUPPLEMENT' ] = ''
        self.df.SUPPLEMENT = self.df.SUPPLEMENT.map(lambda x: x.replace('<HTML>','')).map(lambda x: x.replace('<XML>','')).map(lambda x: x.replace('<SGML>',''))

        self.df.loc[ self.df.FOLLOWUP.isna(), 'FOLLOWUP' ] = ''
        self.df.FOLLOWUP = self.df.FOLLOWUP.map(lambda x: str(x).replace('HTML', '')).map(lambda x: x.replace('<XML>', '')).map(lambda x: x.replace('<SGML>',''))

        self.df['VOC_CAT'] = (
            self.df['SUBJECT'] + ' ' +
            self.df['VOC'] + ' ' +
            self.df['SUPPLEMENT'] + ' ' +
            self.df['FOLLOWUP']
        )

        self.df['VOC_MORPHEME_S'] = self.df['VOC_CAT'].apply(get_self_sufficient_word, result_type='str')
        self.df['SUBJECT_MORPHEME_S'] = self.df['SUBJECT'].apply(get_self_sufficient_word, result_type='str')

        print('self.df.VOC_MORPHEME_S=\n{0}'.format(self.df.VOC_MORPHEME_S))
        print('self.df.SUBJECT_MORPHEME_S=\n{0}'.format(self.df.SUBJECT_MORPHEME_S))

        self.df['SUBMITDATE'] = pd.to_datetime(self.df['SUBMITDATE'])

        self.df.loc[ self.df['LV'].isin([1,2,3]) , 'LV' ] = 'LOW'
        self.df.loc[ self.df['LV'].isin([4,5])   , 'LV' ] = 'HIGH'

        self.df['CORRECT'] = self.df['CORRECT'].astype('string')

        self.df.loc[ self.df.CORRECT.isin([1, 1.0, '1', 'T', 'TRUE' ]), 'CORRECT'] = 'USE'
        self.df.loc[ self.df.CORRECT.isin([0, 0.0, '0', 'F', 'FALSE']), 'CORRECT'] = 'OMIT'
        assert set(self.df.CORRECT.values.unique().tolist()) == set(['OMIT','USE']), 'set(self.df.CORRECT.values.unique().tolist()) = {0}'.format(set(self.df.CORRECT.values.unique().tolist()) )

        print('self.df={0}'.format(self.df))
        print('self.df.columns.to_list()={0}'.format(self.df.columns.to_list()))

        X_idx_list = self.df.index.tolist()

        y_val_df = self.df[['CORRECT',]]

        X_train_df, X_test_df, y_train_s, y_test_s = train_test_split(
            X_idx_list,
            y_val_df,
            test_size = test_size,
            random_state = random_state,
        )
        
        assert len(X_train_df) + len(X_test_df) == len(self.df)

        usecols.append('VOC_MORPHEME_S')
        usecols.append('SUBJECT_MORPHEME_S')
        usecols.append('VOC_CAT')

        if subset == 'train':
            self.df     = self.df[usecols].loc[ self.df.index.isin(X_train_df) ]
            self.data   = self.df.values
            self.target = y_train_s
        elif subset == 'test':
            self.df     = self.df[usecols].loc[ self.df.index.isin(X_test_df) ]
            self.data   = self.df.values
            self.target = y_test_s
        elif subset == 'all':
            self.df     = self.df[usecols]
            self.data   = self.df.values
            self.target = y_val_df
        else:
            assert False, 'subset must be in [train, test, all]'

        self.target_names = sorted(list(set(self.target)))
