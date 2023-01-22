# -*- mode: python; -*-

'''
The MIT License (MIT)
Copyright (C) 2023 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

'''
create a workbook demo

%run Excel_ops.py
'''

import numpy as np
import pandas as pd
import datetime
import xlsxwriter

class Logbook():
    def __init__(self, filename):
        self.filename = filename
        self.workbook = xlsxwriter.Workbook(self.filename)

        #  https://xlsxwriter.readthedocs.io/format.html
        self.cell_format_table_header = self.workbook.add_format({  # for header row cells
            'num_format': '@',  # 文字列
            'font_name': 'ＭＳ ゴシック',
            'bg_color': '#DDEBF7',
        })
        self.cell_format_int64 = self.workbook.add_format({      # for np.int64 cells
            'num_format': '#,##0',  # 123,456
            'font_name': 'ＭＳ ゴシック',
        })
        self.cell_format_float64 = self.workbook.add_format({    # for np.float64 cells
            'num_format': '#,##0.000',  # 123,456.000
            'font_name': 'ＭＳ ゴシック',
        })
        self.cell_format_str = self.workbook.add_format({        # for np.object_ (strings) cells
            'num_format': '@',  # 文字列
            'font_name': 'ＭＳ ゴシック',
        })
        self.cell_format_datetime = self.workbook.add_format({   # for np.dtype('datetime64[ns]') cells
            'num_format': 'yyyy/mm/dd hh:mm:ss',
            'font_name': 'ＭＳ ゴシック',
        })

    def workbook(self):
        return self.workbook

    def close(self):
        self.workbook.close()

    def add_df(self, df, row_base=0, col_base=0):
        def get_cell_format(val):
            if (val == np.int64) or (type(val) == int):
                return self.cell_format_int64
            elif (val == np.float64) or (type(val) == float):
                return self.cell_format_float64
            elif (val == np.object_) or (type(val) == str):
                return self.cell_format_str
            elif (val == np.dtype('datetime64[ns]')) or (type(val) == pd._libs.tslibs.timestamps.Timestamp):
                return self.cell_format_datetime
            else:
                assert False, 'unsupported dtype={0}, val={1}'.format(type(val), str(val))
        # end-of def get_cell_format()

        sheetname = df.name
        self.worksheet = self.workbook.add_worksheet(sheetname)

        column_names  = df.columns.tolist()  # ['int', 'float', 'str', 'datetime64', 'datetime']
        column_dtypes = df.dtypes.tolist()   # [dtype('int64'), dtype('float64'), dtype('O'), dtype('<M8[ns]'), dtype('<M8[ns]')]

        row = 0
        col = 0

        def row_act():
            nonlocal row_base, row
            return row_base + row
        def col_act():
            nonlocal col_base, col
            return col_base + col

        for col_name in column_names:
            # print('col_name={0} {1}'.format(col_name, type(col_name)))
            self.worksheet.write(row_act(), col_act(), column_names[col], self.cell_format_table_header)
            if col_name in ['str']:
                self.worksheet.set_column(col_act(), col_act(), 0)  # hide column
            elif col_name in ['int']:
                self.worksheet.set_column(col_act(), col_act(), 5)  # for int column
            elif col_name in ['datetime64', 'datetime']:
                self.worksheet.set_column(col_act(), col_act(), 20) # for datetime column
            col += 1

        row += 1
        for index, rest in df.iterrows():
            col = 0
            for col_name in column_names:
                self.worksheet.write(row_act(), col_act(), rest[col], get_cell_format(column_dtypes[col]))
                col += 1
            row += 1

df1 = pd.DataFrame(
    data=[[
        1,
        1.23,
        "00000123",
        np.datetime64("2023-01-23"),
        datetime.datetime(2023, 1, 23)
    ]],
    columns=['int', 'float', 'str', 'datetime64', 'datetime']
)
df1.name = 'df1'

df2 = df1.copy()
df2.name = 'df2'

filename = 'xlsxwriter-example.xlsx'
logbook = Logbook(filename)
logbook.add_df(df1)
logbook.add_df(df2, 2, 2)
logbook.close()

