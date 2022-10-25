#! weibull-plotly.py

'''
The MIT License (MIT)
Copyright (C) 2022 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

import sys
import datetime as dt
import datetime
import psycopg2
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit

# import dash
# from dash import dcc
# from dash import html

import dashplotlib

tb_name = 'staging_tb'
conn = None
cur = None

data_df = None

weibull_CF = None
weibull = None

def my_print(msg):
    print('weibull-plotly: {0}'.format(msg))

def conn_db():
    '''データベースとの接続を作成'''
    global conn, cur
    conn = psycopg2.connect("dbname=test user=postgres password=postgres")
    cur = conn.cursor()

def disconn_db():
    '''データベースとの接続を切断'''
    global conn, cur
    cur.close()
    conn.close()

def load_csv():
    '''CSVファイルをステージング行へロード'''
    global conn, tb_name
    my_print('load_csv() CSVファイルをステージング行へロード')

    # ワイブル分析用CSVファイルを元にデータフレーム作成
    csv_file = 'test.csv'
    csv_df = pd.read_csv(csv_file, encoding = 'cp932', sep='\t')

    # --------------------------------
    # ステージング表へINSERT
    # --------------------------------
    conn_db()  # データベースとの接続を作成

    # 既存の行を削除
    cur.execute('delete from {0};'.format(tb_name))

    # DBへINSERTする列値を整形。np.nan を None で置換。
    def nan_to_none(val):
        # my_print('val={0} {1}'.format(val, type(val)))
        if isinstance(val, float):
            # return None if np.isnan(val) else val
            if np.isnan(val):
                return None
            else:
                return int(val) if int(val) == val else val
        else:
            return val

    for index, rest in csv_df.iterrows():
        # my_print('{0}\t{1}'.format(index,rest[0]))
        row = (
            rest[0],               # index                                 79
            rest[1],               # SERIAL                 JKAZRCG13FA006160
            rest[2],               # PARTS_CD                 11_42_7_673_541
            rest[3],               # PARTS_NAME                    OIL_FILTER
            rest[4],               # PARTS_TRADEPRICE                      21
            rest[5],               # OBSERVDAY                     2022/06/09
            rest[6],               # REPORT_HEADER_ID                 R1200GS
            rest[7],               # REPORT_DETAIL_ID               08-09 USA
            rest[8],               # PRODUCT                              K25
            rest[9],               # NUMBER_OF_ITEMS                        1
            rest[10],              # REASON_BODY                      故障交換
            rest[11],              # UMSM                                  UM
            rest[12],              # METER_TOTAL                     26996896
            rest[13],              # REASON_KEY           failure_maintenance
            rest[14],              # OBSERVDAY_DT         2022/06/09 00:00:00
            nan_to_none(rest[15]), # METER_TOTAL_DIFF                     NaN
            nan_to_none(rest[16]), # CYCLE                           591808.0
            rest[17],              # COLOR                                  K
            nan_to_none(rest[18]), # OBSERVDAY_DATE                2022/06/09
            nan_to_none(rest[19])  # OBSERVDAY_DT_TZ      2022/06/09 00:00:00
        )
        sql = 'INSERT INTO {0} VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)'.format(tb_name)
        cur.execute(sql, row)

    conn.commit()

    # cur.execute('SELECT * FROM {0};'.format(tb_name))
    # for row in cur:
    #     my_print(row)

    disconn_db()  # データベースとの接続を切断
# end-of def load_csv


def create_da_table():
    '''ステージング表からDA表を作成'''
    my_print('create_da_table() ステージング表からDA表を作成')

    global data_df
    global weibull_CF, weibull

    # --------------------------------
    # 交換履歴のデータ収集
    # --------------------------------
    conn_db()  # データベースとの接続を作成
    cur.execute('SELECT * FROM {0} ORDER BY INDEX;'.format(tb_name))
    rows = []
    for row in cur:
        rows.append(row)
    my_print('len(rows)={0}'.format(len(rows)))
    columns = [desc[0].upper() for desc in cur.description]
    my_print('columns  ={0}'.format(columns))

    data_df = pd.DataFrame(data = rows, columns = columns)
    my_print('data_df=\n{0}'.format(data_df))
    del rows
    disconn_db()  # データベースとの接続を切断

    # --------------------------------
    # ワイブル分析用データのデータフレーム作成
    # --------------------------------

    # EVENT: '故障', '生存' の2値。データポイントの状態を表す。
    data_df['EVENT'] = ''
    data_df.loc[ data_df['REASON_KEY']=='トラブル', 'EVENT'] = '故障'
    data_df.loc[ data_df['REASON_KEY']=='PM交換'  , 'EVENT'] = '生存'
    data_df.loc[ data_df['REASON_KEY']=='OTHER'   , 'EVENT'] = '生存'

    readcsv_sample = data_df[ ~data_df['CYCLE'].isna() ]  # CYCLEが非NaNを抽出
    readcsv_sample = readcsv_sample.sort_values('CYCLE')
    readcsv_sample.reset_index(drop=True, inplace=True) # indexを再構築

    readcsv_sample['生存数'] = readcsv_sample['CYCLE'].rank(ascending=False) # 平均生存数
    my_print('readcsv_sample=\n{0} {1}'.format(str(readcsv_sample), str(readcsv_sample.shape)))

    # --------------------------------
    # 累積ハザードによるワイブル分析
    # --------------------------------
    # weibull: ワイブル分析用のDF。これは DA表 readcsv_sample を元に作成する。故障カウントを時間tとする。
    weibull = pd.DataFrame({
        'EVENT'  : readcsv_sample['EVENT'],
        'CYCLE'  : readcsv_sample['CYCLE'],
        'PRODUCT': readcsv_sample['PRODUCT'],
    })
    my_print('weibull=\n{0} {1}'.format(str(weibull), str(weibull.shape)))

    # '逆順位': (手順2)着目する時点tの直前の生存数。'生存数' をそのまま用いる。リスクセット数とも。
    weibull['逆順位'] = readcsv_sample['生存数']

    # 'ハザード率': (手順3)ハザード率h(t)推定値。'逆順位' (すなわち生存数) の逆数とする。(生存は除き)故障のみ計算。
    weibull.loc[ weibull['EVENT'] == '故障', 'ハザード率'] = 1/weibull['逆順位']

    # 'H': (手順4) 累積ハザード関数H(t)の推定値。着目する時点tまでのハザード率の累積
    for index, rest in weibull[ weibull['EVENT'] == '故障'].iterrows():
        weibull.loc[index, 'H'] = weibull[ (weibull['EVENT'] == '故障') & (rest['CYCLE'] >= weibull['CYCLE'])]['ハザード率'].sum()
    # weibull["H2"] = weibull["ハザード率"].cumsum() # cumsum()は、CYCLEが同値の行でその数値が異なるので採用しない。

    my_print(weibull)

    my_print('weibull.CYCLE.平均   = {0:,.0f} ({0:.2E})'.format(weibull.CYCLE.mean()))
    my_print('weibull.CYCLE.中央値 = {0:,.0f} ({0:.2E})'.format(weibull.CYCLE.median()))

    # --------------------------------
    # ワイブルパラメータ(最小2乗法)
    # --------------------------------

    # 'F': (手順5) 累積故障率 F(t) = 1-exp(-H(t)) [%] である。
    weibull['F'] = 1-np.exp(-weibull['H'])

    # 直線近似時用およびプロット用列
    # ------------------------------

    # 'log_CYCLE': ln(t)である。直線近似時のX軸座標を与える。
    weibull['log_CYCLE'] = np.log(weibull['CYCLE'])

    # 'log_H': ln(H)である。直線近似時のY軸座標を与える。ハザード確率紙を用いる場合のY軸値。
    weibull['log_H'] = np.log(weibull['H'])

    # 'log_H': (手順5') 累積故障率 ln(H(t)) = ln(ln(1/(1-F(t)))) ワイブル確率紙を用いる場合のY軸値
    # weibull['log_H'] = np.log(np.log(1/(1-weibull['F'])))

    # レンジ外のデータポイントを削除
    # ------------------------------

    # これ以降は近似を行うために不要な生存データを除く
    weibull = weibull.loc[ weibull['EVENT'] == '故障' ]

    # このワイブル分析ではlog_Hが-7以上のデータポイントを使用する。
    weibull = weibull.loc[weibull['log_H'] > -7]

    # 最小2乗法による直線近似を得る
    slope, intercept, r_value, p_value, std_err = stats.linregress(weibull['log_CYCLE'], weibull['log_H'])
    eta = np.exp(-intercept / slope)
    b1  = np.exp((-5 - intercept)/slope)
    b10 = np.exp((-2.250367327 - intercept)/slope)
    b50 = np.exp((-0.366512921 - intercept)/slope)

    my_print('Y切片   = {0:,.2f} ({0:.2E})'.format(intercept))
    my_print('m(形状) = {0:,.2f} ({0:.2E})'.format(slope))
    my_print('η(尺度) = {0:,.0f} ({0:.2E})'.format(eta))
    my_print('b1      = {0:,.0f} ({0:.2E})'.format(b1))
    my_print('b10     = {0:,.0f} ({0:.2E})'.format(b10))
    my_print('b50     = {0:,.0f} ({0:.2E})'.format(b50))

    # 近似線
    # 近似直線の始点と終点を計算
    weibull2 = pd.DataFrame(([np.exp((-7-intercept)/slope), -7], [np.exp((1-intercept)/slope), 1]), columns=['x', 'y'])

    # --------------------------------
    # ワイブルパラメータ(カーブフィット)
    # --------------------------------

    def func_CF(x_CF, a_CF, b_CF, c_CF, d_CF):
        mask_CF = x_CF < c_CF
        result_CF = np.empty(dtype = np.float64, shape = x_CF.shape)
        result_CF[mask_CF] = a_CF * x_CF[mask_CF] + b_CF
        result_CF[~mask_CF] = d_CF * x_CF[~mask_CF] + (a_CF - d_CF) * c_CF + b_CF
        return result_CF
    #カーブフィット
    x_CF = np.log(weibull['CYCLE']) # ln(t)
    y_CF = np.log(weibull['H']) # ln(H)

    # パラメータの初期推定値
    param_ini = (slope, intercept, np.log(b1), slope)
    param_bound = ([0.1, -50, (-5-intercept)/slope, 0.1], [5, 50, np.log(b50) * 10, 5])

    # カーブフィット実行
    # https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html
    result_CF = curve_fit(func_CF, x_CF, y_CF, p0=param_ini, bounds=param_bound)
    # my_print(result_CF[0])

    a_CF = result_CF[0][0] # 結果を取得(a)
    b_CF = result_CF[0][1] # 結果を取得(b)
    c_CF = result_CF[0][2] # 結果を取得(c)
    d_CF = result_CF[0][3] # 結果を取得(d)

    slope_CF1 = a_CF # mを計算(x < c)
    intercept_CF1 = b_CF # 切片を計算(x < c)
    eta_CF1 = np.exp(-intercept_CF1/slope_CF1) # etaを計算(x < c)
    slope_CF2 = d_CF # mを計算(x >= c)
    intercept_CF2 = (a_CF - d_CF) * c_CF + b_CF # 切片を計算(x>=c)
    eta_CF2 = np.exp(-intercept_CF2/slope_CF2) # etaを計算(x>=c)

    # mをセット(CSV出力用)
    weibull_parameter = pd.DataFrame({'m': [slope, slope_CF1, slope_CF2]})
     # etaをセット(CSV出力用)
    weibull_parameter['eta'] = [eta, eta_CF1, eta_CF2]

    if (-2.250367327 - intercept_CF1)/slope_CF1 < c_CF:
        b10_CF = np.exp((-2.250367327-intercept_CF1)/slope_CF1) # B10を計算(x<c)
        weibull_parameter['b10'] = [b10, b10_CF, ''] # B10をセット(CSV出力用)
    else:
        b10_CF = np.exp((-2.250367327-intercept_CF2)/slope_CF2) # B10を計算(x >= c)
        weibull_parameter['b10'] = [b10, '', b10_CF] # B10をセット(CSV出力用)

    if (-0.366512921-intercept_CF1)/slope_CF1 < c_CF:
        b50_CF = np.exp((-0.366512921-intercept_CF1)/slope_CF1) # B50を計算(x < c)
        weibull_parameter['b50'] = [b50, b50_CF, ''] # B50をセット(CSV出力用)
    else:
        b50_CF = np.exp((-0.366512921-intercept_CF2)/slope_CF2) # B50を計算(x >= c)
        weibull_parameter['b50'] = [b50, '', b50_CF] # B50をセット(CSV出力用)

    # 近似線
    if c_CF < (-10-intercept_CF1)/slope_CF1 or c_CF < (-10-intercept_CF2)/slope_CF2: # t=c < 始点
        weibull_CF = pd.DataFrame(([np.exp((-7-intercept_CF2)/slope-CF2), -7], [np.exp((1-intercept_CF2)/slope_CF2), 1]), columns = ['x','y']) # 近似直線の始点と終点を計算
    elif (1-intercept_CF1)/slope_CF1 < c_CF or (1-intercept_CF2)/slope_CF2 < c_CF: # 終点 < t=c
        weibull_CF = pd.DataFrame(([np.exp((-7-intercept_CF1)/slope_CF1),- 7], [np.exp((1-intercept_CF1)/slope_CF1), 1]), columns = ['x','y']) # 近似直線の始点と終点を計算
    else: #始点 < t=c < 終点
        weibull_CF = pd.DataFrame(([np.exp((-7-intercept_CF1)/slope_CF1), -7], [np.exp(c_CF), c_CF * slope_CF1 + intercept_CF1], [np.exp((1-intercept_CF2)/slope_CF2), 1]), columns = ['x','y']) # 近似直線の始点とt=cと終点を計算

    my_print('weibull_CF=\n{}'.format(str(weibull_CF)))

    # CSV出力
    weibull_parameter['point-c'] = ['', 't<' + str(round(np.exp(c_CF), 1)), 't>' + str(round(np.exp(c_CF), 1))] # 変化点cをセット(CSV出力用)
    my_print("weibull_parameter=\n{0}".format(str(weibull_parameter)))
# end-of def create_da_table


def show_weibull_plot():
    ''' ワイブルプロット出力'''
    my_print('show_weibull_plot() ワイブルプロット出力')

    global weibull_CF

    # --------------------------------
    # ワイブルプロット (dash/plotly)
    # --------------------------------

    # コンポーネントのHELP表示
    # help(html.Div)  # dash.html.Div モジュールの Divクラス
    # help(dcc.Graph) # dash.dcc.Graph モジュールの Graphクラス

    # Plotly Expressによるサンプル
    # if False:
    #     # Plotly Express
    #     import plotly.express as px
    #     from dash.dependencies import Input, Output
    #     core_style = {'width': "80%", "margin": "5% auto"}
    #     app = dash.Dash(__name__)
    #     app.layout = html.Div(
    #         [
    #         html.H1("Hello Weibull", style={"textAlign": "center"}),
    #         dcc.Dropdown(
    #             id='my-dropdown',
    #             options=[
    #                 {"label":"white", "value":"white"},
    #                 {"label":"yellow", "value":"yellow"}
    #             ],
    #             value = "A",
    #             style = core_style,
    #         ),
    #         dcc.Graph(
    #             figure = px.scatter(
    #                 weibull,
    #                 x=x, y=y, hover_name="CYCLE",
    #                 # log_x=True, log_y=True
    #             )
    #         ),
    #         ],
    #         id='all-components',
    #     )
    #     @app.callback(
    #         Output("all-components", "style"),
    #         Input("my-dropdown", "value")
    #     )
    #     def update_background(selected_value):
    #         return {"backgroundColor": selected_value, "padding": "3%"}

    # plotly.py によるサンプル
    # import plotly
    # import plotly.graph_objects as go
    # 
    # wplot_trace = go.Scatter(x=x, y=y, name='データポイント', mode='markers')
    # # wplot_fig = go.Figure(data = wplot_trace)
    # # wplot_fig.show()
    # 
    x_line = [weibull_CF.loc[0, 'x'], weibull_CF.loc[1, 'x'], weibull_CF.loc[2, 'x']]
    x_line = np.log(x_line)
    my_print('x_line={}'.format(str(x_line)))
    y_line = [weibull_CF.loc[0, 'y'], weibull_CF.loc[1, 'y'], weibull_CF.loc[2, 'y']]
    my_print('y_line={}'.format(str(y_line)))
    # line_trace = go.Scatter(x=x_line, y=y_line, name='フィッティング')
    # 
    # layout = go.Layout(title='Weibull', width=800, height=800)
    # 
    # line_fig = go.Figure(data = [wplot_trace, line_trace], layout = layout)
    # line_fig.show()

    x_plot = weibull['log_CYCLE']
    my_print('x_plot=\n{0}'.format(str(x_plot)))
    y_plot = weibull['log_H']
    my_print('y_plot=\n{0}'.format(str(y_plot)))
    product_plot = weibull['PRODUCT']
    my_print('product_plot=\n{0}'.format(str(product_plot)))
    
    dashplotlib.plotmain(
        x_plot,
        y_plot,
        product_plot,
        x_line,
        y_line
    )

    my_print('show_weibull_plot() weibull-plotly.pyに戻った')
# end-of def show_weibull_plot
    

def main():
    '''ワイブル分析サンプル開始'''
    my_print('main(): ワイブル分析サンプル開始')
    load_csv()  # CSVファイルをステージング行へロード
    create_da_table()  # ステージング表からDA表を作成
    show_weibull_plot()  # ワイブルプロット出力

if __name__ == "__main__":
    main()

