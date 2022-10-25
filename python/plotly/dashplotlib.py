# dashplotnew.py

'''
The MIT License (MIT)
Copyright (C) 2022 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''

'''
これは単独では動作しない。別のプログラム weibull-plotly.py で import して用いる。

リファレンス:
Log Plots in Python <https://plotly.com/python/log-plot/>
Axes in Python <https://plotly.com/python/axes/>
Line Charts in Python <https://plotly.com/python/line-charts/>

Plotly Log Scale in Subplot Python 一部のサブプロットで対数軸にする
https://stackoverflow.com/questions/61041707/plotly-log-scale-in-subplot-python

参考:
plotly-dash-book <https://github.com/plotly-dash-book/plotly-dash-book/blob/master/ch07_dash_callback/dash_callback_chained.py>
Plotly 基本チャート - 散布図 <http://hxn.blog.jp/archives/10842159.html>
Plotly Scatter(散布図・折れ線グラフ)のモード設定 <https://ai-research-collection.com/plotly-scatter-mode/>
'''
import dash
from dash import dcc
from dash import html
import plotly.express as px
from dash.dependencies import Input, Output
import numpy as np
import pandas as pd

def my_print(msg):
    print('dashplotnew: {0}'.format(msg))

def plotmain(x_plot, y_plot, product_plot, x_line, y_line):
    my_print('len(x_plot)={0}, len(x_line)={1}'.format(len(x_plot), len(x_line)))

    # データの読み込み
    # df = px.data.df()
    df = pd.DataFrame(
        data = {'log_CYCLE':x_plot, 'log_H':y_plot, 'PRODUCT':product_plot}
    )
    my_print('df=\n{0} {1}'.format(str(df), str(df.shape)))
    weibull_CF = pd.DataFrame(
        data = {'x':x_line, 'y':y_line}
    )
    my_print('weibull_CF=\n{0}'.format(str(weibull_CF)))

    dash_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]
    app = dash.Dash(__name__, external_stylesheets=dash_stylesheets)

    # (1) レイアウトの作成
    app.layout = html.Div(
        [
            # (3) 指定したグラフに応じてグラフのタイトルを変更
            html.H3(id="title", style={"textAlign": "center"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("機種選択"),
                            # (4) 機種選択のドロップダウンの作成
                            dcc.Dropdown(
                                id="product_selector",
                                options=[
                                    {"value": product, "label": product}
                                    for product in df.PRODUCT.unique()
                                ],
                                multi=True,
                                value = [product for product in df.PRODUCT.unique()],
                            ),
                        ],
                        className="six columns",
                    ),
                    html.Div(
                        [
                            html.H4("グラフ選択"),
                            # (5) グラフ選択のドロップダウンの作成
                            dcc.Dropdown(
                                id="graph_selector",
                                options=[
                                    {"value": "bar", "label": "bar"},
                                    {"value": "scatter", "label": "scatter"},
                                ],
                                value="scatter",
                            ),
                        ],
                        className="six columns",
                    ),
                ],
                style={"padding": "2%", "margin": "auto"},
            ),
            # (6) グラフの表示場所
            html.Div(
                [
                    dcc.Graph(id="app_graph", style={"padding": "3%"}),
                ],
                style={"padding": "3%", "marginTop": 50},
            ),
        ]
    )

    # (2) コールバックの作成
    @app.callback(
        # (7) Outputインスタンス,Inputインスタンスの順に配置
        Output("title", "children"),
        Output("app_graph", "figure"),
        Input("product_selector", "value"),
        Input("graph_selector", "value"),
    )
    def update_graph(selected_products, selected_graph):
        my_print('update_graph() selected_products={0}, selected_graph={1}'.format(selected_products, selected_graph))
        # (8) データフレームの作成
        selected_df = df.loc[ df["PRODUCT"].isin(selected_products) ]
        my_print('selected_df=\n{0} {1}'.format(str(selected_df), str(selected_df.shape)))
        # (9) 選択されたグラフの種類により、タイトル表示データとグラフを作成
        if selected_graph == "scatter":
            title = "テーブル毎データ（散布図）"
            figure = px.scatter(
                selected_df, x="log_CYCLE", y="log_H", color="PRODUCT", height=600
            )
            return title, figure
        else:
            # if False:
            #     title = ("機種ごとの売り上げ（棒グラフ）",)
            #     figure = px.bar(selected_df, x="log_CYCLE", y="log_H", height=600)
            # else:

            title = "テーブル毎データ（棒グラフ）"
            import plotly.graph_objects as go
            # wplot_trace1 = go.Scatter(
            #     x=df['log_CYCLE'],
            #     y=df['log_H'],
            #     name='データポイント',
            #     mode='markers')
            scatter_fig = go.Figure()
            for product in df['PRODUCT'].unique():
                my_print('product={0}'.format(product))
                plot_df = selected_df[ selected_df['PRODUCT'] == product]
                my_print('product={0}, plot_df=\n{1}'.format(product, str(plot_df)))
                scatter_fig.add_trace(
                    go.Scatter(
                        x = plot_df['log_CYCLE'],
                        y = plot_df['log_H'],
                        name = product,
                        mode = 'markers'
                    )
                )
            x_line = [weibull_CF.loc[0, 'x'], weibull_CF.loc[1, 'x'], weibull_CF.loc[2, 'x']]
            # x_line = np.log(x_line)
            my_print('x_line={0}'.format(str(x_line)))
            y_line = [weibull_CF.loc[0, 'y'], weibull_CF.loc[1, 'y'], weibull_CF.loc[2, 'y']]
            my_print('y_line={0}'.format(str(y_line)))
            # line_trace = go.Scatter(x=x_line, y=y_line, name='フィッティング')
            # layout = go.Layout(title='Weibull', width=800, height=800)
            # line_fig = go.Figure(data = [scatter_fig, line_trace], layout = layout)
            scatter_fig.add_trace(
                go.Scatter(
                    x=x_line,
                    y=y_line,
                    name='フィッティング',
                    mode='lines'
                )
            )
            return title, scatter_fig

    app.run_server(debug=True)
