#! python app.py
# https://dash.plotly.com/layout

'''
The MIT License (MIT)
Copyright (C) 2022 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
'''


from dash import Dash, html, dcc  # dcc: Dash Core Components
import plotly.express as px
import pandas as pd

app = Dash(__name__)

# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
df = pd.DataFrame({
    "フルーツ": ["りんご", "オレンジ", "バナナ", "りんご", "オレンジ", "バナナ"],
    "量": [4, 1, 2, 2, 4, 5],
    "都市": ["サンフランシスコ", "サンフランシスコ", "サンフランシスコ", "モントリオール", "モントリオール", "モントリオール"]
})

fig = px.bar(df, x="フルーツ", y="量", color="都市", barmode="group")

app.layout = html.Div(children=[
    html.H1(children='Dash, ダッシュ, だっしゅ'),

    html.Div(children='''
        Dash: webアプリのフレームワーク for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)
