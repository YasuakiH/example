# -*- mode: Python; -*-
# both_logarithmic_axials_plot_in_matplotlib.py

The MIT License (MIT)
Copyright (C) 2023 YasuakiH

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

# このデモの内容
# 散布図とヒストグラムの2つのチャートを1つの両対数グラフに重ねる

# 引用
# [1] Setting both axes logarithmic in bar plot matploblib
# https://stackoverflow.com/questions/44068435/setting-both-axes-logarithmic-in-bar-plot-matploblib
# [2] How do I overlay multiple plot types (bar + scatter) in one figure, sharing x-axis
# https://stackoverflow.com/questions/57262939/how-do-i-overlay-multiple-plot-types-bar-scatter-in-one-figure-sharing-x-ax

import numpy as np; np.random.seed(1)
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from PIL import Image

# 散布図の描画

npoints=100
xRange=np.arange(0,npoints,1)
# randomdata0=np.abs(np.random.normal(0,1,npoints))
randomdata1=np.random.normal(10,1,npoints)
axtick=[7,10,14]
# ax2tick=[0,1.5,3]

fig0=plt.figure(0)
ax=fig0.gca()
# ax2=ax.twinx()
sns.scatterplot(x=xRange, y=randomdata1, ax=ax)
ax.set_yticks(axtick)
ax.set_ylim([5,20])
# ax2.set_yticks(ax2tick)
# ax2.set_ylim([0,3.5])

# plt.xticks([])
plt.xscale("log")
plt.yscale("log")


canvas0 = FigureCanvas(fig0)
s, (width, height) = canvas0.print_to_buffer()
X0 = Image.frombytes("RGBA", (width, height), s) #Contains the data of the first plot

# ヒストグラムの描画

x = np.logspace(0, 5, num=21)
y = (np.sin(1.e-2*(x[:-1]-20))+3)**10

fig1, ax = plt.subplots()
ax.bar(x[:-1], y, width=np.diff(x), log=True, ec="k", align="edge")
ax.set_xscale("log")

canvas1 = FigureCanvas(fig1)
s, (width, height) = canvas1.print_to_buffer()
X1 = Image.frombytes("RGBA", (width, height), s) #Contains the data of the second plot

plt.figure(13,figsize=(10,10))
plt.imshow(Image.blend(X0,X1,0.5),interpolation='gaussian')
Axes=plt.gca()
Axes.spines['top'].set_visible(False)
Axes.spines['right'].set_visible(False)
Axes.spines['bottom'].set_visible(False)
Axes.spines['left'].set_visible(False)
Axes.set_xticks([])
Axes.set_yticks([])

plt.show()
