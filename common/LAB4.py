'''在本实验中，我们将开始构建各种工具来将模型拟合到数据。目标是理解、实现并比较 chi2 和 MCMC 拟合例程以对假数据进行拟合。
在实验 5 中，我们将应用这些工具到来自 Keck / DEIMOS 的真实光谱数据。
实验的目标是：
使用 Python 模块对数据进行 χ2 拟合
编写并与你自己的 χ2 算法进行比较
探索采样算法
编写一个 MCMC 算法，并与 χ2 结果进行比较
'''

from astropy.io import fits
from astropy.io import ascii
import numpy as np
import matplotlib as plt


data = ascii.read("D:\Documents\GitHub\Yale_Astro330_LABS\data\hogg_2010_data.txt")
# 根据论文内容，移除前四个异常值数据
cleandata = data[data['ID']>4]
print(cleandata)

x = cleandata['x']
y = cleandata['y']
y_err = cleandata['σy']