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
import matplotlib
import matplotlib.pyplot as plt


data = ascii.read("D:\Documents\GitHub\Yale_Astro330_LABS\data\hogg_2010_data.txt")
# 根据论文内容，移除前四个异常值数据
cleandata = data[data['ID']>4]
print(cleandata)
# 返回：最佳拟合值（斜率，截距）及其误差
clean_x = cleandata['x']
clean_y = cleandata['y']
y_err = cleandata['sigma_y']

# 返回拟合系数p = [斜率, 截距]， pcov： 协方差矩阵
p, pcov = np.polyfit(clean_x, clean_y, 1, w=1./y_err, cov='unscaled')
# 假设没有偏轴协方差，参数不确定性是矩阵对角项的平方根（对于线性拟合将有 2 个。你可以使用 np.diag() 从一个正方形数组中提取对角项）
diagonals = np.diag(pcov)
perr = np.sqrt(diagonals)

print('Best fit slope:     {:0.2f} +/- {:0.2f} '.format(p[0],perr[0]))
print('Best fit intercept: {:0.1f} +/- {:0.1f} '.format(p[1],perr[1]))

fig, ax = plt.subplots(figsize=(10,8))
pfit = np.poly1d(p)
x=np.arange(0,300)
plt.plot(x,pfit(x),label='Fit using np.polyfit')
plt.errorbar(data['x'], data['y'], yerr=data['sigma_y'], 
             fmt='.', label='Data', ms=15)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.show()

'Q2'
# for y_pred = m*x + b:
# χ²(m,b) = Σ [(y_i - (m*x_i + b)) / σ_i]²
def chi2(clean_x, clean_y, m, b, y_err):
    '返回值: chi2_value: 卡方'
    residuals = clean_y - (m * clean_x + b)
    chi2 = np.sum((residuals / y_err) ** 2)
    return chi2

def minimize_chi2(clean_x, cleany, y_err, m_range, b_range, n_points=100):
    '该函数将在你的 m 和 b 值的网格上，使用 chi2() 函数评估 chi2'
    m_grid = np.linspace(m_range[0], m_range[1], n_points)
    b_grid = np.linspace(b_range[0], b_range[1], n_points)
    chi2_image = np.zeros((len(m_grid), len(b_grid)))
    # 记住， m 和 b 值不会直接索引你的数组。所以如果你打算这样做，你想要通过类似 for i,m in enumerate(m_grid): 和 for j,b in enumerate(b_grid): 来循环
    for i,m in enumerate(m_grid):
         for j,b in enumerate(b_grid):
             chi2_image[i, j] = chi2(m, b, clean_x, clean_y, y_err)
    # find min chi2
    min_idx = np.unravel_index(np.argmin(chi2_image), chi2_image.shape)
    best_m = m_grid[min_idx[0]]
    best_b = b_grid[min_idx[1]]
    
    return m_grid, b_grid, chi2_image, best_m, best_b