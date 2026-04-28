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
from scipy.optimize import minimize
from scipy.stats import skew, kurtosis

data = ascii.read("D:\Downloads\Documents\GitHub\Yale_Astro330_LABS\data\hogg_2010_data.txt")
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

def minimize_chi2(clean_x, clean_y, y_err, m_range, b_range, n_points=100):
    '该函数将在你的 m 和 b 值的网格上，使用 chi2() 函数评估 chi2'
    m_grid = np.linspace(m_range[0], m_range[1], n_points) 
    b_grid = np.linspace(b_range[0], b_range[1], n_points)
    image = np.zeros((len(m_grid), len(b_grid)))
    # 记住， m 和 b 值不会直接索引你的数组。所以如果你打算这样做，你想要通过类似 for i,m in enumerate(m_grid): 和 for j,b in enumerate(b_grid): 来循环
    for i,m in enumerate(m_grid):
         for j,b in enumerate(b_grid):
            image[i, j] = chi2(clean_x, clean_y,  m, b, y_err)
    # find min chi2
    min_idx = np.unravel_index(np.argmin(image), image.shape)
    m_min = m_grid[min_idx[0]]
    best_b = b_grid[min_idx[1]]
    
    return m_grid, b_grid, image, m_min, best_b

m_range = [1.8, 2.7]
b_range = [-20, 80]
n_points=100
# ========== 运行网格搜索 ==========
m_grid, b_grid, chi2_image, best_m, best_b = minimize_chi2(
    clean_x, clean_y, y_err, m_range, b_range, n_points=100)
# ========== 计算最小 chi2 ==========
min_chi2 = chi2(clean_x, clean_y, best_m, best_b, y_err)

print('='*50)
print(f'Best fit slope:     {best_m:.2f}')
print(f'Best fit intercept: {best_b:.2f}')
print(f'Minimum chi2:       {min_chi2:.2f}')
print(f'Degrees of freedom: {len(clean_x) - 2}')
print(f'Reduced chi2:       {min_chi2/(len(clean_x) - 2):.2f}')
print('='*50)

# ========== Scipy 优化验证 ==========
def chi2_func(params):
    m, b = params
    return chi2(clean_x, clean_y, m, b, y_err)

result = minimize(chi2_func, x0=[best_m, best_b], method='Nelder-Mead')
m_opt, b_opt = result.x
chi2_opt = result.fun

print(f'\nScipy opt results: m={m_opt:.4f}, b={b_opt:.4f}, χ²={chi2_opt:.4f}')
print(f'Grid search:       m={best_m:.4f}, b={best_b:.4f}, χ²={min_chi2:.4f}')
print('='*50)      

extent = [m_grid.min(), m_grid.max(), b_grid.min(), b_grid.max()]
fig, ax = plt.subplots(figsize=(10, 8))

# 1. χ² 热力图 
# 计算纵横比
im = ax.imshow(chi2_image, origin='lower', extent=extent, 
               aspect='auto', cmap='viridis', vmin=min_chi2, vmax=min_chi2+10)
# 2. 置信区间等高线 
levels = [min_chi2 + 1, min_chi2 + 2.3, min_chi2 + 6.17]
ax.contour(chi2_image, levels=levels, extent=extent, colors='white', linewidths=1.5)
# 3. 标记最佳拟合点
ax.plot(best_m, best_b, 'o', color='blue', markersize=5, label=f'Grid: χ²={min_chi2:.1f}')
ax.plot(m_opt, b_opt, 'o', color='red', markersize=5, label=f'Scipy: χ²={chi2_opt:.1f}')
# 4. 颜色条和标签
plt.colorbar(im, ax=ax, label=r'$\chi^2$')
ax.set_xlabel('Slope (m)')
ax.set_ylabel('Intercept (b)')
ax.set_title('χ² Grid Search')
ax.legend()
ax.grid(True, alpha=0.3, ls='--')

plt.tight_layout()
plt.show()

'Q3'
'''Question 3
Determine the best fit parameters and one-sigma errors from Question 1.2. 
The best-fit value can either be the minimum chi2 value or (bonus) by fitting a function to your chi2 values and interpolating the best fit.

Determine the 1-sigma errors on your best-fit parameters. by noting the surface where chi2 = chi2 +2.3
'''
# 找 Δχ² = 2.3 范围内的所有 (m, b) 点
mask = (chi2_image - min_chi2) <= 2.3

indices = np.where(mask)
m_vals = m_grid[indices[0]]
b_vals = b_grid[indices[1]]

m_err_plus  = m_vals.max() - m_opt
m_err_minus = m_opt - m_vals.min()

b_err_plus  = b_vals.max() - b_opt
b_err_minus = b_opt - b_vals.min()

print('='*50)
print(f'Best fit slope:     {m_opt:.3f} +{m_err_plus:.3f} -{m_err_minus:.3f}')
print(f'Best fit intercept: {b_opt:.3f} +{b_err_plus:.3f} -{b_err_minus:.3f}')
print(f'Minimum chi2:       {chi2_opt:.3f}')
print('='*50)

'Q4:MCMC初步'
'''
Problem 1: 
Look up (or choose) definitions for the mean, variance, skewness, and 
kurtosis of a distribution. Also look up or compute the analytic values of these four statistics for a top-hat (uniform) distribution.
 Write a computer program
that usessome standard package (such as numpy11) to generate K random numbers x from a uniform distribution in the interval 0 < x < 1. 
 Now use those K numbers
to compute a sampling estimate of the mean, variance, skewness, and kurtosis(four estimates; look up definitions as needed). Make four plot of these four
estimates as a function of 1/K or perhaps log2 K, for K = 4n for n = 1 up to n= 10 (that is, K = 4, K = 16, and so on up to K = 1048576). 
Over-plot the analytic answers.
 What can you conclude?
'''

n = np.arange(1, 11)
K_list = 4 ** n
# K = 4n for n = 1 up to n = 10 (that is, K = 4, K = 16, and so on up to K = 1048576).
est_means = []
est_variances = []
est_skewnesses = []
est_kurtosises = []

rng = np.random.default_rng()

for K in K_list:
   
    x = rng.random(K)
    est_means.append(np.mean(x))
    est_variances.append(np.var(x))
    est_skewnesses.append(skew(x))
    est_kurtosises.append(kurtosis(x))
    print(f"Done for K = {K}")

# 解析解
true_mean = 0.5
true_var = 1/12
true_skew = 0.0
true_kurt = -1.2

x_axis = np.log2(K_list) 
# 创建 2x2 的画布
fig, axs = plt.subplots(2, 2, figsize=(12, 10))
# --- 图 1: 均值 ---
axs[0, 0].plot(x_axis, est_means, 'bo-', label='Sample Estimate')
axs[0, 0].axhline(true_mean, color='r', linestyle='--', label=f'Analytic ({true_mean})')
axs[0, 0].set_title('Mean')
axs[0, 0].legend()
# --- 图 2: 方差 ---
axs[0, 1].plot(x_axis, est_variances, 'bo-', label='Sample Estimate')
axs[0, 1].axhline(true_var, color='r', linestyle='--', label=f'Analytic ({true_var:.4f})')
axs[0, 1].set_title('Variance')
axs[0, 1].legend()
# --- 图 3: 偏度 ---
axs[1, 0].plot(x_axis, est_skewnesses, 'bo-', label='Sample Estimate')
axs[1, 0].axhline(true_skew, color='r', linestyle='--', label=f'Analytic ({true_skew})')
axs[1, 0].set_title('Skewness')
axs[1, 0].legend()
# --- 图 4: 峰度 ---
axs[1, 1].plot(x_axis, est_kurtosises, 'bo-', label='Sample Estimate')
axs[1, 1].axhline(true_kurt, color='r', linestyle='--', label=f'Analytic ({true_kurt})')
axs[1, 1].set_title('Excess Kurtosis')
axs[1, 1].legend()
# 设置公共标签
for ax in axs.flat:
    ax.set_xlabel('log2(K)')
    ax.set_ylabel('Value')
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()