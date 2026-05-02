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

'================================================================================'
'Q4 - Problem 1: 采样估计均匀分布的统计量'
'================================================================================'
'''
█ 统计量定义 (for a distribution with PDF f(x)):
─────────────────────────────────────────────────────────────────────────
 1. Mean (均值)     μ = E[X] = ∫ x·f(x) dx
                    描述分布的中心位置

 2. Variance (方差) σ² = E[(X-μ)²] = ∫ (x-μ)²·f(x) dx
                    描述分布的离散程度

 3. Skewness (偏度) γ₁ = E[(X-μ)³] / σ³
                    描述分布的不对称性 (正=右尾长, 负=左尾长)

 4. Excess Kurtosis (超值峰度) γ₂ = E[(X-μ)⁴] / σ⁴ - 3
                    描述分布的尾部厚度 (正=厚尾, 负=薄尾)
                    减3使得正态分布的峰度为0

█ Uniform(0,1) 的解析解 (Analytic Values):
─────────────────────────────────────────────────────────────────────────
 PDF: f(x) = 1  (0 < x < 1), 0 otherwise
 
 E[X]   = ∫₀¹ x·1 dx = [x²/2]₀¹ = 1/2           → Mean     = 0.5
 E[X²]  = ∫₀¹ x²·1 dx = [x³/3]₀¹ = 1/3
 Var(X) = E[X²] - (E[X])² = 1/3 - 1/4 = 1/12     → Variance = 1/12 ≈ 0.08333
 
 E[X³]  = ∫₀¹ x³·1 dx = 1/4
 E[(X-μ)³] = E[X³] - 3μE[X²] + 2μ³
           = 1/4 - 3·(1/2)·(1/3) + 2·(1/8)
           = 1/4 - 1/2 + 1/4 = 0                  → Skewness = 0
 
 E[X⁴]  = ∫₀¹ x⁴·1 dx = 1/5
 E[(X-μ)⁴] = ∫₀¹ (x-½)⁴ dx = 1/80
 Kurtosis (raw) = E[(X-μ)⁴]/σ⁴ = (1/80) / (1/12)² = 144/80 = 1.8
 Excess Kurtosis = 1.8 - 3 = -1.2                → Excess Kurtosis = -1.2
'''
# ==================== 参数设置 ====================
n = np.arange(1, 11)                     # n = 1, 2, ..., 10
K_list = 4 ** n                          # K = 4, 16, 64, ..., 1048576

# ==================== 存储结果 ====================
est_means = []
est_variances = []
est_skewnesses = []
est_kurtosises = []

rng = np.random.default_rng()

# ==================== 采样计算 ====================
print('\n' + '='*75)
print('Running sampling estimates for Uniform(0,1)...')
print('='*75)
print(f'{"K":>8s}  {"Mean":>10s}  {"Variance":>10s}  {"Skewness":>10s}  {"Ex.Kurtosis":>10s}')
print('-' * 60)

for K in K_list:
    x = rng.random(K)                    # 从 Uniform(0,1) 生成 K 个随机数
    
    # 使用 numpy 计算样本均值、方差（无偏样本方差 ddof=1）
    sample_mean = np.mean(x)
    sample_var  = np.var(x, ddof=1)       # ddof=1 使用无偏样本方差 (除以 K-1)
    
    # 使用 scipy.stats 计算偏度和峰度 (bias=False 返回无偏估计)
    sample_skew = skew(x, bias=False)
    sample_kurt = kurtosis(x, bias=False) # 默认返回 excess kurtosis (Fisher)
    
    est_means.append(sample_mean)
    est_variances.append(sample_var)
    est_skewnesses.append(sample_skew)
    est_kurtosises.append(sample_kurt)
    
    print(f'{K:8d}  {sample_mean:10.6f}  {sample_var:10.6f}  {sample_skew:+10.6f}  {sample_kurt:+10.6f}')

# ==================== 解析解 (Theoretical Values) ====================
true_mean = 0.5          # μ = 1/2
true_var  = 1/12         # σ² = 1/12 ≈ 0.08333
true_skew = 0.0          # γ₁ = 0 (对称分布)
true_kurt = -1.2         # Excess Kurtosis = -6/5 = -1.2

# ==================== 图 1: 以 log2(K) 为 x 轴 ====================
x_axis_logK = np.log2(K_list)            # log₂(K)

fig1, axs = plt.subplots(2, 2, figsize=(14, 10))
fig1.suptitle('Sampling Estimates of Uniform(0,1) Statistics vs log\u2082(K)', fontsize=14)

# --- 图 1a: Mean ---
axs[0, 0].plot(x_axis_logK, est_means, 'bo-', markersize=6, label='Sample Estimate')
axs[0, 0].axhline(true_mean, color='r', linestyle='--', linewidth=2,
                  label=f'Analytic = {true_mean}')
axs[0, 0].fill_between(x_axis_logK, true_mean - 3/np.sqrt(K_list), true_mean + 3/np.sqrt(K_list),
                       alpha=0.15, color='red', label='\u00b13/\u221aK (CLT)')
axs[0, 0].set_title('Mean (\u03bc) \u2014 converges as 1/\u221aK')
axs[0, 0].set_ylabel('Mean')
axs[0, 0].legend(fontsize=8)
axs[0, 0].grid(True, alpha=0.3)

# --- 图 1b: Variance ---
axs[0, 1].plot(x_axis_logK, est_variances, 'bo-', markersize=6, label='Sample Estimate')
axs[0, 1].axhline(true_var, color='r', linestyle='--', linewidth=2,
                  label=f'Analytic = {true_var:.4f}')
axs[0, 1].set_title('Variance (\u03c3\u00b2)')
axs[0, 1].set_ylabel('Variance')
axs[0, 1].legend(fontsize=8)
axs[0, 1].grid(True, alpha=0.3)

# --- 图 1c: Skewness ---
axs[1, 0].plot(x_axis_logK, est_skewnesses, 'bo-', markersize=6, label='Sample Estimate')
axs[1, 0].axhline(true_skew, color='r', linestyle='--', linewidth=2,
                  label=f'Analytic = {true_skew}')
axs[1, 0].set_title('Skewness (\u03b3\u2081) \u2014 symmetric distribution')
axs[1, 0].set_xlabel('log\u2082(K)')
axs[1, 0].set_ylabel('Skewness')
axs[1, 0].legend(fontsize=8)
axs[1, 0].grid(True, alpha=0.3)

# --- 图 1d: Excess Kurtosis ---
axs[1, 1].plot(x_axis_logK, est_kurtosises, 'bo-', markersize=6, label='Sample Estimate')
axs[1, 1].axhline(true_kurt, color='r', linestyle='--', linewidth=2,
                  label=f'Analytic = {true_kurt}')
axs[1, 1].set_title('Excess Kurtosis (\u03b3\u2082) \u2014 lighter tails than Gaussian')
axs[1, 1].set_xlabel('log\u2082(K)')
axs[1, 1].set_ylabel('Excess Kurtosis')
axs[1, 1].legend(fontsize=8)
axs[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/LAB4/Q4_sampling_vs_logK.png', dpi=150, bbox_inches='tight')
plt.show()
print('[Saved] outputs/LAB4/Q4_sampling_vs_logK.png')

# ==================== 图 2: 以 1/K 为 x 轴 ====================
x_axis_invK = 1.0 / K_list               # 1/K

fig2, axs2 = plt.subplots(2, 2, figsize=(14, 10))
fig2.suptitle('Sampling Estimates of Uniform(0,1) Statistics vs 1/K', fontsize=14)

# --- 图 2a: Mean ---
axs2[0, 0].plot(x_axis_invK, est_means, 'bo-', markersize=6, label='Sample Estimate')
axs2[0, 0].axhline(true_mean, color='r', linestyle='--', linewidth=2,
                   label=f'Analytic = {true_mean}')
axs2[0, 0].set_title('Mean (\u03bc)')
axs2[0, 0].set_ylabel('Mean')
axs2[0, 0].set_xlabel('1/K')
axs2[0, 0].legend(fontsize=8)
axs2[0, 0].grid(True, alpha=0.3)
axs2[0, 0].set_xlim(x_axis_invK.max() * 1.1, -0.001)

# --- 图 2b: Variance ---
axs2[0, 1].plot(x_axis_invK, est_variances, 'bo-', markersize=6, label='Sample Estimate')
axs2[0, 1].axhline(true_var, color='r', linestyle='--', linewidth=2,
                   label=f'Analytic = {true_var:.4f}')
axs2[0, 1].set_title('Variance (\u03c3\u00b2)')
axs2[0, 1].set_ylabel('Variance')
axs2[0, 1].set_xlabel('1/K')
axs2[0, 1].legend(fontsize=8)
axs2[0, 1].grid(True, alpha=0.3)
axs2[0, 1].set_xlim(x_axis_invK.max() * 1.1, -0.001)

# --- 图 2c: Skewness ---
axs2[1, 0].plot(x_axis_invK, est_skewnesses, 'bo-', markersize=6, label='Sample Estimate')
axs2[1, 0].axhline(true_skew, color='r', linestyle='--', linewidth=2,
                   label=f'Analytic = {true_skew}')
axs2[1, 0].set_title('Skewness (\u03b3\u2081)')
axs2[1, 0].set_ylabel('Skewness')
axs2[1, 0].set_xlabel('1/K')
axs2[1, 0].legend(fontsize=8)
axs2[1, 0].grid(True, alpha=0.3)
axs2[1, 0].set_xlim(x_axis_invK.max() * 1.1, -0.001)

# --- 图 2d: Excess Kurtosis ---
axs2[1, 1].plot(x_axis_invK, est_kurtosises, 'bo-', markersize=6, label='Sample Estimate')
axs2[1, 1].axhline(true_kurt, color='r', linestyle='--', linewidth=2,
                   label=f'Analytic = {true_kurt}')
axs2[1, 1].set_title('Excess Kurtosis (\u03b3\u2082)')
axs2[1, 1].set_ylabel('Excess Kurtosis')
axs2[1, 1].set_xlabel('1/K')
axs2[1, 1].legend(fontsize=8)
axs2[1, 1].grid(True, alpha=0.3)
axs2[1, 1].set_xlim(x_axis_invK.max() * 1.1, -0.001)

plt.tight_layout()
plt.savefig('outputs/LAB4/Q4_sampling_vs_invK.png', dpi=150, bbox_inches='tight')
plt.show()
print('[Saved] outputs/LAB4/Q4_sampling_vs_invK.png')

# ==================== 结论输出 ====================
print('\n' + '=' * 75)
print('CONCLUSIONS (结论)')
print('=' * 75)
print('''
 1. Mean (均值):
    - 随着 K 增大，样本均值迅速收敛到真值 μ = 0.5
    - 根据中心极限定理 (CLT)，标准误 σ_mean = σ/√K = √(1/12)/√K ≈ 0.2887/√K
    - 当 K = 10⁶ 时，标准误仅约 0.00029，估计非常精确
    - 在 1/K 图中表现为：K → ∞ 时均值 → 0.5

 2. Variance (方差):
    - 样本方差收敛到 σ² = 1/12 ≈ 0.08333
    - 使用 ddof=1 (无偏估计) 消除了小样本中的系统偏差
    - 方差估计的收敛速度也随 1/√K 下降

 3. Skewness (偏度):
    - Uniform(0,1) 是对称分布，理论偏度 γ₁ = 0
    - 小样本时 (如 K=4) 偏度估计波动非常大
    - K ≥ 10³ 后，偏度估计稳定在 0 附近
    - 高阶矩 (3阶) 需要更多样本才能收敛

 4. Excess Kurtosis (超值峰度):
    - Uniform 分布尾部比正态分布薄，γ₂ = -1.2 < 0
    - 这是 4 个统计量中收敛最慢的
    - K=4 时样本估计可能严重偏离真值
    - K ≥ 10⁴ 后才较为稳定

 5. Overall Conclusions (总体结论):
    - 所有样本统计量都随着 K → ∞ 收敛到解析值
    - 收敛速度排序：均值 > 方差 > 偏度 > 峰度
      (低阶矩比高阶矩收敛更快，需要的样本量更少)
    - log₂(K) 图能更清晰地展示大范围 K 的变化趋势
    - 1/K 图能直观显示"当 K 很大时，估计趋近真值"
    - 实际应用中，样本量 K 的选择取决于需要估计的统计量阶数
      和所需的精度
''')
print('=' * 75)

'================================================================================'
'Q4 - Problem 2: Metropolis-Hastings MCMC 采样器'
'================================================================================'
'''
Problem 2:
Write a very simple M-H MCMC sampler. Sample in a single parameter x and give the
sampler as its density function p(x) a Gaussian density with mean 2 and variance 2.
(Note that variance is the square of the standard deviation.)
Give the sampler a proposal distribution q(x' | x) a Gaussian pdf for x' with mean x
and variance 1. Initialize the sampler with x = 0 and run for more than 10^4 steps.
Plot the results as a histogram with the true density over-plotted sensibly.
'''
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ==================== 目标分布 ====================
# p(x) ~ N(μ=2, σ²=2), 即 σ = √2
true_mu = 2.0
true_sigma = np.sqrt(2)

def target_pdf(x):
    """目标分布: Gaussian(2, 2) 的概率密度"""
    return norm.pdf(x, loc=true_mu, scale=true_sigma)

# ==================== M-H 采样器 ====================
def mh_sampler(n_steps, x_init=0.0, proposal_scale=1.0):
    """
    Metropolis-Hastings MCMC 采样器
    - 目标分布: N(2, 2)
    - 提议分布: N(x_current, proposal_scale²)
    - 由于提议分布对称 (q(x'|x) = q(x|x')),
      M-H 简化为 Metropolis 算法: α = min(1, p(x')/p(x))
    """
    samples = np.zeros(n_steps)
    x_current = x_init
    n_accepted = 0

    for i in range(n_steps):
        # 1. 从提议分布采样
        x_proposal = np.random.normal(x_current, proposal_scale)

        # 2. 计算接受率 (提议对称, 提议比 = 1)
        p_current = target_pdf(x_current)
        p_proposal = target_pdf(x_proposal)

        # 安全处理: 若当前概率为 0 则强制接受
        if p_current == 0:
            alpha = 1.0
        else:
            alpha = min(1.0, p_proposal / p_current)

        # 3. 接受或拒绝
        if np.random.random() < alpha:
            x_current = x_proposal
            n_accepted += 1

        samples[i] = x_current

    return samples, n_accepted

# ==================== 运行采样 ====================
N = 15000                     # > 10⁴ 步
burn_in = 5000                # 燃烧期 (burn-in)

print('\n' + '=' * 75)
print('Running M-H MCMC sampler...')
print(f'Target:    N(μ={true_mu}, σ²={true_sigma**2})')
print(f'Proposal:  N(x_current, σ²=1)')
print(f'Initial:   x = 0')
print(f'Steps:     {N} (burn-in: {burn_in})')
print('=' * 75)

samples, n_acc = mh_sampler(N, x_init=0.0, proposal_scale=1.0)
samples_after_burnin = samples[burn_in:]
acceptance_rate = n_acc / N * 100

# ==================== 绘图: 1x2 子图 (轨迹 + 直方图) ====================
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('M-H MCMC Sampler for N(\u03bc=2, \u03c3\u00b2=2)', fontsize=14)

# ---- 图 1: 轨迹图 (Trace Plot) ----
axes[0].plot(samples, alpha=0.6, linewidth=0.3, color='steelblue')
axes[0].axvline(burn_in, color='red', linestyle='--', linewidth=1.5,
                label=f'Burn-in ({burn_in} steps)', alpha=0.7)
axes[0].axhline(true_mu, color='green', linestyle='-', linewidth=1.5,
                label=f'True \u03bc = {true_mu}')
axes[0].set_xlabel('Step')
axes[0].set_ylabel('x')
axes[0].set_title(f'Trace Plot\nAcceptance Rate: {acceptance_rate:.1f}%')
axes[0].legend(fontsize=8, loc='upper right')
axes[0].grid(True, alpha=0.3)

# ---- 图 2: 直方图 + 真实密度 ----
axes[1].hist(samples_after_burnin, bins=60, density=True, alpha=0.6,
             color='steelblue', edgecolor='white',
             label=f'MCMC Samples (n={len(samples_after_burnin)})')

x_grid = np.linspace(-4, 8, 500)
axes[1].plot(x_grid, target_pdf(x_grid), 'r-', linewidth=2.5,
             label=f'True: N(\u03bc={true_mu}, \u03c3\u00b2={true_sigma**2:.0f})')
axes[1].set_xlabel('x')
axes[1].set_ylabel('Probability Density')
axes[1].set_title('Histogram vs True Distribution')
axes[1].legend(fontsize=9)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/LAB4/Q4_MH_MCMC.png', dpi=150, bbox_inches='tight')
plt.show()
print('[Saved] outputs/LAB4/Q4_MH_MCMC.png')

# ==================== 统计摘要 ====================
sample_mean = np.mean(samples_after_burnin)
sample_var  = np.var(samples_after_burnin)
sample_std  = np.std(samples_after_burnin)

print('\n' + '=' * 75)
print('Results (after burn-in):')
print('=' * 75)
print(f'  Sample mean:     {sample_mean:.4f}  (true: {true_mu})')
print(f'  Sample std:      {sample_std:.4f}  (true: {true_sigma:.4f})')
print(f'  Sample variance: {sample_var:.4f}  (true: {true_sigma**2:.4f})')
print(f'  Acceptance rate: {acceptance_rate:.1f}%')
print('=' * 75)
