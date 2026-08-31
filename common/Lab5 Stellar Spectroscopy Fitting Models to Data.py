"""Lab 5: Stellar Spectroscopy + Fitting Models to Data

当前进度是 Q3-Q6：先观察恒星光谱，再用合成模板和 chi-square 测量视向速度。
"""

"""在本实验中，我们将指导您读取、绘制和拟合银河系球状星团中恒星的光谱。
科学目标是确定少数恒星的速度及其误差，从而判断它们是球状星团的成员，还是银河系的前景恒星。
编程目标是在模型较为复杂时，应用 χ2 拟合和 MCMC 拟合技术。"""

"""Goals of this lab:  本实验的目标：¶
Explore a maintained software package (pypeit).
探索一个维护良好的软件包 pypeit。
Read a complicated fits file and plot a spectrum.
读取复杂的 fits 文件并绘制光谱图。
Find parameters and errors via chi2 fitting when the model is not an analytic function
当模型不是解析函数时，通过卡方拟合找到参数和误差
Find parameters and errors via MCMC.
利用 MCMC 方法查找参数和误差。
Fitting polynomials to 2D surfaces, corner plots
将多项式拟合到二维曲面，角图"""

import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

# 优先使用 Windows 中文字体，避免新增的中文图标题显示成方框。
# 当前 lab5_figure 总标题已改为英文；字体设置仍保留，方便其他中文图中文字正常显示。
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OBSERVED_FILE = r"D:\Documents\GitHub\Yale_Astro330_LABS\data\lab5_data\spec1d_DE.20110603.45055-n7006a_DEIMOS_2011Jun03T123053.021.fits"
TEMPLATE_FILE = r"D:\Documents\GitHub\Yale_Astro330_LABS\data\lab5_data\dmost_lte_5000_3.0_-2.0_.fits"

STAR_IDS = [121, 135, 157]
CAII_REST_WAVELENGTHS = [8500.36, 8544.44, 8664.52]
C_KMS = 2.997924e5  # 光速，单位 km/s；速度也使用 km/s。

# 这个 FITS 文件里有很多 extension；每颗星对应其中一个 extension。
observed_hdu = fits.open(OBSERVED_FILE)

figure_number = 1


# %% Q3：先看数据，并用 Ca II 吸收线目测速度

'Q3:Question 3: Plotting 1D PypeIt output spectra and fitting by eye'

for star_id in STAR_IDS:
    spectrum = observed_hdu[star_id].data

    fig, ax = plt.subplots(figsize=(15, 3))
    fig.suptitle(
        f"lab5_figure{figure_number}_Q3_star{star_id}_observed_spectrum"
    )
    ax.plot(spectrum["OPT_WAVE"], spectrum["OPT_COUNTS"])
    ax.set_xlim(8300, 8850)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Counts")
    ax.set_title(f"Star ID: {star_id}")
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    plt.show()
    figure_number += 1

'''为了大致了解每颗恒星的速度，你可以尝试“目测”粗略估计一下。上面光谱中最强的三条谱线来自 CaII: 8500.36、8544.44 和 8664.52 埃。
你估计它们的速度是多少？'''

# 虚线是 Ca II 的静止波长，吸收线实际位置与它的偏移量可以给出一个粗略速度
EXAMPLE_STAR_ID = 121
example_spectrum = observed_hdu[EXAMPLE_STAR_ID].data
example_wave = example_spectrum["OPT_WAVE"]
example_flux = example_spectrum["OPT_COUNTS"]

for rest_wavelength in CAII_REST_WAVELENGTHS:
    fig, ax = plt.subplots(figsize=(8, 3))
    fig.suptitle(
        f"lab5_figure{figure_number}_Q3_star{EXAMPLE_STAR_ID}_"
        f"CaII_{rest_wavelength:.2f}Angstrom_absorption_line"
    )
    ax.plot(example_wave, example_flux)
    ax.axvline(rest_wavelength, color="black", linestyle="--", label="Rest wavelength")
    ax.set_xlim(rest_wavelength - 25, rest_wavelength + 10)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Counts")
    ax.set_title(f"Star {EXAMPLE_STAR_ID}: Ca II near {rest_wavelength:.2f} Å")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.subplots_adjust(top=0.80)
    plt.show()
    figure_number += 1


# %% Q4：读取合成模板

'''Question 4: Synthetic model spectra'''
'''可以通过测量已知吸收线的中心（目测或拟合高斯函数）并将其与恒星的静止波长进行比较，来测量恒星的速度。
虽然这种方法可以估算恒星的速度，但它浪费了完整光谱中的大部分信息。为了确定更精确的速度，我们采用“模板拟合”方法，
即将已知速度的光谱与待测光谱进行比较。
模板光谱可以是经验性的（已知速度的标准恒星的观测光谱），也可以是合成的（根据恒星模型数值计算得出）。
这里我们将使用来自 PHEONIX 库的合成模板:https://phoenix.astro.physik.uni-goettingen.de/'''

# 目测只使用几条谱线；模板拟合则会利用整个波段的信息。
def read_synthetic_spectrum(file_path):
    """读取 PHOENIX 模板，返回真空波长和模板通量。"""
    with fits.open(file_path) as template_hdu:
        template_table = template_hdu[1].data

    template_flux = np.asarray(template_table["flux"]).flatten()

    # 文件保存的是空气波长的自然对数，先还原空气波长。
    air_wavelength = np.exp(np.asarray(template_table["wave"]).flatten())

    # CONVERTING AIR WAVELENGTHS TO VACUUM
    # 观测数据使用真空波长，因此按课程给出的公式把空气波长转换成真空波长。
    s = 1e4 / air_wavelength
    refractive_index = (
        1
        + 0.00008336624212083
        + 0.02408926869968 / (130.1065924522 - s**2)
        + 0.0001599740894897 / (38.92568793293 - s**2)
    )
    vacuum_wavelength = air_wavelength * refractive_index

    return vacuum_wavelength, template_flux


# %% Q5：把理论模板变成能与观测光谱直接比较的模型

'''Question 5: Smoothing and continuum fitting
把理论模板处理成和观测光谱可以直接比较的模型。
课程思路：先平滑模板，再把模板插值到观测波长，最后匹配连续谱。
'''

def smooth_spectrum(flux, sigma=25):
    """把高分辨率合成光谱平滑到 DEIMOS 的分辨率。

    把模板中过窄的谱线平滑到 DEIMOS 能分辨的宽度。
    """
    # 课程给出的宽度是 0.5 Å，而模板每个像素是 0.02 Å：0.5 / 0.02 = 25。
    # DEIMOS 的线宽约 0.5 Å，模板每个像素约 0.02 Å，所以 sigma = 0.5 / 0.02 = 25。
    return gaussian_filter1d(flux, sigma, truncate=3)


def fit_continuum(template_wave, template_flux, data_wave, data_flux):
    """将模板重采样到观测波长，并调整到与观测 counts 相同的连续谱。

    让模板和观测数据使用相同的波长点、相同的 counts 尺度。
    """
    # 第一步：rebin。让模板和观测数据拥有完全相同的波长位置。
    # 模板和数据的波长采样不同。插值以后，两条光谱才能逐点比较。
    template_on_data_grid = np.interp(
        data_wave,
        template_wave,
        template_flux,
        left=np.nan,
        right=np.nan,
    )

    # 第二步：观测 / 模板。吸收线以外的整体变化就是需要拟合的 continuum。
    # 模板通常在 1 附近，观测值却是 counts。data/template 给出需要乘回模板的
    # 缓慢变化背景，也就是这里要拟合的 continuum。
    response = data_flux / template_on_data_grid

    # 第三步：照课程答案去掉最低 20% 和最高 1%，避免吸收线和离群点影响拟合。
    # 很低的点多半是吸收线，很高的点可能是坏像素；都不适合决定背景形状。
    good = np.isfinite(response)
    low, high = np.percentile(response[good], [20, 99])
    good &= (response > low) & (response < high)

    # 第四步：用一次多项式（直线）拟合 continuum，并乘回模板得到最终模型。
    # 这个波段的背景变化不大，用一条直线已经足够。
    continuum_coefficients = np.polyfit(data_wave[good], response[good], 1)
    continuum_curve = np.polyval(continuum_coefficients, data_wave)

    # 最终 model 同时具有模板的吸收线形状和观测数据的 counts 尺度。
    model_flux = template_on_data_grid * continuum_curve
    return model_flux


template_wave, template_flux = read_synthetic_spectrum(TEMPLATE_FILE)
template_smooth = smooth_spectrum(template_flux)

for star_id in STAR_IDS:
    # 从 PypeIt 输出中读取波长、counts 和 inverse variance。
    spectrum = observed_hdu[star_id].data
    data_wave = spectrum["OPT_WAVE"]
    data_flux = spectrum["OPT_COUNTS"]

    # Q5 暂时假设速度为 0，只检查平滑与连续谱匹配是否正确。
    # Q5 暂时令 v=0。此时主要检查模型的宽度和整体高度是否接近数据；
    # 吸收线横向错开是正常的，Q6 才会寻找正确速度。
    zero_velocity_model = fit_continuum(
        template_wave, template_smooth, data_wave, data_flux
    )

    # 保存数据，Q6 会直接使用，避免再次读取 FITS。
    # 这条原始注释按要求保留；当前清晰版在 Q6 中重新按列命名这些数据。

    fig, ax = plt.subplots(figsize=(12, 4))
    fig.suptitle(
        f"lab5_figure{figure_number}_Q5_star{star_id}_"
        f"zero_velocity_template_vs_observed_spectrum"
    )
    ax.plot(data_wave, data_flux, label="Science spectrum")
    ax.plot(data_wave, zero_velocity_model, label="Zero-velocity model")
    ax.set_xlim(8300, 8700)
    ax.set_xlabel("Wavelength (Å)")
    ax.set_ylabel("Counts")
    ax.set_title(f"Q5: Star ID {star_id}")
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    plt.show()
    figure_number += 1


# %% Q6：尝试很多速度，找出 chi-square 最小的那个

'''Question 6: chi-square fitting to find velocity

课程思路：尝试一系列速度；每个速度都移动模板、重新匹配连续谱并计算 chi-square。
chi-square 最小的位置就是最佳速度，chi2_min + 1 给出一个参数的 1-sigma 误差。
'''

def calc_chi2(data_flux, model_flux, inverse_variance):
    """使用 inverse variance 计算加权 chi-square。

    返回观测光谱与模型之间的加权差异；数值越小，匹配越好。
    """
    # 课程公式：chi2 = Σ inverse_variance × (data - model)^2。
    # ivar <= 0 或非有限值代表无效像素，不能参与拟合。
    # inverse variance 越大，说明这个像素越可靠，在总分中占的权重也越大。
    good = (
        np.isfinite(data_flux)
        & np.isfinite(model_flux)
        & np.isfinite(inverse_variance)
        & (inverse_variance > 0)
    )
    residual = data_flux[good] - model_flux[good]
    return np.sum(inverse_variance[good] * residual**2)


def estimate_velocity_error(velocities, chi2_values, best_index):
    """用 chi2_min + 1 左右两个交点估计速度的 1-sigma 误差。"""
    # 课程用 chi2 < chi2_min + 1 估计 1-sigma。
    # 这里再对左右交点做线性插值，避免误差因为网格离散而显示成 0。
    target_chi2 = chi2_values[best_index] + 1

    # 从最小值向左、向右找第一个越过 target_chi2 的位置。
    left_candidates = np.where(chi2_values[:best_index] >= target_chi2)[0]
    right_candidates = np.where(chi2_values[best_index:] >= target_chi2)[0]
    left_index = left_candidates[-1]
    right_index = best_index + right_candidates[0]

    # 交点通常落在两个网格点之间，所以做一次直线插值，避免误差被显示成 0。
    left_velocity = np.interp(
        target_chi2,
        chi2_values[left_index:left_index + 2][::-1],
        velocities[left_index:left_index + 2][::-1],
    )
    right_velocity = np.interp(
        target_chi2,
        chi2_values[right_index - 1:right_index + 1],
        velocities[right_index - 1:right_index + 1],
    )
    return (right_velocity - left_velocity) / 2


def fit_velocity(
    data_wave,
    data_flux,
    inverse_variance,
    template_wave,
    template_flux,
    velocity_guess,
):
    """在初始猜测附近搜索最佳速度及其 1-sigma 误差。

    在目测速度附近搜索，返回速度网格、chi-square、最佳速度和误差。
    """
    # 以 Q3 的目测速度为中心，搜索前后 25 km/s；课程答案的步长是 0.05 km/s。
    candidate_velocities = np.arange(
        velocity_guess - 25, velocity_guess + 25, 0.05
    )
    chi2_values = []

    for velocity in candidate_velocities:
        # Doppler 公式：移动模板波长 λ_shift = λ × (1 + v/c)。
        # 这个速度假设只改变模板谱线的横向位置，不改变模板的吸收线形状。
        shifted_template_wave = template_wave * (1 + velocity / C_KMS)

        # 速度改变后要重新 rebin 和拟合 continuum，再与观测数据比较。
        # template_smooth 已在 Q5 平滑过；不必在每次循环中重复平滑。
        # 把移动后的模板放到观测波长网格，并重新匹配 continuum。
        model_flux = fit_continuum(
            shifted_template_wave, template_flux, data_wave, data_flux
        )
        chi2_values.append(calc_chi2(data_flux, model_flux, inverse_variance))

    chi2_values = np.array(chi2_values)
    # 最小 chi-square 对应最佳速度。
    best_index = np.argmin(chi2_values)
    best_velocity = candidate_velocities[best_index]
    velocity_error = estimate_velocity_error(
        candidate_velocities, chi2_values, best_index
    )

    return candidate_velocities, chi2_values, best_velocity, velocity_error


# Q3 的目测初值；每颗星只需在初值附近搜索。
velocity_guesses = {121: -400, 135: -405, 157: -50}

# 分别拟合三颗星，并打印、绘制每颗星的 chi-square 曲线。
for star_id in STAR_IDS:
    spectrum = observed_hdu[star_id].data
    data_wave = spectrum["OPT_WAVE"]
    data_flux = spectrum["OPT_COUNTS"]
    inverse_variance = spectrum["OPT_COUNTS_IVAR"]

    velocities, chi2_values, best_velocity, velocity_error = fit_velocity(
        data_wave,
        data_flux,
        inverse_variance,
        template_wave,
        template_smooth,
        velocity_guesses[star_id],
    )

    print(
        f"Star {star_id}: "
        f"v = {best_velocity:.1f} +/- {velocity_error:.3f} km/s"
    )

    fig, ax = plt.subplots(figsize=(10, 4))
    fig.suptitle(
        f"lab5_figure{figure_number}_Q6_star{star_id}_velocity_chi-square_curve"
    )
    ax.plot(velocities, chi2_values, ".")
    ax.axvline(best_velocity, color="red", label="Best velocity")
    ax.set_xlabel("Velocity (km/s)")
    ax.set_ylabel("χ²")
    ax.set_title(
        f"Q6: Star {star_id}, "
        f"v = {best_velocity:.1f} +/- {velocity_error:.3f} km/s"
    )
    ax.legend()
    fig.tight_layout()
    fig.subplots_adjust(top=0.82)
    plt.show()
    figure_number += 1



'''Question 7: χ2 fitting with more parameters'''
'''在问题 6 中，我们将平滑值固定为 25 像素，并使用单个函数来匹配合成数据与科学数据的连续性。
接下来，我们重新处理 χ2，但这次将这些值包含在拟合中。这将是一个双参数卡方拟合。
同时寻找“谱线应该移动多少”和“谱线应该变宽多少”；找到最佳组合以后，
让平滑宽度可以自由变化，再根据 χ² 最低点附近允许的速度范围计算速度误差。
 '''
# #先找到整个二维网格里的最低 χ²：
# chi2_min = np.min(chi2_values)
# 然后寻找满足下面条件的区域：
# chi2_values < chi2_min + 1

# 输入：wave，Flux，ivar，
#      template_wave，template_flux，
#      搜索范围：velocity_grid，smooth_grid
#输出：best_velocity
#      best_smoothing
#      velocity_error

## 伪代码：
# 使用数据：观测波长, 观测通量, 观测 inverse variance, 模板波长, 模板通量

# 对每颗星：
#     筛选波长范围
#     建立速度grid：根据每颗星的大致速度v0设置搜索范围与步长精度
#     建立平滑grid
#     准备保存结果 list： velocity，smoothing，chi2
#       对grid中的每个速度：
#           移动模板波长
#               对grid中的每个平滑宽度：
#                 平滑模板
#                 拟合连续谱
#                 计算 chi2
#                 保存结果到 list
#                 在所有chi2中寻找最小值，并返回best_velocity，best_smoothing
#                 计算速度误差：
#                     找到最低点附近满足 chi2 < chi2_min + 1 的区域。
#                     查看这个区域在速度方向上延伸到哪里。
#                     得到速度的左边界和右边界。
#                     用左右宽度估计速度误差。 Verror = (vright-vleft) / 2

for star_id in STAR_IDS:
    spectrum = observed_hdu[star_id].data
    data_wave = spectrum["OPT_WAVE"]
    data_flux = spectrum["OPT_COUNTS"]
    inverse_variance = spectrum["OPT_COUNTS_IVAR"]

    velocity_guess = velocity_guesses[star_id]
    velocity_grid = np.arange(velocity_guess - 25, velocity_guess + 25, 0.05)
    smooth_grid = np.arange(10, 50, 1)











    