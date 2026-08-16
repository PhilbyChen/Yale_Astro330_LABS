"""Lab 5: Stellar Spectroscopy + Fitting Models to Data"""

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
import matplotlib as mpl
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d

from astropy.io import fits
file = r"D:\Documents\GitHub\Yale_Astro330_LABS\data\lab5_data\spec1d_DE.20110603.45055-n7006a_DEIMOS_2011Jun03T123053.021.fits"

hdu = fits.open(file)

'Q3:Question 3: Plotting 1D PypeIt output spectra and fitting by eye'
starlist = [121,135,157]
for i in starlist:
    data = hdu[i].data

    fig,ax = plt.subplots(figsize=(15,3))
    plt.plot(data['OPT_WAVE'],data['OPT_COUNTS'])
    plt.title('Star ID: {}'.format(i))
    plt.xlim(8300,8850)
    plt.xlabel('Wavelength (Ang)')
    plt.ylabel('Counts')
    plt.show()

'''为了大致了解每颗恒星的速度，你可以尝试“目测”粗略估计一下。上面光谱中最强的三条谱线来自 CaII: 8500.36、8544.44 和 8664.52 埃。
你估计它们的速度是多少？'''
data = hdu[121].data
wave = data["OPT_WAVE"]
flux = data["OPT_COUNTS"]

plt.figure(figsize=(10, 4))
plt.plot(wave, flux)
ca_lines = [8500.36, 8544.44, 8664.52]
for line in ca_lines:
    plt.figure(figsize=(8, 3))
    plt.plot(wave, flux)
    plt.axvline(line, linestyle="--")
    plt.xlim(line - 25, line + 10)
    plt.xlabel("Wavelength (Å)")
    plt.ylabel("Counts")
    plt.title(f"Ca II line near {line:.2f} Å")
    plt.grid()
    plt.show()
'''Question 4: Synthetic model spectra'''
'''可以通过测量已知吸收线的中心（目测或拟合高斯函数）并将其与恒星的静止波长进行比较，来测量恒星的速度。
虽然这种方法可以估算恒星的速度，但它浪费了完整光谱中的大部分信息。为了确定更精确的速度，我们采用“模板拟合”方法，
即将已知速度的光谱与待测光谱进行比较。
模板光谱可以是经验性的（已知速度的标准恒星的观测光谱），也可以是合成的（根据恒星模型数值计算得出）。
这里我们将使用来自 PHEONIX 库的合成模板:https://phoenix.astro.physik.uni-goettingen.de/'''

template_file = r"D:\Documents\GitHub\Yale_Astro330_LABS\data\lab5_data\dmost_lte_5000_3.0_-2.0_.fits"
def read_synthetic_spectrum(pfile):
    '''
    Function to load synthetic template file into python using vacuum wavelengths
    
    Parameters
    ----------
    pfile: str
        path to the synthitic fits file to load. 
        
    Returns
    -------
    pwave: float array
        Wavelengths of synthetic spectrum
    pflux: float array
        Flux of sythetic spectrum
    '''

    with fits.open(pfile) as hdu:
        data     = hdu[1].data
        
    pflux = np.array(data['flux']).flatten()
    awave = np.exp((data['wave']).flatten())
    
    # CONVERTING AIR WAVELENGTHS TO VACUUM
    s = 10**4 / awave
    n = 1. + 0.00008336624212083 + \
            (0.02408926869968 / (130.1065924522 - s**2)) +\
            (0.0001599740894897 / (38.92568793293 - s**2))

    pwave  = awave*n
    
    return pwave, pflux
'''Question 5: Synthetic model spectra -- Smoothing and Continuum fitting'''
'''我们将把合成光谱拟合到科学数据，目的是确定科学光谱的速度。合成光谱的速度为零。为了与科学数据匹配，我们需要：
(1)将合成光谱平滑到科学数据的波长分辨率；
(2)将合成光谱平移到科学数据的速度；
(3)重新分箱合成光谱并匹配连续谱水平。'''

'Smoothing the templates '
'''我们首先讨论如何平滑合成光谱以使其与数据匹配。我们将在下文中拟合该值，但现在，我们先根据单点估计选择一个数值。
DEIMOS 光谱线可以用线宽约为 0.5 埃的 1- σ 高斯函数很好地拟合。合成光谱的分辨率为 0.02 埃。
因此，我们需要使用 0.5/0.02 = 25 像素的高斯核来平滑合成光谱。'''

def Gaussian_smooth(pwave, pflux, sigma):
    '''
    Function to smooth synthetic template spectrum using a Gaussian kernel
    
    Parameters
    ----------
    pwave: float array
        Wavelengths of synthetic spectrum
    pflux: float array
        Flux of sythetic spectrum
    sigma: float
        Width of Gaussian kernel in pixels
        
    Returns
    -------
    pflux_smooth: float array
        Smoothed flux of sythetic spectrum
    '''
    
    pflux_smoothed = gaussian_filter1d(pflux, sigma)
    
    return pflux_smoothed

'Fitting the Continuum'
'''接下来，我们将讨论上述步骤 (3)，即光谱的整体形状和数值，我们称之为“连续谱”。
让我们用一个函数拟合合成光谱，使其与科学光谱大致相同。
对于我们正在处理的光谱部分，线性函数（例如，我们在实验 4 中拟合的函数）就足够了。
为此，我们首先将合成光谱按波长重新分箱到与数据相同的数组中。'''

# 思路
# # 1) 读模板
# template_wave, template_flux = read_synthetic_spectrum(template_file)

# # 2) 平滑模板
# template_flux = Gaussian_smooth(template_wave, template_flux, sigma=25)

# # 3) 重采样到观测数据波长
# data_wave = data["OPT_WAVE"]
# # 把模板插值到观测数据的波长网格
# template_on_data_wave = np.interp(data_wave, template_wave, template_flux)
# '''np.interp 会在每个观测波长位置，取出对应模板值
# 这样两者就能一一对应地比较了。'''

# # 4) 选择一个波长区间
# mask = (data_wave > 8400) & (data_wave < 8700)
# x = data_wave[mask]
# y = template_on_data_wave[mask]

# # 5) 迭代拟合 continuum
# for _ in range(5):
#     m, b = np.polyfit(x, y, 1)
#     continuum = m * x + b
#     residual = y - continuum
#     good = np.abs(residual) < 3 * np.std(residual)
#     x = x[good]
#     y = y[good]

# # 6) 归一化
# template_norm = template_on_data_wave / (m * data_wave + b)


def rebin_template_to_data(data_wave, template_wave, template_flux):
    rebinned_flux = np.interp(data_wave, template_wave, template_flux)
    return rebinned_flux


def fit_continuum(data_wave, data_flux, template_wave, template_flux):
    rebinned_template = rebin_template_to_data(data_wave, template_wave, template_flux)

    mask = (data_wave > 8400) & (data_wave < 8700)
    x = data_wave[mask]
    y = rebinned_template[mask]

    for _ in range(10):
        m, b = np.polyfit(x, y, 1)
        continuum = m * x + b
        residual = y - continuum
        good = np.abs(residual) < 3 * np.std(residual)

        if np.sum(good) < 10:
            break

        x = x[good]
        y = y[good]

    coeff = np.polyfit(x, y, 1)
    continuum_model = np.polyval(coeff, data_wave)

    normalized_template = rebinned_template / continuum_model
    normalized_data = data_flux / continuum_model

    return continuum_model, normalized_template, normalized_data
'绘制结果'
template_wave, template_flux = read_synthetic_spectrum(template_file)
template_flux = Gaussian_smooth(template_wave, template_flux, sigma=25)

data_wave = data["OPT_WAVE"]
data_flux = data["OPT_COUNTS"]

continuum_model, norm_template, norm_data = fit_continuum(
    data_wave, data_flux, template_wave, template_flux
)

plt.figure(figsize=(12, 5))
plt.title("Continuum fit and normalized spectra")
plt.plot(data_wave, data_flux, label="science spectrum", alpha=0.8)
plt.plot(data_wave, continuum_model, label="continuum fit", color="red", lw=2)

plt.plot(data_wave, norm_data, label="normalized science", color="orange")
plt.plot(data_wave, norm_template, label="normalized template", color="green")

plt.xlim(8400, 8700)
plt.xlabel("Wavelength")
plt.ylabel("Flux")
plt.legend()
plt.show()