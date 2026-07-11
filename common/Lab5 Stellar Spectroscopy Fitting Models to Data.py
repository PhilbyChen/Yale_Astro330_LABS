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

'为了大致了解每颗恒星的速度，你可以尝试“目测”粗略估计一下。上面光谱中最强的三条谱线来自 CaII: 8500.36、8544.44 和 8664.52 埃。你估计它们的速度是多少？'
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
虽然这种方法可以估算恒星的速度，但它浪费了完整光谱中的大部分信息。为了确定更精确的速度，我们采用“模板拟合”方法，即将已知速度的光谱与待测光谱进行比较。
模板光谱可以是经验性的（已知速度的标准恒星的观测光谱），也可以是合成的（根据恒星模型数值计算得出）。
这里我们将使用来自 PHEONIX 库的合成模板:https://phoenix.astro.physik.uni-goettingen.de/'''

template_file = 'dmost_lte_5000_3.0_-2.0_.fits'
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