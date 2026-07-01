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