""" Lab 3: 构建光度测量流程 """

"""
在本实验中，我们将使用类和函数来构建一个自动提取图像中恒星通量的流程。我们都很熟悉天体测量法，但在此情况下，我们将额外进行一步操作，即对图像进行点扩散函数（PSF）卷积。

步骤：
   Reading in an image 
   读取图像

   Finding the local peaks in the image (the stars)
   在图像中找到局部峰值（即星星）

   Calculating the centroid of each peak region
   计算每个峰值区域的质心

   Convolving with the PSF and extracting flux. 
   与点扩散函数卷积并提取通量。
"""
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import sep
from photutils.aperture import SkyRectangularAperture
from photutils.aperture import aperture_photometry
import os
from scipy.ndimage import maximum_filter

sys.path.append(r"D:\\Documents\\GitHub\\Yale_Astro330_LABS")

def load_fits(filepath, extension = 0, *args):
    '''
    Load a specific extension from a FITS file.
    
    Parameters:
    filepath : str
        Path to the FITS file.
        
    extension : int, optional
        Extension number to load (default 0).
        
    Returns:
    header : astropy.io.fits.header.Header
        Header object from the specified extension.
        
    data : numpy.ndarray or None
        Data array from the specified extension.
        Returns None if extension has no data.
    '''
    with fits.open(filepath) as hdul:
        hdu = hdul[extension]
        return hdu.header.copy(), hdu.data

def strip_SIP(header):
    A_prefixes = [i for i in header.keys() if i.startswith('A_')]
    B_prefixes = [i for i in header.keys() if i.startswith('B_')]
    for a,b in zip(A_prefixes,B_prefixes):
        del header[a]
        del header[b]
    return header

def implot(image, figsize=(9, 9), cmap ='gray_r', scale = 0.5, 
           colorbar = False, header = None, wcs = None, 
           **kwargs):
    '''
    Plot an astronomical image, setting default options and easy tweaking of parameters
    
    Parameters
    ----------
    image: array_like
        2D array containing an astronomical image to be plotted. Cutouts should be input as cutout.data.
    figsize: tuple, optional
        figure size to use. Default: (15,13)
    cmap: str, optional
        Colormap to use for the image. Default: 'gray_r'
    scale: float, optional
        By default, function will scale image to some number of standard deviations about the mean pixel value. Scale sets this number (or fraction). Default: 0.5.
    colorbar: bool, optional
        Whether to add a colorbar or not. Default: False
    header: dict, optional
        If input, function will attempt to create a WCS object from header and plot in celestial coordinates. Default: None
    wcs: WCS object
        If input, the function will plot using a projection set by the WCS. Default: None
    **kwargs
        Additional arguments are passed to matplotlib plotting commands. Currently supported: vmin, vmax.
        
    Returns
    -------
    fig, ax
        figure and axes objects containing currently plotted data.
    '''
    # 如果提供了header但没提供wcs，从header创建wcs
    # if header is not None 和 if header 不一样！
    final_wcs = None
    
    if wcs is not None:
        final_wcs = wcs
    elif header is not None:
        try:
            # 先复制header，再修复，避免修改原始数据
            header_copy = header.copy()
            header_copy = strip_SIP(header_copy)
            final_wcs = WCS(header_copy)
        except:
            final_wcs = None
    # 创建图形和坐标轴，如果提供了wcs则使用投影.
    if final_wcs is not None:  # 检查wcs
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': final_wcs})
    else:
        fig, ax = plt.subplots(figsize=figsize)

    # 计算基于scale的默认值
    default_vmin = np.nanpercentile(image, 5)
    default_vmax = np.nanpercentile(image, 98)
    vmin = kwargs.pop('vmin', default_vmin)
    vmax = kwargs.pop('vmax', default_vmax)
    im = ax.imshow(image, cmap = cmap, origin='lower', vmin=vmin, vmax=vmax)

    if final_wcs is not None:
        ax.coords[0].set_axislabel('Right Ascension [hms]', fontsize=15)
        ax.coords[1].set_axislabel('Declination [degrees]', fontsize=15)
        ax.coords[0].set_ticklabel(size=15)
        ax.coords[1].set_ticklabel(size=15)
        ax.coords.grid(color='gray', alpha=0.5, linestyle='solid')

    if colorbar:
        plt.colorbar(im, ax = ax)

    return fig, ax, im

header, data = load_fits("D:\\Documents\\GitHub\\Yale_Astro330_LABS\\data\\lab3_data\\2020-04-15-0001.fits")
fig, ax, im = implot(data, header=header, colorbar=True)
ax.set_title("M81/M82 Field - Dragonfly Telephoto Array")
plt.show()



class PSFPhot():
    def __init__(self, data_fpath, dark_path, flat_path):
        self.header, self.data_init = load_fits(data_fpath)
        # 校准图像
        self.data_calibrated = self.flat_field(
            self.dark_subtract(self.data_init,dark_path),flat_path
            )
        self.image = None  
        self.background = None

    ''' 暗场扣除 去掉探测器自身的热噪声 '''
    def dark_subtract(self, image, dark_path):
        _, dark_im = load_fits(dark_path)
        return image - dark_im
    
    ''' 平坦场校正 '''
    def flat_field(self, image, flat_path):
        _, flat_im = load_fits(flat_path)
        return image / (flat_im / np.max(flat_im))

    def subtract_background(self,mask=None):
        data_copy = self.data_calibrated.copy(order = 'C')
        bkg = sep.Background(data_copy, mask=mask)
        self.background = bkg.back()
        self.image = self.data_calibrated - self.background
        print("Background estimated; output saved to attribute 'image' ")
        return self
    
    '''
     我们的目标是估计上述图像的点扩散函数（PSF），然后测量这里恒星和星系的通量，同时考虑 PSF。
     要开始这个过程，我们需要在这张图像中定位星星。上次我们看到如何使用 sep 分割图像，但在本次实验中我们将自己执行这个步骤，使用两种方法。
    '''
    def set_image_mask(self, mask):
        if hasattr(self,'image'):
            self.image = np.ma.masked_array(self.image,mask=mask)
        else:
            self.image = np.ma.masked_array(self.data_calibrated,mask=mask)
        return self
    
    # def find_peak(self, image, threshold):
    #     '''
    #     通过遍历每个像素并检查其邻域来查找图像中的峰值，其中“峰值”被定义为比所有相邻像素（即 8 个周围像素）具有更高通量的区域。
    #     为了不拾取随机噪声像素，还需要输入一个名为 threshold 的参数。
    #     在你的算法中，不要返回任何像素值低于此阈值的“峰值”像素

    #     Algorithm for finding peaks (above a threshold) in an image
    
    #     Parameters
    #     ----------
    #     image: array_like
    #         2D array containing the image of interest.
    #     threshold: float
    #         minimum pixel value for inclusion in search
    
    #     Returns
    #     -------
    #     peak_x_values, peak_y_values: array_like, array_like
    #         arrays containing the x and y coordinates of peak regions.
    #     '''
    #     peaks = []
    #     data = self.image.data if hasattr(self.image, 'mask') else self.image

    #     for y in range(1, data.shape[0] - 1):       # 跳过边界值，没有完整的八个邻居
    #         for x in range(1, data.shape[0] - 1):       # shape[0] = 3 （行数/高度）
    #                                                     # shape[1] = 4 （列数/宽度）
    #             center = data[y, x]             # 取图像中第y行、第x列的像素值

    #             if center <= threshold:
    #                 continue

    #             # 邻居结构
    #             # [y-1, x-1] [y-1, x] [y-1, x+1]
    #             # [y,   x-1] [y,   x] [y,   x+1]   中心[y,x]
    #             # [y+1, x-1] [y+1, x] [y+1, x+1]
    #             if (center > data[y-1, x-1] and 
    #                 center > data[y-1, x] and 
    #                 center > data[y-1, x+1] and
    #                 center > data[y, x-1] and 
    #                 center > data[y, x+1] and
    #                 center > data[y+1, x-1] and 
    #                 center > data[y+1, x] and 
    #                 center > data[y+1, x+1]):
    #                 peaks.append((y, x))
    #     return peaks

    def findpeaks_maxfilter(self, threshold=0, windowsize = 8):
        '''
         有几种解决方案，通常涉及过滤图像或使用模板与图像进行交叉相关。这里有一个这样的解决方案。
        '''
        neighborhood = np.ones((windowsize, windowsize), dtype=bool)                    # just 3x3 True, defining the neighborhood over which to filter
        # find local maximum for each pixel
        amax = maximum_filter(self.image, footprint=neighborhood)       #max filter will set each 9-square region in the image to the max in that region.
    
        peaks = np.where((self.image == amax) & (self.image >= threshold))    #find the pixels unaffected by the max filter.
        peaks = np.array([peaks[0],peaks[1]]).T
     
        outpeaks = []
        data = self.image.data if hasattr(self.image, 'mask') else self.image
        for y in range(1, data.shape[0]-1):
            for x in range(1, data.shape[1]-1):
                center = data[y, x]
                if center > threshold and center == data[y-1:y+2, x-1:x+2].max():
                    if np.sum(data[y-1:y+2, x-1:x+2] > center*0.8) >= 4:
                        outpeaks.append([y, x])
        return np.array(outpeaks)
    

    '''我们现在有一个函数可以返回给定图像中的峰值（星星）。我们的下一步将是使用它们的质心来估计这些峰值（星星）的确切中心。'''
    def centroid_cutout(self, image, peak_x, peak_y, windowsize=11):
        '''
        1. 接收一个星星位置 (x,y)
        2. 以(x,y)为中心，切N×N的小方块
        3. 计算小方块里每个像素的"权重"（亮度）
        4. 用加权平均公式：
        cx = Σ(每个像素的x坐标 × 像素亮度) ÷ Σ(所有像素亮度)
        cy = Σ(每个像素的y坐标 × 像素亮度) ÷ Σ(所有像素亮度)
        5. 返回(cx, cy)
        '''
        half = windowsize // 2
        y_min = peak_y - half
        y_max = peak_y + half + 1
        x_min = peak_x - half
        x_max = peak_x + half + 1
        windowlumi = self.image[y_min:y_max, x_min:x_max]
        rows, cols = np.mgrid[y_min:y_max, x_min:x_max]

        total_weight = np.sum(windowlumi)
        centroid_y = np.sum(rows * windowlumi) / total_weight
        centroid_x = np.sum(cols * windowlumi) / total_weight
        return centroid_x, centroid_y
        










base_path = r"D:\\Documents\\GitHub\\Yale_Astro330_LABS\\data\\lab3_data"

pipe = PSFPhot(
    f"{base_path}/2020-04-15-0001.fits",
    f"{base_path}/2020-04-15-dark.fits",
    f"{base_path}/2020-04-15-flat.fits"
)

mask = np.zeros_like(pipe.data_calibrated, dtype=bool)
mask[900:1250,0:300] = True
mask[850:1200,900:1100] = True

pipe.subtract_background(mask)
pipe.set_image_mask(mask)
peaks = pipe.findpeaks_maxfilter(threshold=np.mean(pipe.image)+3*np.std(pipe.image))
# Note that this is returned in row, column form (so y,x).
fig, ax, im = implot(pipe.image,scale=0.5)
ax.plot(peaks[:,1],peaks[:,0],'o',color='None',mec='r',ms=10,alpha=0.8);


implot(pipe.image)
implot(pipe.background,colorbar=True)
plt.show()