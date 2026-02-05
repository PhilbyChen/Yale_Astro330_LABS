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
import pandas as pd
import sys
from astropy.io import fits
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from astropy.wcs import WCS
import sep
from scipy.ndimage import maximum_filter
from astropy.modeling.functional_models import Gaussian2D




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
    
    '''
     我们的目标是估计上述图像的点扩散函数（PSF），然后测量这里恒星和星系的通量，同时考虑 PSF。
     要开始这个过程，我们需要在这张图像中定位星星。上次我们看到如何使用 sep 分割图像，但在本次实验中我们将自己执行这个步骤，
     使用两种方法。
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

    def findpeaks_maxfilter(self, threshold=None, image=None):
        
        """
        使用最大值滤波查找图像中的峰值
        
        参数:
        threshold: 检测阈值
        image: 要分析的图像（如果为None则使用self.image）
        
        返回:
        峰值位置数组，同时保存到self.peaks属性
        """
        if image is None:
            image = self.image
        # 设置默认阈值
        if threshold is None:
            threshold = np.mean(image) + 3 * np.std(image)

        
        neighborhood_size = 5

        data_max = maximum_filter(image, neighborhood_size)

        maxima = (image == data_max)

        maxima[image < threshold] = 0

        peaks_y, peaks_x = np.where(maxima)

        self.peaks = np.column_stack((peaks_y, peaks_x))
        
        return self.peaks
    

    '''我们现在有一个函数可以返回给定图像中的峰值（星星）。我们的下一步将是使用它们的质心来估计这些峰值（星星）的确切中心。'''
    def centroid_cutout(self, image, peak_x, peak_y, windowsize=11):
        '''
        计算质心并保存到类属性
        '''
        half = windowsize // 2
        
        # 处理边缘情况
        y_min = max(0, peak_y - half)
        y_max = min(image.shape[0], peak_y + half + 1)
        x_min = max(0, peak_x - half)
        x_max = min(image.shape[1], peak_x + half + 1)
        
        windowlumi = image[y_min:y_max, x_min:x_max]
        
        if windowlumi.size == 0:
            return peak_x, peak_y
        
        rows, cols = np.mgrid[y_min:y_max, x_min:x_max]
        
        total_weight = np.sum(windowlumi)
        
        if total_weight <= 0:
            return peak_x, peak_y
        
        centroid_y = np.sum(rows * windowlumi) / total_weight
        centroid_x = np.sum(cols * windowlumi) / total_weight
        
        # 初始化或更新质心列表
        if not hasattr(self, 'centroids'):
            self.centroids = []
        
        self.centroids.append({
            'peak_position': (peak_x, peak_y),
            'centroid_position': (centroid_x, centroid_y),
            'window_size': windowsize
        })
        
        return centroid_x, centroid_y
    

    def eval_gauss(self, x_arr, y_arr, sigma_x, sigma_y, mu_x, mu_y):
        g = Gaussian2D.evaluate(x=x_arr, y=y_arr, amplitude=1, theta=0,
                            x_mean=mu_x, y_mean=mu_y,
                            x_stddev=sigma_x, y_stddev=sigma_y)
        # g /= np.sum(g)
        return g

    def second_moment(self, image_cutout, xx, yy, centroid_x, centroid_y):
        '''计算二阶矩并保存结果到类属性'''
        T = np.sum(image_cutout)
        
        if T <= 0:
            return 1.0, 1.0
        
        T2x = np.sum(xx**2 * image_cutout)
        T2y = np.sum(yy**2 * image_cutout)
        
        sigma_x = np.sqrt(T2x/T - centroid_x**2)
        sigma_y = np.sqrt(T2y/T - centroid_y**2)
        
        # 处理可能的负值
        sigma_x = max(0.5, sigma_x)
        sigma_y = max(0.5, sigma_y)
        
        # 初始化或更新sigma列表
        if not hasattr(self, 'sigma_values'):
            self.sigma_values = []
        
        self.sigma_values.append({
            'sigma_x': sigma_x,
            'sigma_y': sigma_y,
            'centroid_x': centroid_x,
            'centroid_y': centroid_y
        })
        
        return sigma_x, sigma_y
    

    '''
      我们将保持质心和二阶矩函数不变，因为它们作用于单个恒星。相反，我们将编写一个新的最终方法，称为 psf_photometry 。
      当用户运行这个方法时，它应该首先将峰值逐个输入质心代码，组装一组质心。然后，它应该围绕每个峰值（或质心）构造 self.image 的切割区域，
      并将这些以及质心输入到二阶矩函数中，以保存每个恒星的一对 sigma_x 和 sigma_y。
      最后，它应该使用我上面提供的 eval_gauss 函数来执行 PSF 测光。
    '''
    def psf_photometry(self, threshold=None, background_subtract=True):
        # 用图像
        img = self.image if background_subtract else self.data_calibrated
        # 找峰值
        peaks = self.findpeaks_maxfilter(threshold=threshold or np.mean(img)+3*np.std(img))
        # 算质心
        centroids = []
        for y, x in self.peaks:
            try:
                cx, cy = self.centroid_cutout(img, x, y)
                centroids.append([cx, cy])
            except:
                pass
        
        if not centroids:
            return pd.DataFrame()
        
        centroids = np.array(centroids)
        
        # 去重
        filtered = []
        for i, p1 in enumerate(centroids):
            if i == 0 or np.all(np.sqrt(np.sum((centroids[:i] - p1)**2, axis=1)) >= 5):
                filtered.append(p1)
        
        # 测光
        results = []
        for cx, cy in filtered:
            half = 10
            y_min, y_max = max(0, int(cy)-half), min(img.shape[0], int(cy)+half+1)
            x_min, x_max = max(0, int(cx)-half), min(img.shape[1], int(cx)+half+1)
            
            if (y_max-y_min) < 5 or (x_max-x_min) < 5:
                continue
            
            cut = img[y_min:y_max, x_min:x_max]
            yy, xx = np.indices(cut.shape)
            
            # 计算sigma
            try:
                sigma_x, sigma_y = self.second_moment(cut, xx, yy, cx-x_min, cy-y_min)
                if sigma_x <= 0.1 or sigma_y <= 0.1:
                    continue
            except:
                continue
            
            # PSF测光
            try:
                yy_abs, xx_abs = np.mgrid[y_min:y_max, x_min:x_max]
                psf = self.eval_gauss(xx_abs, yy_abs, sigma_x, sigma_y, cx, cy)
                flux_psf = np.sum(cut * psf)
                results.append({
                    'centroid_x': cx, 'centroid_y': cy,
                    'sigma_x': sigma_x, 'sigma_y': sigma_y,
                    'cutout_flux': cut.sum(), 'psf_flux': flux_psf
                })
            except:
                pass
        
        return pd.DataFrame(results)




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


'''查找峰值并计算质心'''
threshold = np.mean(pipe.image) + 3 * np.std(pipe.image)
peaks = pipe.findpeaks_maxfilter(threshold=threshold)
all_cen = []
for i in range(len(peaks)):
    try:
        y = peaks[i][0]
        x = peaks[i][1]
        cx, cy = pipe.centroid_cutout(pipe.image, x, y, windowsize=11) 
        all_cen.append([cx,cy])
    except AssertionError:
        continue
all_cen = np.array(all_cen)

'''去除重复质心'''
out_cen = []
for i in range(len(all_cen)):
    # sub2 = (所有质心 - 当前质心)²
    sub2 = (all_cen - all_cen[i])**2
    sq = np.sum(sub2,axis=1)
    # sqt = √((Δx)² + (Δy)²) = 欧几里得距离
    sqt = np.sqrt(sq)
    # 找出距离小于20像素的质心
    ind, = np.where(sqt<20)
    # 如果只有自己（len(ind) == 1）或没有其他点太近, 保留这个质心
    if len(ind) < 2:
        out_cen.append(all_cen[i])
out_cen = np.array(out_cen)

fig, ax, im = implot(pipe.image,scale=0.5)
ax.plot(out_cen[:,0],out_cen[:,1],'o',color='None',mec='r',ms=10)
plt.tight_layout()
plt.show()


pipe.subtract_background(mask)
results = pipe.psf_photometry(background_subtract=True)
print(results)

fig, ax = plt.subplots(figsize=(7,7))
ax.plot([6e4,1e7],[6e4,1e7],lw=3.5,color='C3',alpha=0.7,label='1:1 relation')
ax.plot(results['cutout_flux'], results['psf_flux'], 'o', alpha=0.5, 
        mec='k', ms=10, label='data')

ax.set_xscale('log')
ax.set_yscale('log')
ax.tick_params(which='both', direction='in', top=True, right=True, labelsize=16)
ax.tick_params(which='major', length=10, width=2)
ax.tick_params(which='minor', length=5, width=2)

for axis in ['top','bottom','left','right']:
    ax.spines[axis].set_linewidth(2)

ax.set_xlabel('Cutout Flux (square aperture)', fontsize=18)
ax.set_ylabel('PSF Flux (gaussian model)', fontsize=18)

plt.show()