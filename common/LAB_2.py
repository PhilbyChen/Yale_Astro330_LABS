from astropy.io import fits
import os
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

## 问题 2： Data I/O
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


filepath = "D:\\Documents\\GitHub\\Yale_Astro330_LABS\\data\\lab2_data\\antenna_Rband.fits"

header, data = load_fits(filepath)

print("\n=== 图像数据 ===")
print(f"形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print(f"像素值范围: [{np.min(data):.2f}, {np.max(data):.2f}]")
print(f"数据形状: {data.shape}")

fig, ax, im = implot(image=data, figsize=(10, 10), cmap='gray_r', scale=0.5, header = header)

plt.tight_layout()
plt.show()


## 问题 3：图像裁剪和孔径光度测量

# coord = SkyCoord('12:01:53.6 -18:53:11',unit=(u.hourangle,u.deg))
# cutout = Cutout2D(data, coord, size=(1*u.arcmin, 1*u.arcmin), wcs = WCS(header))

coord = SkyCoord('12:01:55.0 -18:52:45',unit=(u.hourangle,u.deg))
cutout = Cutout2D(data, coord, size=(1*u.arcmin, 1*u.arcmin), wcs = WCS(header))
fig, ax, im = implot(cutout.data, scale=2, wcs=cutout.wcs)
# plt.tight_layout()
# plt.show()

def run_sep(data):
    '''
    Sep wrapper... runs a basic set of sep commands on a given image
    
    Parameters
    ----------
    data: array_like
        the input image
    thresh_scale: float
        a scaling parameter used by sep to determine when to call something a source (see sep documentation)
        
    Returns
    -------
    objects: numpy struct array
        numpy structured array containing the sep-extracted object locations, etc. 
    '''
    # Background subtraction
    data_sepuse = data.copy(order = 'C')
    bkg = sep.Background(data_sepuse, bw=64, bh=64, fw=3, fh=3)
    data_sub = data - bkg
    # Object detection
    objects = sep.extract(data_sub, 1.5, err=bkg.globalrms)
    return objects
objects = run_sep(cutout.data)
len(objects)

fig, ax, im= implot(cutout.data,scale=2,vmin=10,wcs=cutout.wcs)
ax.plot(objects['x'],objects['y'],'o',ms=15,color='None',mec='r')
plt.tight_layout()
plt.show()

'''虽然 sep 可以自己执行光圈光度测量（阅读文档，您可以看到输入对象和像素半径非常简单），
但我们对事情会更加小心。为了更好地可视化和处理这些数据，我希望它是一个 pandas DataFrame 。我们将在本课程中使用这些对象。
将 numpy 线性数组转换为数据帧'''
df = pd.DataFrame(objects)
print(df)

plt.figure()
plt.plot(df.flux,'.')
# plt.show()


def remove_outliers(df, flux_min, flux_max):
    '''
    编写一个名为 remove_outliers 的函数，
    读取一个数据框和一个 flux-min 和 flux-max。
    它应该过滤数据框，只包含输入值之间的通量，并返回新的数据框。
    然后使用这个函数对你的数据进行处理，选择一个合适的截止值。
    '''
    fixed_df = df[(df['flux'] >= flux_min) & (df['flux'] <= flux_max)]
    return fixed_df
df2 = remove_outliers(df, 0, 10000)

plt.figure()
plt.plot(df2.flux,'.')
plt.show()

fig, ax, im = implot(cutout.data,scale=2,vmin=10,wcs=cutout.wcs)
ax.plot(df2['x'],df2['y'],'o',ms=15,color='None',mec='r')

# Continuum Subtraction 连续谱减除
header_Halpha, data_Halpha = load_fits("D:\\Documents\\GitHub\\Yale_Astro330_LABS\data\\lab2_data\\antenna_Haband.fits")
header_copy2 = header.copy()
header_copy2 = strip_SIP(header_copy2)
cutout_Halpha = Cutout2D(data_Halpha, coord, size=(1*u.arcmin, 1*u.arcmin), wcs = WCS(header_Halpha))

fig, ax, im = implot(cutout_Halpha.data,scale=2,vmin=10,wcs=cutout_Halpha.wcs)
ax.plot(df2['x'],df2['y'],'o',ms=15,color='None',mec='r')
ax.contour(cutout_Halpha.data,levels=np.logspace(1.8,4,20),colors='r',alpha=0.5)
ax.set_title("Continuum Subtraction", fontsize=14)
plt.show()

'''我们现在必须尝试对数据进行连续体减法。我们不能简单地从 Hα
 波段中减去 R
 波段，因为 R
 波段滤镜的通带更宽，因此在相似的曝光时间内，收集的光子比更窄的 Hα
 滤镜要多得多。理想情况下，可以使用一组前景恒星（在两张图像中都是纯连续体源）来测量两者的通量，并找到一个缩放常数。
 在这里，我们看到图像中的所有源很可能实际上是 HII 区。这意味着如果我们用它们来缩放图像之间，我们可能会过度减去真实通量。
 相反，我要在这里做的是（这可能有点粗糙），选择一个没有 Hα等值线的连续体发射的“空白”区域，并断言这个区域在两张图像中必须具有相同的通量。
 在上面的图中，我在天空中标记了一个蓝色矩形区域, 这就是我们将用来测量连续体到窄带比率的东西。

 为了制作这个区域，我们将使用 SkyRectangularAperture ，来自 photutils 。如果你没有它，可以在你的 a330 环境中使用 pip install photutils 。
 '''
patch_cent = SkyCoord('12:01:53.7 -18:52:52',unit=(u.hourangle,u.deg))  # 孔径中心的天文坐标。这可以是标量坐标或坐标数组。
aper = SkyRectangularAperture(patch_cent, 0.27*u.arcsec, 0.2*u.arcsec)
R_flux = aperture_photometry(cutout.data, aper, error=None, mask=None, method='exact', subpixels=5, wcs=cutout.wcs)['aperture_sum'][0]
Ha_flux = aperture_photometry(cutout_Halpha.data, aper, error=None, mask=None, method='exact', subpixels=5, wcs=cutout_Halpha.wcs)['aperture_sum'][0]
fluxratio = R_flux / Ha_flux
print(fluxratio)

'''
Now, fill in the function below, which should read in your rectangular patch, the R
 band cutout object, and the Hα
 cutout object. Within the function, copy in the code that determines the ratio, and then use the ratio you found to scale your R
 band image, then subtract it from the Hα
 image. The function should return this new image array.
 Plot up your continuum subtracted image using your implot() function, adding back in the apertures we found earlier and the contours we made from the full Hα image.
'''