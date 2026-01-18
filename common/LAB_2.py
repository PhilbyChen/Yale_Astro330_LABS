from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np
from astropy.wcs import WCS
from astropy.nddata import Cutout2D
from astropy.coordinates import SkyCoord
import astropy.units as u
import sep

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

if __name__ == "__main__":
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
plt.tight_layout()
plt.show()

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