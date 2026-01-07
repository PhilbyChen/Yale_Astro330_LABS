from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np

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
    pass

filepath = "D:\\Documents\\GitHub\\Yale_Astro330_LABS\\data\\lab2_data\\antenna_Rband.fits"

def implot(image, figsize=(15, 13), cmap ='gray_r', scale = 0.5, **kwargs):
    '''
    input:
    image (2D array or masked array)
    cmap 颜色映射名称--反向灰度
    scale 对比度缩放因子
    output:
    fig, ax
    '''
    header, image, data = load_fits(filepath)
    print(f"header: {header}")
    print(f"数据形状: {data.shape}")

    fig, ax = plt.subplots(figsize=figsize)

    mu = np.nanmean(image)      # 平均值
    sigma = np.nanstd(image)    # 标准差
    # 计算基于scale的默认值
    default_vmin = mu - scale * sigma
    default_vmax = mu + scale * sigma
    # 检查kwargs中是否有vmin/vmax
    if 'vmin' not in kwargs:
        kwargs['vmin'] = default_vmin
    if 'vmax' not in kwargs:
        kwargs['vmax'] = default_vmax

    im = ax.imshow(image, cmap = cmap, **kwargs)
    
    return fig, ax

plt.tight_layout()
plt.show()