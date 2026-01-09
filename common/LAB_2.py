from astropy.io import fits
import os
import matplotlib.pyplot as plt
import numpy as np


## 2 Data I/O
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

def implot(image, figsize=(15, 13), cmap ='gray_r', scale = 0.5, **kwargs):
    '''
    input:
    image (2D array or masked array)
    cmap 颜色映射名称--反向灰度
    scale 对比度缩放因子
    output:
    使用 ax.imshow() 来实际绘制图像
    用 im = ax.imshow(...)保存输出
    '''
    fig, ax = plt.subplots(figsize = figsize)
    mu = np.nanmedian(image)      # 平均值
    sigma = np.nanstd(image)    # 标准差
    # 计算基于scale的默认值
    default_vmin = np.nanpercentile(image, 5)
    default_vmax = np.nanpercentile(image, 98)
    vmin = kwargs.pop('vmin', default_vmin)
    vmax = kwargs.pop('vmax', default_vmax)
    im = ax.imshow(image, cmap = cmap, origin='lower', vmin=vmin, vmax=vmax)

    return fig, ax, im

filepath = "D:\\Documents\\GitHub\\Yale_Astro330_LABS\\data\\lab2_data\\antenna_Rband.fits"

header, data = load_fits(filepath)

print("\n=== 图像数据 ===")
print(f"形状: {data.shape}")
print(f"数据类型: {data.dtype}")
print(f"像素值范围: [{np.min(data):.2f}, {np.max(data):.2f}]")
print(f"数据形状: {data.shape}")

fig, ax, im = implot(image=data, figsize=(10, 10), cmap='gray_r', scale=0.5)

plt.tight_layout()
plt.show()


'''
## 2.3

在本节中，我们将允许（可选地）在图形中添加颜色条。
我们还将添加功能，使图形能够在天球坐标（即 RA 和 DEC）而不是像素单位下绘制，如果图像信息（通过世界坐标系统，WCS）存在于图像头信息中。

给你的函数添加三个新的可选参数。
colorbar = False
header = None
wcs = None

让我们从色条开始。在你的绘图命令结束后，检查 colorbar=True ，如果存在，则通过 plt.colorbar() 创建色条，将 mappable 参数设置为之前保存 ax.imshow() 输出的内容。
同时将 ax 参数设置为你的 ax；这将告诉 matplotlib 从该轴中占用一些空间来放置色条。
'''