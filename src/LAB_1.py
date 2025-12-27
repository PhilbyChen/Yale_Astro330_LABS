#热身环节
'''Create a 2D array of dimensions 1000 x 1000, 
in which the values in each pixel are random-gaussian distributed about a mean of 10, with a sigma of 2, 
and then use matplotlib to display this image. Make sure (0,0) is in the lower lefthand corner.'''

import numpy as np
import matplotlib.pyplot as plt

array01 = np.random.normal(loc = 10, scale = 2, size = (1000,1000))

plt.figure(figsize = (10,10))
plt.imshow(array01, cmap='viridis', aspect='auto', interpolation='nearest', origin ='lower')



'''The distribution of pixels in your above image should not have many outliers beyond 3-sigma from the mean, 
but there will be some. Find the location of any 3-sigma outliers in the image, and highlight them by circling their location. 
Confirm that the fraction of these out of the total number of pixels agrees with the expectation for a normal distribution.
你上面图片中的像素分布应该不会有太多离群值，超过平均值的 3 西格玛，但还是会有一些。
找到图像中任意 3 西格玛离群值的位置，并通过圈出它们的位置来高亮它们。确认这些像素数中所占比例是否符合正态分布的预期'''

mean, std = array01.mean(), array01.std()

threshold = 3 * std
outliers = np.abs(array01 - mean > threshold)
y_coords, x_coords = np.where(outliers)

plt.plot(x_coords[:30], y_coords[:30], 
         'wo',           # 'w'=白色, 'o'=圆圈
         markersize=12,  # 更大一些
         markeredgewidth=3,      # 边缘更粗
         markeredgecolor='red',  # 红色边缘
         markerfacecolor='none', # 关键：空心
         alpha=0.8)


plt.tight_layout()
plt.show()