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

'''When dealing with astronomical data, it is sometimes advisable to not include outliers in a calculation being performed on a set of data (in this example, an image). 
We know, of course, that the data we're plotting ARE coming from a gaussian distribution, 
so there's no reason to exclude, e.g., 3-sigma outliers, but for this example, let's assume we want to.
处理天文数据时，有时建议不在对一组数据（本例中为图像）的计算中包含异常值。当然，我们知道我们绘制的数据确实来自高斯分布，所以没有理由排除，比如 3 西格玛离群值，但在这个例子中，假设我们想要排除

Create a numpy masked array in which all pixels that are > 3σ
 from the image mean are masked. Then, calculate the mean and sigma of the new masked array.
 创建一个 numpy 蒙罩数组，其中所有 σ 距离图像平均值> 3 的像素都被蒙蔽。然后计算新掩码阵列的均值和σ
 
Clipping the outliers of this distribution should not affect the mean in any strong way, but should noticably decrease σ
截断该分布的离群值不应对均值产生强烈影响，但应显著减少 σ'''

mask = outliers
masked_arr = np.ma.masked_array(array01, mask=True)

new_mean = masked_arr.mean()
new_std = masked_arr.std()

print(f"蒙蔽像素比例: {mask.sum()/array01.size:.4%}")


'''Using Array indexing, re-plot the same array from above, but zoom in on the inner 20% of the image, such that the full width is 20% of the total. 
Note: try not to hard code your indexing. You should be able to flexibly change the percentage. For this one, use a white-to-black color map.
使用数组索引，重新绘制上方相同的数组，但放大图像内侧 20%，使得全宽度占总图的 20%。
注意：尽量不要硬编码你的索引。你应该可以灵活调整百分比。对于这个，使用白到黑的色彩映射。

Your image should now be 200 by 200 pixels across. Note that your new image has its own indexing. 
A common “gotcha” when working with arrays like this is to index in, but then try to use indices found (e.g., via where()) in the larger array on the cropped in version, 
which can lead to errors.
你的图片现在应该是 200×200 像素。
请注意，你的新图片有自己的索引。
在处理类似数组时，一个常见的“陷阱”是先索引进去，但试图在裁剪后的版本中使用大数组中找到的索引（例如通过 where() ），这可能导致错误'''