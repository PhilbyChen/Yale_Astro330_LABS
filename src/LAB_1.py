#热身环节
'''Question 1

Create a 2D array of dimensions 1000 x 1000, 
in which the values in each pixel are random-gaussian distributed about a mean of 10, with a sigma of 2, 
and then use matplotlib to display this image. Make sure (0,0) is in the lower lefthand corner.'''

import numpy as np
import matplotlib.pyplot as plt

array01 = np.random.normal(loc = 10, scale = 2, size = (1000,1000))

plt.figure(figsize = (10,10))
plt.imshow(array01, cmap='viridis', aspect='auto', interpolation='nearest', origin ='lower')
plt.tight_layout()
plt.show()



'''Question 2

The distribution of pixels in your above image should not have many outliers beyond 3-sigma from the mean, 
but there will be some. Find the location of any 3-sigma outliers in the image, and highlight them by circling their location. 
Confirm that the fraction of these out of the total number of pixels agrees with the expectation for a normal distribution.
你上面图片中的像素分布应该不会有太多离群值，超过平均值的 3 西格玛，但还是会有一些。
找到图像中任意 3 西格玛离群值的位置，并通过圈出它们的位置来高亮它们。确认这些像素数中所占比例是否符合正态分布的预期'''

mean, std = array01.mean(), array01.std()

threshold = 3 * std
outliers = np.abs(array01 - mean) > threshold
y_coords, x_coords = np.where(outliers)

plt.figure(figsize = (10,10))
plt.imshow(array01, cmap='viridis', aspect='auto', interpolation='nearest', origin='lower')
plt.plot(x_coords[:30], y_coords[:30], 
         'wo',           # 'w'=白色, 'o'=圆圈
         markersize=12,  # 更大一些
         markeredgewidth=3,      # 边缘更粗
         markeredgecolor='red',  # 红色边缘
         markerfacecolor='none', # 关键：空心
         alpha=0.8)
plt.title(f'标记3σ离群值\nμ={mean:.2f}, σ={std:.2f}')
plt.tight_layout()
plt.show()



'''Question 3

When dealing with astronomical data, it is sometimes advisable to not include outliers in a calculation being performed on a set of data (in this example, an image). 
We know, of course, that the data we're plotting ARE coming from a gaussian distribution, 
so there's no reason to exclude, e.g., 3-sigma outliers, but for this example, let's assume we want to.
处理天文数据时，有时建议不在对一组数据（本例中为图像）的计算中包含异常值。当然，我们知道我们绘制的数据确实来自高斯分布，所以没有理由排除，比如 3 西格玛离群值，但在这个例子中，假设我们想要排除

Create a numpy masked array in which all pixels that are > 3σ
 from the image mean are masked. Then, calculate the mean and sigma of the new masked array.
 创建一个 numpy 蒙罩数组，其中所有 σ 距离图像平均值> 3 的像素都被蒙蔽。然后计算新掩码阵列的均值和σ
 
Clipping the outliers of this distribution should not affect the mean in any strong way, but should noticably decrease σ
截断该分布的离群值不应对均值产生强烈影响，但应显著减少 σ'''

mask = outliers
masked_arr = np.ma.masked_array(array01, mask=outliers)

new_mean = masked_arr.mean()
new_std = masked_arr.std()

print(f"蒙蔽像素比例: {mask.sum()/array01.size:.4%}")



'''Question 4

Using Array indexing, re-plot the same array from above, but zoom in on the inner 20% of the image, such that the full width is 20% of the total. 
Note: try not to hard code your indexing. You should be able to flexibly change the percentage. For this one, use a white-to-black color map.
使用数组索引，重新绘制上方相同的数组，但放大图像内侧 20%，使得全宽度占总图的 20%。
注意：尽量不要硬编码你的索引。你应该可以灵活调整百分比。对于这个，使用白到黑的色彩映射。

Your image should now be 200 by 200 pixels across. Note that your new image has its own indexing. 
A common “gotcha” when working with arrays like this is to index in, but then try to use indices found (e.g., via where()) in the larger array on the cropped in version, 
which can lead to errors.
你的图片现在应该是 200×200 像素。
请注意，你的新图片有自己的索引。
在处理类似数组时，一个常见的“陷阱”是先索引进去，但试图在裁剪后的版本中使用大数组中找到的索引（例如通过 where() ），这可能导致错误'''

zoom_percent = 20

def calculate_zoom_region(array01, zoom_percent):
    # 计算放大区域的索引
    
    # 返回: row_start, row_end, col_start, col_end: 索引范围
    
    rows, cols = array01.shape
    # 中心位置
    center_row = rows // 2
    center_col = cols // 2

    zoom_width = int(rows * zoom_percent / 100)
    # 缩放边界位置
    row_start = center_row - zoom_width // 2
    row_end = row_start + zoom_width
    col_start = center_col - zoom_width // 2
    col_end = col_start + zoom_width

    zoomed_array = array01[row_start: row_end, col_start: col_end]
    
    return zoomed_array

zoomed_result = calculate_zoom_region(array01, zoom_percent)

plt.figure(figsize=(10, 10))
# 'gray_r' 是白到黑色彩映射（r表示reverse）
plt.imshow(zoomed_result, cmap='gray_r', origin='lower', aspect='auto')
plt.colorbar()

plt.tight_layout()
plt.show()



'''Question 5

Often, we have an expression to calculate of the form
∑i∑jaibj
Your natural impulse for coding this double sum might look like this:

which, mathematically, makes sense! But as it turns out, there’s a way we can do this without any loops at all — and when a⃗ 
and b⃗ get long, this becomes hugely important in our code.
从数学上讲，这很合理！但事实证明，我们有一种方法可以在完全没有循环的情况下做到这一点——当 a⃗ 循环 b⃗ 变得很长时，这在我们的代码中变得非常重要。

The trick we’re going to use here is called array broadcasting, which you can read about at the link if you’re not already familar. I’m going to give you a⃗ 
and b⃗ 
below. For this exercise, calculate the double sum indicated above without the use of a for-loop. Check that your code worked by using the slow double-loop method.
我们这里要用的技巧叫做阵列广播，如果你还不熟悉，可以在链接中阅读相关内容。我将给你 a⃗ 
和 b⃗ 下面。在此练习中，计算上述的双重和，不使用 for 循环。用慢双循环法检查你的代码是否能正常工作。

Hint  提示
The command np.newaxis will be useful here, or for a slightly longer solution, try np.repeat and reshape().
该命令 np.newaxis 在这里会很有用，或者如果有更长的解法，试着 np.repeat 和 reshape() 。
'''

a = np.array([1,5,10,20])
b = np.array([1,2,4,16])

# np.repeat ：
a_repeated = np.repeat(a[:, np.newaxis], len(b), axis=1)
b_repeated = np.repeat(b[np.newaxis, :], len(a), axis=0)
bc_result_repeat = a_repeated * b_repeated

# np.reshaped
a_reshaped = a.reshape(-1, 1)
b_reshaped = b.reshape(1, -1)
bc_result_reshaped = a_reshaped * b_reshaped

# np.axis
bc_result_axis = a[:, np.newaxis] * b[np.newaxis, :]

doublesum1 = np.sum(bc_result_repeat)
doublesum2 = np.sum(bc_result_reshaped)
doublesum3 = np.sum(bc_result_axis)

print(f"\n总和 ∑∑ aᵢbⱼ = {doublesum1, doublesum2, doublesum3}")



'''Question 6
Often in astronomy we need to work with grids of values. For example, let’s say we have a model that describes some data, and the model has 2 parameters, a and b.
在天文学中，我们经常需要使用数值网格。例如，假设我们有一个描述某个数据的模型，该模型有两个参数，a b
We might choose different combinations of a and b, and determine a metric for how well models of such combinations fit our data (e.g., χ2).
我们可以选择不同的组合和 a b ，并确定这些组合模型与数据拟合程度的度量

We may then want to plot this χ2 value for each point on our grid – that is, at each grid position corresponding to some ai and bj.
然后我们可能想为网格上的每个点绘制这个 χ2 值——即在对应某个 ai 和 bj的每个网格位置.

Below, I provide a function, chi2, which returns a single number given some singular inputs a and b.
下面，我提供了一个函数， chi2 给定一些奇异输入 a 和 b ，返回一个单一数字

Create some arrays of a and b to test that range between 1 and 25, and have 10 entries evenly spaced between those values. Then, loop over them and find the χ2
 using my function.
创建一个数组 b 和 a ，测试 1 到 25 之间的范围，并在这些值之间均匀分布 10 个条目。然后，循环它们，找到 χ2 使用我的函数。

 Once you’ve stored the χ2 values for each combination of a and b, create a plot with a and b as the axes and show using colored circles the χ2 value at each location. Add a colorbar to see the values being plotted.
 存储好 每个组合 a 和的 χ2 值后 b ，创建一个以 a 和 b 为轴的图，并用彩色圆圈 显示每个位置的 χ2 值。添加一个颜色条来查看正在绘制的数值。

 To create this grid, use the np.meshgrid() function. For your plot, make sure the marker size is big enough to see the colors well.
 要创建这个网格，请使用函数 np.meshgrid() 对于你的地块，确保记号笔尺寸足够大，能清楚看到颜色。
'''





'''Question 7
Re-show your final plot above, making the following changes:

label your colorbar as χ2
 using latex notation, with a fontsize>13

Make your ticks point inward and be longer

Make your ticks appear on the top and right hand axes of the plot as well

If you didn’t already, label the x and y axes appropriately and with a font size > 13

Make sure the numbers along the axes have fontsizes > 13'''



'''Question 8
Some quick list comprehensions! For any unfamilar, comprehensions are pythonic statements that allow you to compress a for-loop (generally) into a single line, and usually runs faster than a full loop (but not by a ton).
快速理解一下清单！对于任何不熟悉的人来说，理解是非常有技巧的语句，可以让你把一个 for 循环（通常）压缩成一行，通常比完整循环快（但不会快很多）。
Take the for-loop below and write it as a list comprehension.
    
拿下面的 for-loop 写成列表理解:
visited_cities = ['San Diego', 'Boston', 'New York City','Atlanta']
all_cities = ['San Diego', 'Denver', 'Boston', 'Portland', 'New York City', 'San Francisco', 'Atlanta']

not_visited = []
for city in all_cities:
    if city not in visited_cities:
        not_visited.append(city)
        
print(not_visited)

Next, create an array of integers including 1 through 30, inclusive. Using a comprehension, create a numpy array containing the squared value of only the odd numbers in your original array. (Hint, remember the modulo operator)
接着，创建一个包含 1 到 30 的整数数组。利用理解法，创建一个 numpy 数组，只包含原始数组中奇数的平方值。（提示，记住模算子）

In the next example, you have a list of first names and a list of last names. Use a list comprehension to create an array that is a list of full names (with a space between first and last names).
在下一个例子中，你有一个名字列表和一个姓氏列表。使用列表理解创建一个数组，包含全名（名字和姓氏之间留空格）。

'''



'''Question 8
Take the arrays XX, YY, and ZZ below and create one multidimensional array in which they are the columns. Print to confirm this worked.
取数组 XX 、 YY 和 ZZ 下，创建一个多维数组，其中它们是列。打印确认是否有效。
'''



'''Question 10
 This next question serves as an introduction to the units module in astropy

The standard import for this library is u, so be careful not to name any variables that letter.
该库的标准导入是 u ，所以请注意不要把变量命名为该字母。
'''