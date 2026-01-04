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

 Once you’ve stored the χ2 values for each combination of a and b, create a plot with a and b as the axes and show using colored circles the χ2 value at each location. 
 Add a colorbar to see the values being plotted.
 存储好 每个组合 a 和的 χ2 值后 b ，创建一个以 a 和 b 为轴的图，并用彩色圆圈 显示每个位置的 χ2 值。添加一个颜色条来查看正在绘制的数值。

 To create this grid, use the np.meshgrid() function. For your plot, make sure the marker size is big enough to see the colors well.
 要创建这个网格，请使用函数 np.meshgrid() 对于你的地块，确保记号笔尺寸足够大，能清楚看到颜色。
'''

def chi2(a,b):
    return ((15-a)**2+(12-b)**2)**0.2 #note, this is nonsense, but should return a different value for each input a,b
# 在二维参数空间 a,b 中以参数拟合 χ2，并用颜色表示

# np.linspace(start, stop, num)
a_values = np.linspace(1, 25, 10)
b_values = np.linspace(1, 25, 10)

A_grid, B_grid = np.meshgrid(a_values, b_values)

chi2_matrix = np.zeros((10, 10))
# 使用双重循环计算每个 a,b 坐标的 χ²值
for i in range(10):      # i: 行索引，对应 b_values[i]
    for j in range(10):  # j: 列索引，对应 a_values[j]
        chi2_matrix[i, j] = chi2(a_values[j], b_values[i])

plt.figure(figsize=(12, 10))
scatter = plt.scatter(A_grid, B_grid, c=chi2_matrix, s=800, cmap='viridis')
plt.colorbar(label='χ²')

plt.tight_layout()
plt.show()




'''Question 7
Re-show your final plot above, making the following changes:

label your colorbar as χ2 using latex notation, with a fontsize>13

Make your ticks point inward and be longer

Make your ticks appear on the top and right hand axes of the plot as well

If you didn’t already, label the x and y axes appropriately and with a font size > 13

Make sure the numbers along the axes have fontsizes > 13'''

ax = plt.gca() # 获取当前坐标轴命令
ax.tick_params(
    direction='in',   # 刻度方向：向内
    length=8,         # 刻度长度：8点
    width=1.5,        # 刻度宽度：1.5点  
    labelsize=14,     # 刻度数字大小：14号
    which='both',     # 应用到：主刻度和次刻度
    top=True,         # 顶部显示刻度
    right=True        # 右侧显示刻度
)
ax.set_xlabel('Parameter a', fontsize=15)
ax.set_ylabel('Parameter b', fontsize=15)

cbar = plt.colorbar(scatter)
cbar.set_label(r'$\chi^2$',   # LaTeX格式的χ²
               fontsize=16,   # 字体大小16
               rotation=0,    # 不旋转（0度）
               labelpad=15)   # 与颜色条的距离

plt.tight_layout()
plt.show()





'''Question 8
Some quick list comprehensions! For any unfamilar, comprehensions are pythonic statements that allow you to compress a for-loop (generally) into a single line, 
and usually runs faster than a full loop (but not by a ton).
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

Next, create an array of integers including 1 through 30, inclusive. Using a comprehension, 
create a numpy array containing the squared value of only the odd numbers in your original array. (Hint, remember the modulo operator)
接着，创建一个包含 1 到 30 的整数数组。利用理解法，创建一个 numpy 数组，只包含原始数组中奇数的平方值。（提示，记住模算子）

In the next example, you have a list of first names and a list of last names. Use a list comprehension to create an array that is a list of full names 
(with a space between first and last names).
在下一个例子中，你有一个名字列表和一个姓氏列表。使用列表理解创建一个数组，包含全名（名字和姓氏之间留空格）。
'''
visited_cities = ['San Diego', 'Boston', 'New York City','Atlanta']
all_cities = ['San Diego', 'Denver', 'Boston', 'Portland', 'New York City', 'San Francisco', 'Atlanta']

# not_visited = []
# for city in all_cities:
#     if city not in visited_cities:
#         not_visited.append(city)

not_visited = [city for city in all_cities if city not in visited_cities]
print(not_visited)

Rarr = np.arange(1,31)
odd_num = [x**2 for x in Rarr if x % 2 == 1]
print(odd_num)



'''Question 8
Take the arrays XX, YY, and ZZ below and create one multidimensional array in which they are the columns. Print to confirm this worked.
取数组 XX 、 YY 和 ZZ 下，创建一个多维数组，其中它们是列。打印确认是否有效。
'''
XX = np.array([1,2,3,4,5,6,7,8,9])
YY = np.array([5,6,7,8,9,10,11,12,13])
ZZ = np.array([10,11,12,13,14,15,16,17,18])

multidim_arr = np.column_stack((XX, YY, ZZ))



##################################################################################################################################################
'''Question 10
Units, units, units. The bane of every scientists’ existence… except theorists that set every constant equal to 1.
单位，单位，单位。每个科学家的噩梦......除了那些将每个常数都设为 1 的理论家。

In the real world, we measure fluxes or magnitudes in astronomical images, infer temperatures and densities from data and simulations, 
and ultimately have to deal with units one way or another.
在现实世界中，我们测量天文图像中的通量或星等，通过数据和模拟推断温度和密度，最终不得不以某种方式处理单位。

Thankfully, our friends at astropy know this, and they’ve come to save the day. This next question serves as an introduction to the units module in astropy,
which can be both a live saver and a pain in the ass, but at the end of the day is absolutely worth learning.
幸运的是，我们的朋友 astropy 们知道这一点，他们来拯救了大家。接下来这个问题是对 AstroPy 模块 units 的入门介绍，这个模块既是救命宝，也可能麻烦，但归根结底绝对值得学习。

The standard import for this library is u, so be careful not to name any variables that letter.
该库的标准导入是 u ，所以请注意不要把变量命名为该字母。

To “assign” units to a variable, we multiply by the desired unit as follows. 
Note that generally the module knows several aliases/common abrreviations for a unit, if it is uniquely identifiable.
要“赋值”给变量，我们按如下方式乘以所需的单位。注意，通常该模块知道多个别名/常见缩写，如果该单元是唯一可识别的。
 star_temp = 5000*u.K 
 star_radius = 0.89 * u.Rsun 
 star_mass = 0.6 * u.Msun

We can perform trivial conversions using the .to() method.
我们可以用该 .to() 方法进行平凡的转换。

 star_radius.to(u.km)

Once we attach units to something, it is now a Quantity object. Quantity objects are great, above, we saw they have built-in methods to facilitate conversion. 
They can also be annoying – sometimes another function we’ve written needs just the raw value or array back out. To get this, we use the .value attribute of a quantity object:
一旦我们把单位附加到某物上，它就是一个 Quantity 对象。数量对象很棒，上面我们看到它们内置了促进转换的方法。它们也可能让人烦——有时我们写的另一个函数只需要重新输出原始值或数组。
为此，我们使用一个数量对象的 .value 属性：

 star_mass.to(u.kg).value
 1.1930459224188305e+30
This now strips away all Quantity stuff and gives us an array or value to use elsewhere in our code.
这样就剥离了所有 Quantity 内容，给出了一个数组或值，可以在代码的其他地方使用。


Units are great because they help us combine quantities while tracking units and dimensional analysis. 
A common operation in astronomy is converting a flux to a luminosity given a distance, using
单位很棒，因为它们帮助我们在跟踪单位和量纲分析时合并数量。天文学中常见的作是将通量转换为给定距离的光度，使用

F=L4πD2
where L
 is the luminosity and D
 is the distance to the source.
其中 L
 是光度， D
 是距离光源的距离。
What if I’ve made a flux measurement in astronomical units such as erg/s/cm2
, and I want to know the luminosity in solar luminosities, and my distance happens to be in Mpc? Regardless of my input units, I can easily do this:
如果我用天文单位（如 erg/s/cm 2）测量通量，想知道太阳光度中的光度，而我的距离恰好是以 Mpc 为单位呢？无论输入单位如何，我都能轻松做到：

 L = 4 * np.pi * (3.6*u.Mpc)**2 * (7.5e-14 * u.erg/u.s/u.cm**2)
 L.to(u.Lsun)

 
The virial temperature of a galaxy halo is given roughly by
星系晕的维里温度大致为

Tvir≃5.6×104K(μ0.59)(Mhalo1010M⊙)2/3(1+z4)
where here, we can assume μ
 is 0.59.
这里，我们可以假设 μ
 为 0.59。

Write a function that takes as an input a halo mass, redshift, and optionally μ
 (default 0.59), and returns the virial temperature in Kelvin. Your function should take in an astropy quantity with mass units,
 but should allow for the mass to be input with any appropriate units.
写一个函数，输入为光环质量、红移，以及可选 μ的（默认 0.59），并返回开尔文单位的维里温度。你的函数应当接受一个带有质量单位的天体量，但也应允许用任何合适的单位输入质量。
'''

import astropy.units as u

def virial_T(halomass,redshift,mu=0.59):

    if isinstance(halomass, u.Quantity):
        # 带有单位的Quantity输入
        M_halo = halomass.to(u.Msun).value
    else:
        # 纯数字输入（假定为太阳质量）
        M_halo = halomass

    Tvir = 5.6e4 * (mu/0.59) * (M_halo/1e10) ** (2/3) * ((1 + redshift) / 4)
    return Tvir * u.K