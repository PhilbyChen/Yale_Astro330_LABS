# Lab 5 所需的 Python 复习指南

这不是一本完整的 Python 教科书。它只讲理解 Astro 330 Lab 5 的 Q5、Q6 所需要的内容。

建议分三遍阅读：

1. 第一遍只看“大图景”、变量词典和伪代码，不要求看懂每个符号。
2. 第二遍运行小例子，观察每个变量里装了什么。
3. 第三遍再回到 `Lab5 Stellar Spectroscopy Fitting Models to Data.py` 对照正式代码。

如果一次读不完，第一遍读到“函数”和“循环”就可以停下来。

---

## 0. 先记住整个实验在做什么

观测光谱可以想成一张有很多点的图：

- 横坐标是波长 `wave`；
- 纵坐标是观测到的强度 `flux`；
- 吸收线表现为光谱向下凹的地方。

我们还有一条理论计算得到的模板光谱：

- `template_wave` 是模板波长；
- `template_flux` 是模板强度。

Q5 的工作是：

```
把理论模板处理成和观测光谱可以直接比较的模型。
```

Q6 的工作是：

```
不断移动模板，找到它与观测光谱最匹配的速度。
```

完整数据流如下：

```
template_flux
    │
    ├── Gaussian 平滑
    ▼
template_smooth
    │
    ├── 按某个速度移动 template_wave
    ├── 插值到观测 wave
    ├── 匹配观测 continuum
    ▼
model
    │
    ├── 与观测 flux 比较
    ├── 使用 ivar 加权
    ▼
chi2
    │
    └── chi2 最小的位置 = 最佳速度
```

你暂时不必知道每一步的实现细节。先记住“输入 → 处理 → 输出”的方向。

---

## 1. 最重要的变量词典

读代码时，先判断一个变量在物理上是什么，不要先纠结它的 Python 写法。

| 变量 | 含义 | 单位或特点 | 可以怎样想象 |
|---|---|---|---|
| `wave` | 观测光谱的波长 | Å | 观测图的横坐标 |
| `flux` | 观测到的 counts | counts | 观测图的纵坐标 |
| `ivar` | inverse variance | 值越大，像素越可靠 | 每个点的可信度 |
| `template_wave` | 理论模板的波长 | Å | 模板图的横坐标 |
| `template_flux` | 理论模板的强度 | 通常在 1 附近 | 模板图的纵坐标 |
| `template_smooth` | 平滑后的模板强度 | 与 `template_flux` 同长度 | 更接近仪器分辨率的模板 |
| `shifted_wave` | 按速度移动后的模板波长 | Å | 吸收线移动后的横坐标 |
| `model` | 已插值并匹配 continuum 的模板 | counts | 可以直接和 `flux` 比较 |
| `velocities` | 要尝试的所有速度 | km/s | 速度候选列表 |
| `chi2_values` | 每个速度对应的 chi-square | 越小越好 | 每个候选速度的成绩 |
| `best_v` | 最佳速度 | km/s | `chi2_values` 最小处的速度 |
| `v_error` | 速度的形式误差 | km/s | chi-square 最小值附近的宽度 |

遇到看不懂的变量，先问三个问题：

1. 它是一个数，还是一整列数？
2. 它在物理上表示波长、强度、速度，还是误差？
3. 它是原始数据，还是经过处理的中间结果？

### 3.2 列表 `list`

```python
starlist = [121, 135, 157]
```

这是一个列表，里面放了三个星的编号。

```python
starlist[0]   # 121
starlist[1]   # 135
starlist[2]   # 157
```

Python 从 0 开始编号。

### 3.3 字典 `dict`

```python
velocity_guesses = {
    121: -400,
    135: -405,
    157: -50,
}
```

字典保存“键 → 值”的对应关系：

```python
velocity_guesses[121]   # -400
velocity_guesses[157]   # -50
```

这里的意思是：星 `121` 的速度初值是 `-400 km/s`。

### 3.4 元组 `tuple` 和拆包

代码中保存一颗星的数据：

```python
star_data[star_id] = (wave, flux, ivar)
```

右边的 `(wave, flux, ivar)` 是一个元组，可以理解为把三个有关联的对象打包。

之后可以拆开：

```python
wave, flux, ivar = star_data[star_id]
```

这等价于：

```python
package = star_data[star_id]
wave = package[0]
flux = package[1]
ivar = package[2]
```

### 3.5 NumPy 数组

光谱不是一个数，而是很多数排列成的一列：

```python
wave = np.array([8499.8, 8500.0, 8500.2, 8500.4])
flux = np.array([950.0, 910.0, 870.0, 930.0])
```

NumPy 数组最大的特点是可以一次对整列数字计算：

```python
flux / 2
```

结果是：

```python
array([475., 455., 435., 465.])
```

不需要自己写循环逐个除以 2。

---

## 4. 函数：一台有输入和输出的小机器

### Lab 5 中的平滑函数

```python
def smooth_spectrum(flux, sigma=25):
    smoothed = gaussian_filter1d(flux, sigma)
    return smoothed
```

结构是：

```text
输入原始模板 flux
    ↓
Gaussian 平滑
    ↓
返回平滑模板 smoothed
```

调用：

```python
template_smooth = smooth_spectrum(template_flux)
```

这里没有传入 `sigma`，所以 Python 使用默认值 `25`。

也可以自己指定：

```python
template_smooth = smooth_spectrum(template_flux, sigma=30)
```

### 如何阅读一个陌生函数

不要一开始逐行钻进去。先只找四样东西：

1. 函数名；
2. 输入参数；
3. `return` 返回什么；
4. 用一句中文概括它的工作。

例如：

```python
def fit_continuum(template_wave, template_flux, data_wave, data_flux):
    ...
    return model
```

第一遍只需要理解：

```text
输入模板和观测数据，输出一个可以与观测数据比较的 model。
```

函数内部细节可以第二遍再看。

---

## 5. `for` 循环：把相同工作重复几次
---

## 6. 数组索引和切片

假设：

```python
numbers = np.array([10, 20, 30, 40, 50])
```

### 取一个元素

```python
numbers[0]   # 10
numbers[2]   # 30
numbers[-1]  # 50，最后一个
```

### 取一段元素

```python
numbers[1:4]
```

结果是：

```python
array([20, 30, 40])
```

规则是“包括左边，不包括右边”。

### 倒序

```python
numbers[::-1]
```

结果是：

```python
array([50, 40, 30, 20, 10])
```

Q6 的误差插值中出现了 `[::-1]`。它只是把数组反过来，让 `np.interp()` 的横坐标按照从小到大的顺序排列。

这部分是整段代码中较难的细节。第一遍可以把它当成“求 chi-square 左右交点的工具”，不用马上掌握。

---

## 7. 布尔数组和筛选

假设：

```python
values = np.array([1, 5, 10, 20])
```

比较：

```python
values > 6
```

结果不是一个 `True`，而是一整列：

```python
array([False, False, True, True])
```

把它作为索引：

```python
values[values > 6]
```

结果是：

```python
array([10, 20])
```

### 同时满足两个条件

```python
good = (values > 3) & (values < 15)
```

`&` 表示“并且”。结果是：

```python
array([False, True, True, False])
```

因此：

```python
values[good]
```

得到：

```python
array([5, 10])
```

### `good &= condition` 是什么

```python
good &= response < high
```

是下面写法的缩写：

```python
good = good & (response < high)
```

意思是：在原有筛选条件上，再增加一个新条件。

---

## 8. 怎样检查变量里到底装了什么

代码一多时，不要只盯着变量名猜。直接打印。

```python
print(type(wave))
print(wave.shape)
print(wave[:5])
```

分别告诉你：

1. `wave` 是什么类型；
2. 它有多少个元素；
3. 前五个数是什么。

对于 Lab 5，可以临时加入：

```python
print("star_id:", star_id)
print("wave shape:", wave.shape)
print("wave first 3:", wave[:3])
print("flux first 3:", flux[:3])
print("ivar first 3:", ivar[:3])
```

字符串只是标签，帮助你认出输出属于哪个变量。

还可以检查范围：

```python
print("wave range:", np.min(wave), np.max(wave))
print("flux range:", np.min(flux), np.max(flux))
```

学习阶段多打印不是坏习惯。确认理解后再删除即可。

---

## 9. 从 FITS 文件读取光谱

代码：

```python
spectrum = hdu[star_id].data
wave = spectrum["OPT_WAVE"]
flux = spectrum["OPT_COUNTS"]
ivar = spectrum["OPT_COUNTS_IVAR"]
```

可以这样理解：

```text
hdu
 └── 第 star_id 个数据单元
      ├── OPT_WAVE        → wave
      ├── OPT_COUNTS      → flux
      └── OPT_COUNTS_IVAR → ivar
```

`spectrum["OPT_WAVE"]` 类似从表格中选择名为 `OPT_WAVE` 的一列。

---

## 10. Q5：把理论模板变成可以比较的模型

### 10.1 为什么不能直接比较

理论模板可能是：

```text
0.98, 1.00, 0.91, 0.99, ...
```

观测 counts 可能是：

```text
850, 900, 780, 870, ...
```

二者整体尺度不同，不能直接相减。

Q5 要完成三件事：

```text
平滑模板 → 插值到观测波长 → 匹配观测 continuum
```

### 10.2 平滑模板

```python
template_smooth = smooth_spectrum(template_flux)
```

模板的分辨率很高，而 DEIMOS 观测的吸收线更宽。平滑使理论吸收线宽度更接近真实仪器数据。

### 10.3 `np.interp()`：让两个数组使用同一套横坐标

模板和观测数据的波长位置通常不同：

```text
模板波长：8499.98, 8500.00, 8500.02, ...
观测波长：8499.91, 8500.24, 8500.57, ...
```

不能直接比较不同位置上的值。

```python
template = np.interp(
    data_wave,
    template_wave,
    template_flux,
)
```

可以读成：

```text
请计算模板在每一个 data_wave 位置上应该是多少。
```

输出的 `template` 与 `data_wave`、`data_flux` 长度相同。

### 10.4 `response = data / template`

```python
response = data_flux / template
```

这是逐元素除法：

```text
response[0] = data_flux[0] / template[0]
response[1] = data_flux[1] / template[1]
...
```

模板描述吸收线，二者相除后剩下的整体趋势主要是观测 continuum。

### 10.5 为什么排除最低 20% 和最高 1%

```python
low, high = np.percentile(response[good], [20, 99])
good &= (response > low) & (response < high)
```

- 很低的点可能是深吸收线；
- 很高的点可能是 cosmic ray 或坏像素；
- continuum 应该由中间较稳定的数据点决定。

### 10.6 一次多项式拟合

```python
coefficients = np.polyfit(
    data_wave[good],
    response[good],
    1,
)
```

最后的 `1` 表示拟合一次多项式：

```text
continuum = a × wavelength + b
```

`coefficients` 中保存的是 `a` 和 `b`。

在所有波长位置上计算直线：

```python
continuum = np.polyval(coefficients, data_wave)
```

最后：

```python
model = template * continuum
```

`model` 已经具有与观测 `flux` 相同的 counts 尺度。

### 10.7 Q5 的人话伪代码

```text
读取模板
平滑模板

对于每颗星：
    读取 wave、flux、ivar
    把模板插值到 wave
    计算 flux / template
    排除吸收线和离群点
    用直线拟合 continuum
    model = template × continuum
    画出 flux 和 model
```

---

## 11. Q6：通过 chi-square 寻找速度

### 11.1 核心想法

如果模板速度不对，吸收线位置对不上，`model` 和 `flux` 的差会比较大。

如果速度正确，吸收线位置对齐，二者的差会比较小。

```text
速度不对 → chi2 大
速度正确 → chi2 小
```

### 11.2 速度如何移动波长

低速近似下：

```python
shifted_wave = template_wave * (1 + velocity / C_KMS)
```

对应：

```text
移动后的波长 = 原波长 × (1 + 速度 / 光速)
```

负速度会使模板向较短波长移动，正速度会向较长波长移动。

### 11.3 为什么需要 `ivar`

课程使用：

```python
chi2 = np.sum(ivar * (flux - model)**2)
```

对应公式：

```text
chi2 = Σ inverse_variance × (观测值 - 模型值)²
```

- `ivar` 大：这个像素更可靠，影响更大；
- `ivar` 小：这个像素噪声更大，影响更小；
- `ivar <= 0`：通常代表无效像素，应该排除。

### 11.4 建立速度候选数组

```python
velocities = np.arange(
    velocity_guess - 25,
    velocity_guess + 25,
    0.05,
)
```

如果 `velocity_guess = -400`，它会产生近似：

```text
-425.00, -424.95, -424.90, ..., -375.05
```

也就是在初值前后 25 km/s 搜索，每次移动 0.05 km/s。

### 11.5 对每个速度重复相同计算

```python
chi2_values = []

for velocity in velocities:
    shifted_wave = template_wave * (1 + velocity / C_KMS)
    model = fit_continuum(shifted_wave, template_smooth, wave, flux)
    one_chi2 = calc_chi2(flux, model, ivar)
    chi2_values.append(one_chi2)
```

第一次循环可能是：

```text
velocity = -425.00 → chi2 = 15342
```

第二次：

```text
velocity = -424.95 → chi2 = 15291
```

一直计算到整个速度数组结束。

`append()` 表示把一个新结果加到列表最后。

### 11.6 找到最小值

```python
best_index = np.argmin(chi2_values)
```

`np.argmin()` 返回的不是最小的 chi-square 本身，而是它的位置。

例如：

```python
values = np.array([8, 5, 2, 6])
np.argmin(values)  # 2
```

因为最小值 `2` 位于索引 `2`。

所以最佳速度是：

```python
best_v = velocities[best_index]
```

### 11.7 `chi2_min + 1` 误差

对于只拟合一个参数的情况，课程使用：

```text
chi2 < chi2_min + 1
```

来估计约 (1\sigma) 的速度范围。

正式代码还对左右交点进行了插值。第一次阅读时，可以把
`estimate_velocity_error()` 整体理解为：

```text
寻找 chi2_min + 1 左右两边对应的速度，二者宽度的一半就是 v_error。
```

暂时不必逐个理解 `np.where()`、切片和 `[::-1]`。这段是辅助数值细节，不是 Q6 的核心物理逻辑。

### 11.8 Q6 的人话伪代码

```text
对每颗星：
    根据 Q3 得到一个速度初值
    建立初值附近的速度列表

    对每一个候选速度：
        移动模板波长
        重新插值并匹配 continuum
        计算 model 和 flux 的 chi2
        保存 chi2

    找到最小 chi2 的位置
    该位置对应最佳速度
    用 chi2_min + 1 计算误差
    画速度—chi2 曲线
```

---

## 12. 把正式函数当作“黑盒”使用

在完全理解函数内部之前，可以先记住它们的合同：输入什么，输出什么。

### `smooth_spectrum`

```text
输入：template_flux
输出：template_smooth
作用：降低模板分辨率，使它接近 DEIMOS
```

### `fit_continuum`

```text
输入：template_wave、template_flux、data_wave、data_flux
输出：model
作用：重采样模板并匹配观测 continuum
```

### `calc_chi2`

```text
输入：flux、model、ivar
输出：一个 chi2 数字
作用：评价模型与观测数据有多不匹配
```

### `fit_velocity`

```text
输入：wave、flux、ivar、velocity_guess
输出：velocities、chi2_values、best_v、v_error
作用：搜索最佳速度并估计误差
```

当你能说清这四个“合同”时，即使暂时不懂所有内部代码，也已经理解了程序结构。

---

## 13. 一次只追踪一颗星

三颗星一起运行会增加认知负担。学习时可以临时改成：

```python
starlist = [121]
```

这样只处理一颗星，输出和图都更少。

理解以后再改回：

```python
starlist = [121, 135, 157]
```

也可以先测试较少的速度：

```python
velocities = np.arange(-402, -401, 0.1)
```

这样更容易打印每个速度的结果，确认循环在做什么。

学习完成后再恢复课程使用的正式范围。

---

## 14. 推荐的单步观察方法

在 Q5 中临时加入：

```python
print("template_flux:", template_flux[:5])
print("template_smooth:", template_smooth[:5])
print("model_zero_v:", model_zero_v[:5])
```

在 Q6 循环中只打印前几个速度：

```python
for index, velocity in enumerate(velocities):
    shifted_wave = template_wave * (1 + velocity / C_KMS)
    model = fit_continuum(shifted_wave, template_smooth, wave, flux)
    one_chi2 = calc_chi2(flux, model, ivar)

    if index < 5:
        print(index, velocity, one_chi2)
```

`enumerate()` 同时给出：

- `index`：这是第几轮；
- `velocity`：这一轮的速度。

---

## 15. 常用 NumPy 函数速查

| 函数 | 人话解释 |
|---|---|
| `np.array(...)` | 建立 NumPy 数组 |
| `np.arange(start, stop, step)` | 按固定步长生成一列数字 |
| `np.interp(new_x, old_x, old_y)` | 估计 `old_y` 在 `new_x` 位置上的值 |
| `np.isfinite(x)` | 检查是否不是 `NaN` 或无穷大 |
| `np.percentile(x, 20)` | 找到数组的第 20 百分位数 |
| `np.polyfit(x, y, 1)` | 用直线拟合 x 和 y |
| `np.polyval(coefficients, x)` | 在 x 上计算拟合好的多项式 |
| `np.sum(x)` | 把数组中所有元素相加 |
| `np.argmin(x)` | 返回最小值所在的位置 |
| `np.where(condition)` | 返回满足条件的元素位置 |
| `np.min(x)` / `np.max(x)` | 数组的最小值 / 最大值 |

---

## 16. 小练习

不要一次做完。每次只做一两题。

### 练习 1：列表和循环

预测输出：

```python
stars = [121, 135, 157]

for star in stars:
    print(star + 1)
```

### 练习 2：字典

```python
guesses = {121: -400, 135: -405, 157: -50}
```

怎样取出星 `135` 的速度初值？

### 练习 3：数组运算

预测结果：

```python
flux = np.array([100, 200, 300])
template = np.array([1.0, 0.5, 1.0])
response = flux / template
```

### 练习 4：布尔筛选

```python
values = np.array([1, 4, 7, 10, 15])
good = (values > 3) & (values < 12)
print(values[good])
```

预测会打印哪些数？

### 练习 5：函数

写一个函数：输入速度 `v`，返回 `v / 299792.4`。

### 练习 6：`np.argmin`

```python
velocities = np.array([-2, -1, 0, 1, 2])
chi2_values = np.array([10, 4, 1, 3, 9])
```

最佳速度是多少？

### 练习 7：用中文描述函数

不用解释每行代码，只用一句中文描述：

```python
def calc_chi2(flux, model, ivar):
    good = np.isfinite(flux) & np.isfinite(model) & (ivar > 0)
    return np.sum(ivar[good] * (flux[good] - model[good])**2)
```

### 练习 8：追踪变量

对下面每个变量标注“波长 / 强度 / 速度 / 拟合优度”：

```text
shifted_wave
template_smooth
best_v
chi2_values
```

---

## 17. 练习答案

### 答案 1

```text
122
136
158
```

### 答案 2

```python
guesses[135]
```

结果是 `-405`。

### 答案 3

```python
array([100., 400., 300.])
```

这是逐元素除法。

### 答案 4

```python
array([4, 7, 10])
```

### 答案 5

```python
def speed_fraction(v):
    return v / 299792.4
```

### 答案 6

最小 chi-square 是 `1`，位置为索引 `2`：

```python
best_index = np.argmin(chi2_values)
best_v = velocities[best_index]
```

所以最佳速度是 `0`。

### 答案 7

```text
排除无效像素后，使用 inverse variance 计算观测光谱与模型之间的加权 chi-square。
```

### 答案 8

```text
shifted_wave    → 波长
template_smooth → 强度
best_v          → 速度
chi2_values     → 拟合优度
```

---

## 18. 最后：怎样面对一大段代码

不要从第一行一直读到最后一行。按下面顺序阅读：

### 第一层：只看章节

```text
Q5：建立模型
Q6：搜索速度
```

### 第二层：只看函数名

```text
smooth_spectrum
fit_continuum
calc_chi2
fit_velocity
```

### 第三层：只看输入和 return

弄清每个函数“吃进去什么、吐出来什么”。

### 第四层：再看函数内部的四五个步骤

一次只研究一个函数，不要同时追踪整个文件。

### 第五层：最后再看复杂切片和误差插值

这些是实现细节，不是理解整体逻辑的前提。

当代码让你混乱时，回到这四句话：

```text
wave 是横坐标。
flux 是观测纵坐标。
model 是处理后的理论纵坐标。
chi2 衡量 flux 和 model 有多不一样。
```

只要这四个对象没有混淆，Q5 和 Q6 的主线就不会丢。

---

## 19. 与正式代码和课程答案对照

正式代码：

```text
common/Lab5 Stellar Spectroscopy Fitting Models to Data.py
```

重点位置：

```text
Q5：第 84 行附近
Q6：第 148 行附近
```

课程参考答案：

<https://astro-330.github.io/Lab5/Lab5_solutions.html>

当前正式代码遵循课程的核心步骤：Gaussian smoothing、rebin、continuum fitting、Doppler shift、inverse-variance weighted chi-square 和 `chi2_min + 1` 误差估计。

建议下一步只学习 Q5。等你能够用自己的话解释 `smooth_spectrum()` 和 `fit_continuum()` 后，再进入 Q6。
