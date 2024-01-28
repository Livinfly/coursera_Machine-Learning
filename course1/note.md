## supervised-learning

输入 x ，给出 y 。

-   回归，预测。
-   分类。
-   Gradient Decrease

## unsupervised-learning

只有输入，寻找结构。

-   聚类，成组。
-   异常查找。
-   数据降维。



---



## 特征值缩放



归一化使得不同 feature 规模一致，使得 $w_i$ 的 GD 变化过程比较一致。

feature scaling，x / MAX

mean normalization，均值归一化，(x - avg)  / (MAX-MIN) 等。。

x-min / max-min

Z-score normalization 标准化，(x - avg) / sigma_i（标准差）标准化后的数据**保持异常值中的有用信息**，使得算法对异常值不太敏感。



靠近 [-1, 1] 大部分也就行。



（1）数据的分布本身就服从**正态分布**，使用Z-Score。

（2）有**离群值**的情况：使用Z-Score。

这里不是说有离群值时使用Z-Score不受影响，而是，Min-Max对于离群值十分敏感，因为离群值的出现，会影响数据中max或min值，从而使Min-Max的效果很差。相比之下，虽然使用Z-Score计算方差和均值的时候仍然会受到离群值的影响，但是相比于Min-Max法，影响会小一点。

（3）如果对输出结果**范围有要求**，用归一化。

（4）如果数据较为稳定，不存在极端的最大最小值，用归一化。

（5）如果数据存在异常值和较多噪音，用标准化，可以间接通过中心化避免异常值和极端值的影响。





## 梯度下降是否收敛

如果不是连续下降，可能出 bug 了，或者 lr 过大。



## 对特征进行组合 - feature engineering

如，把长和宽组合成面积，保留原本的长和宽，添加面积这一特征，可能会有更好的效果。

同时要注意 feature scaling



np.c_[]  转类型 矩阵转置



---



## Logistic regression（逻辑斯蒂回归） 

用于二分类问题

损失函数（Loss Function ）是定义在单个样本上的，算的是一个样本的误差。
代价函数（Cost Function ）是定义在整个训练集上的，是所有样本误差的平均，也就是损失函数的平均。

maximum likelihood 最大似然。

Logistic regression 的损失函数的选择是统计学里的最大似然的衍生产物。

loss:  - y log(f(x)) - (1-y) log(1 - f(x))

GD 的形式和平方误差函数类似，拥有一些相似的性质。



## 过拟合 和 欠拟合

high bias, high variance



## 处理过拟合

更更多的数据，适当减少考虑的特征，Regularization 正则化，

把高次的权重变低，类似于下面这样。

$cost = J(x_i) + 1000{w_3}^2 + 1000{w_4} ^ 2$



$cost = J(x_i) + \frac{\lambda}{2m} \sum{w_j} ^ 2$

$\lambda$ 过小仍然过拟合，过大欠拟合，也是需要调参的量。

正则化的工作原理，在对 GD 的式子整理后，发现先把 w 缩小一些，然后再减原 GD 的值。



对于 linear regression 和 logistic regression 都是加后面这一部分。 



两个公式：

一个是 loss / cost，cost记得求和完后取平均。

一个是 gradient decrease