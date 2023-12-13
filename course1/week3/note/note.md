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