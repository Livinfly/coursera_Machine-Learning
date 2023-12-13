## Clustring

### k-means

设置质心，移动质心。

设定有几个重心（几类）；染色，更新质心。



随机选 k 个例子。

重复跑 50-1000 次，选 distortion 最小的那个。



#### 选择 K 的值

Elbow method，在 k 变大，J 降低比较快的地方，但不一定好用。

基于后面想要用来干什么。（你想要分成几类）



图片压缩，颜色的距离，选出 K 个颜色代替。



## Anomaly detection

高斯分布，概率，设置阈值。

对特征值的优化，对特征可以 （+C 取 log） 、幂次，使其更符合高斯分布。

不同的特征值进行组合，因为一个值高可能是合理的，但一个值高另一个值却低可能是不合理的。



训练，正常的，cv 正常和不正常的，ts 不一定要。

模型可用 F_1 score 评估。

true 答案为 true

positive 预测为 positive

$prec = \frac{tp}{tp+fp}, rec = \frac{tp}{tp+fn}, F_1 = \frac{2\cdot prec \cdot rec}{prec + rec}$



### 对比有监督学习

数据集中错误的数量比较少，错误的种类可能并没有完全统计完，还是选择anomaly detection。

