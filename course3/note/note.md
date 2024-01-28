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



---



推荐系统，不是学得很明白。



## PCA (Principal Component Analysis)

简化（组合、删减）特征数到两三个，方便可视化。



---



Discount Factor，$Return = R_1 + \gamma R_2 + \gamma^2 R_3 ...$ .

reward, return

## Markov Decision Process (MDP) 马尔可夫决策过程

未来的只和现在的状态有关。



## Bellman Equation

$Q(s, a) = R(s) + \gamma \max \limits_{a'} Q(s',a') $



stochastic environment

$Return = Average(Q(s, a)) = E(Q(s, a))$



利用贝尔曼方程，建立有监督学习神经网络。

y 要用到的 Q 函数，我们不知道，是我们要求的，但是我们可以随机它的值，由于有值是准确的（终止状态）

训练下去，我们的函数会越来越像 Q 靠近，这样 y 会越来越准确，训练的函数也会越来越准确。

Deep Q-Network (DQN) 就是这样学习 Q 函数的神经网络。



## 改进网络结构

把 action 的选择从 input 放到 output，这样，一个状态只要进入一次我们拟合的 Q 函数。



## Epsilon-Greedy policy

0.95 的概率还是去选 action max Q(s, a). Greedy, **Exploitation**

0.05 ($\epsilon$) 的概率去随机选 action. 	**Exploration**

防止在过程中，有些选择没有怎么被怎么选到，使得策略更加合理。

**trick**

一开始 $\epsilon$ 大，后面减小。 1.0 -> 0.01.

选参数更讲究了。



## mini-batch 和 soft update

当整个数据集太大，进行一次更新参数的计算量太大，选择用小批次来更新，比如总共1e9，可以分为若干个1e4。

$W = 0.01W_{new} + 0.99W$ （主要用于强化学习，因为不知道学得是不是对的（（
