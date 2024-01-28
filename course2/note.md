$w \cdot a + b$ ，思考得到的这个节点的值是由和上一层的所有节点的连线权值决定的。

forward propagation - inference

backward propagation - learning

## TF

linear_layer = tf.keras.layers.Dense(units=1, activation = 'linear', )

中的 linear_layer 方法，返回的是 infer 的值

```python
pos = Y_train == 1
neg = Y_train == 0
X_train[pos]
```

```python
np.array([200, 17]) 	# 一维 2
np.array([[200, 17]]) 	# 二维 1x2
```

Tile/copy our data to increase the training set size and reduce the number of training epochs.

```python
Xt = np.tile(Xn,(1000,1))
Yt= np.tile(Y,(1000,1)) 
```

标准化，需要先根据数据学均值、方差之类的；之后测试也是要对数据先进行标准化。

```python
norm_l = tf.keras.layers.Normalization(axis=-1)
norm_l.adapt(X)  # learns mean, variance
Xn = norm_l(X)

X_testn = norm_l(X_test)

yhat = (predictions >= 0.5).astype(int)
```

dot(a, b)

a^{T}b

logistic loss  <---> know as binary cross entropy - 主要用于二分类问题



---



## Activation function

ReLU (Rectified linear unit)   ---  max(x, 0)

softmax()  								--- e^{x_i} / sigma e^{x_j}

## How to choose in Output-Layer

-   Binary classification - y = 0/1 -> Sigmoid
-   Regression - y = +/- -> Linear activation function
-   Regression - y = 0 or + -> ReLU

## Why ReLU no Sigmoid in Hidden-Layer

-   faster
-   ReLU is flat and not flat, Sigmoid is flat and flat, what make GD slow.

## softmax

loss - SparseCategoricalCrossentropy

-log(a)



使用 softmax 的时候，建议把输出层不用激活函数（Linear），

```python
model.compile(loss= ...(from_logits=True)) # loss 采用对应的 loss 。
```

如果直接在层内部跑，会先把 z 算出来，而里面的指数函数比较大，容易引起精度误差，影响效率和性能。

同样的，在 sigmoid 中也可以这样处理。（在 sigmoid 里面的影响没这么大）

这样输出的值是没有经过激活函数，需要再加。

```python
logits = model(X)
f_x = tf.nn.softmax(logits)
np.argmax(f_x)
```

## Adam

Adam: **Ada**ptive **M**oment estimation

```python
model = Sequential([
    ...
])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3), ...)
```

## Convolutional-Layer

这样看，之前一直用的 Dense-Layer 是全连接层。

卷积层，更通用的理解应该就是部分连接。

相比 Dense 优势在于 更快、需要的训练数据更少、泛化性更好。

## FP && BP

-   前向传播
    -   从左往右，都算出来。
-   反向传播
    -   从右往左，计算梯度，根据链式法则。
-   复杂度 N + P （点数 + 参数数）



sympy 可以用来符号求导。



## Assignment

model 设置层的时候，如果没有设置输入层的参数的话，不算建立。

因为没有说明 input_dim（其实也就是特征的维度了） 或者可以再第一层隐藏层说明这个参数。

```python
model = Sequential(
    [               
        ### START CODE HERE ### 
#         tf.keras.Input(shape=(400,)),
        Dense(units=25, input_dim=1, activation="relu"),
        Dense(units=15, activation="relu"),
        Dense(units=10, activation="linear")
        ### END CODE HERE ### 
    ], name = "my_model" 
)
model.summary()
```



---



## Debug a learning algorithm

误差很大，可以尝试：

-   更多训练数据         - - 解决过拟合
-   多/少一点特征        - 解决过/欠拟合
-   加多项式特征         - 解决欠拟合
-   增/减正则化的 $\lambda$    - 解决过/欠拟合

## Evaluating model

数据分成 training set, test set，例如 7 : 3 。

数据分成 training set, cross validation (dev set), test set, 6 : 2 : 2。

训练，选择拟合较好的，测试泛化能力。





test set 只在用 dev set 选出最好模型之后。

diagnostic 是看他work不work的原因。



## 等价

```python
print(f"training MSE (using sklearn function): {mean_squared_error(y_train, yhat) / 2}")

# for-loop implementation
total_squared_error = 0

for i in range(len(yhat)):
    squared_error_i  = (yhat[i] - y_train[i])**2
    total_squared_error += squared_error_i                                              

mse = total_squared_error / (2*len(yhat))

print(f"training MSE (for-loop implementation): {mse.squeeze()}")
```

## Bias & Variance

High bias 			-	 underfit		- 	J_train h, J_cv h

High variance	 -	 overfit		   - 	J_train l,  J_cv h

$\lambda$ large -> underfit, small -> overfit.



```python
layer = Dense(..., kernel_regularizer=L2(0,01)) # lambda
```



error analysis，人工去看误差大的有什么特征，分析改善模型。

data augmentation，增加数据，增加新的特征、从已有的去变化（例如图像旋转，添加噪声），生成和真实类似的数据。



fine-tune 的方法，Transfer Learning，迁移学习。解决相似类型的模型，用已训练好的（一般是更大的），只训练输出层，或者再训练全部参数。



## 道德问题

和别人讨论，可能的风险。

查找这个领域出现过的事件。

在发布前就有审计系统



## skewed data

数据的答案分布不是50-50的，设比较少的类别为 a。

-   precision 精准度，预测为 a 类别的中真实是 a 类别的百分比。
-   recall ，真实是 a 类别的中预测为 a 类别的百分比。

如果模型不分析，直接判断，会导致两个值为 0。



F1 score 通过 P 和 R 判断好坏

P 和 R 的调和级数。harmonic mean

$\frac{1}{\frac{1}{2}(\frac{1}{P} + \frac{1}{R})} = \frac{2PR}{P+R}$





---



## Decision Tree

决策树的过程用了信息论的东西吧，熵什么的。

entropy: $H(p_1) = -p_1\log_2(p_1) - -p_0\log_2(p_0) = -p_1\log_2(p_1) - (1-p_1)\log_2(1- p_1)$ .

信息增益（Information Gain）: $H(P_1^{root} - (w^{left}H(p_1^{left} + w^{right}H(p_1^{right})))$ .

递归地选取一个特征，按照是或者不是的情况进行分裂出子节点。

在纯度100%、达到最大深度、信息增益小于某个临界值、节点的样本数小于某个临界值停止。

### one-hot encoding

把一个有多个特征值的，分成特征值 1，0。

### 连续变量

有 n 个样本，就选 n-1 个中间点把样本分成两边，比较，选取类似，最后取值是所在节点的均值。

### Tree ensemble

多个决策树，每棵树能有 vote。

### simple with replacement

有放回的取样。

Bagging，使产生的决策树不同，能把所有 simple 作为 pool，然后在里面有放回的抽取，抽到和 original 一样的规模后结束。

### Random forest algorithm

由于 Bagging 产生的变化不大，在 Bagging 的基础上，加上 n 个 feature 只随机选 k 个进行比较选取，来产生差异化相比大一点的决策树，k 一般取 $\sqrt{n}$ 。

### Boosted Tree

选择样本时，在 Bagging 的基础上，更高概率地选择前一棵决策树中错误分类的样本。

### XGBoost (eXtreme Gradient Boosting)

实现比较复杂，一般调库。（x）

```python
from xgboost import XGBClassifier, XGBRegression
model = XGBClassifier() # 如果是回归问题，XGBRegression
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

## 什么时候用决策树

-   决策树
    -   在表格化（结构化）的数据上效果不错，不建议用于非结构化（图片、音频、文本）
    -   快，训练
    -   小的决策树，是可解释的
-   神经网络
    -   所有数据类型都行
    -   大部分比较慢，训练
    -   可以用 transfer learning 去 fune-tune

## 其他

```python
np.log()  # ln
np.log2() # log2
```

