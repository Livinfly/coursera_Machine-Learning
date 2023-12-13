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

