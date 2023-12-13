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

