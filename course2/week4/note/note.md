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

