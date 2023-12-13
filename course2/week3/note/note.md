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





