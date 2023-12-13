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

