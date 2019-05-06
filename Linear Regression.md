# Linear Regression

[TOC]

## 1 Predict Function

|           | $f(X^{(i)};W)=WX^{(i)}$ |
| :-------: | :---------------------: |
|    $W$    |        $k*(n+1)$        |
| $X^{(i)}$ |        $(n+1)*1$        |
|    $f$    |          $k*1$          |

|       | $f(X;W)=XW^T$ |
| :---: | :-----------: |
|  $X$  |   $m*(n+1)$   |
| $W^T$ |   $(n+1)*k$   |
|  $f$  |     $m*k$     |

==Every row of $W$ is a classifier for one of the classes.==

![img](assets/wb.jpeg)

## 2 Loss Function

- Without regularization

$$
L(W)=-\frac{1}{m}\sum_{i=1}^m\left[y^{(i)}log\left(f(X^{(i)};W)\right)+\left(1−y^{(i)}\right)log\left(1−f(X^{(i)};W)\right)\right]
$$

## 3 Find $\theta$

### 3.1 Gradient Descent

$$
\theta:=\theta−\frac{\alpha}{m}X^T(g(XW^T)−y)
$$

### 3.2 Normal Equation

$$
\begin{align*}
& \theta = \left( X^TX \right)^{-1} X^Ty \\
& \theta = \left( X^TX + \lambda \cdot L \right)^{-1} X^Ty \newline& \text{where}\ \ L = \begin{bmatrix} 0 & & & & \newline & 1 & & & \newline & & 1 & & \newline & & & \ddots & \newline & & & & 1 \newline\end{bmatrix}
\end{align*}
$$

- L is a matrix with 0 at the top left and 1’s down the diagonal, with 0’s everywhere else. 
- $$L: (n+1)×(n+1)$$
- $$\lambda:\ $$a real number

### 3.3 Gradient Descent VS. Normal Equation

|      Gradient Descent      |     Normal Equation     |
| :------------------------: | :---------------------: |
|    Need to choose alpha    | No need to choose alpha |
|   Needs many iterations    |   No need to iterate    |
|        $$O(kn^2)$$         |       $$O(n^3)$$        |
| Works well when n is large | Slow if n is very large |
|         n > 10,000         |       n < 10,000        |

**Pay attention to:**

1. ==No need to do feature scaling with the normal equation.==
2. When implementing the normal equation in octave we want to ==use the `pinv` function rather than `inv`==. The `pinv` function will give you a value of $\theta$ even if $X^TX$ is not invertible.
3. If $X^TX$ is **noninvertible**, the common causes might be having :
   1. Redundant features, where two features are very closely related (i.e. they are linearly dependent)
   2. Too many features (e.g. m ≤ n). In this case, delete some features or use regularization.
   3. If m < n, then $X^TX$ is non-invertible. However, ==when we add the term $\lambda L$, then $X^TX + \lambda L$ becomes invertible.==
4. Solutions to the above problems include deleting a feature that is linearly dependent with another or deleting one or more features when there are too many features.

