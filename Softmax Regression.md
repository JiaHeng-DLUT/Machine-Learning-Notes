# Softmax Regression

[TOC]

## 1 Predict Function

- [ ] Todo

## 2 Loss Function

$$
f(x_i; W) =  W x_i\\
$$

Interpret scores as the unnormalized log probabilities for each class and replace the *hinge loss* with a **cross-entropy loss (交叉熵损失)**.

>Possibly confusing naming conventions. To be precise, the SVM classifier uses the hinge loss, or also sometimes called the max-margin loss. The Softmax classifier uses the cross-entropy loss. The Softmax classifier gets its name from the softmax function, which is used to squash the raw class scores into normalized positive values that sum to one, so that the cross-entropy loss can be applied. In particular, note that technically it doesn’t make sense to talk about the “softmax loss”, since softmax is just the squashing function, but it is a relatively commonly used shorthand.

$$
L_i = -\log\left(\frac{e^{f_{y_i}}}{ \sum_j e^{f_j} }\right)\\
\text{or equivalently}\\
L_i = -f_{y_i} + \log\sum_j e^{f_j}
$$
The function $f_j(z) = \frac{e^{z_j}}{\sum_k e^{z_k}}$ is called the **softmax function**: It takes a vector of arbitrary real-valued scores (in $z$) and squashes it to a vector of values between zero and one that sum to one. 

## 3 Find $\theta$

- [ ] Todo

## 4 Interpretation

### Information Theory View

The *cross-entropy* between a “true” distribution $p$ and an estimated distribution $q$ is defined as:
$$
H(p,q) = - \sum_x p(x) \log q(x)
$$
The Softmax classifier is hence minimizing the cross-entropy between the estimated class probabilities ($q = e^{f_{y_i}}  / \sum_j e^{f_j}$ as seen above) and the “true” distribution, which in this interpretation is the distribution where all probability mass is on the correct class (i.e. $p=[0,…1,…,0]$ contains a single 1 at the $y_i$ -th position.). 

### Probabilistic Interpretation

$$
P(y_i \mid x_i; W) = \frac{e^{f_{y_i}}}{\sum_j e^{f_j} }
$$

The (normalized) probability assigned to the correct label $y_i$ given the image $x_i$ and parameterized by $W$. To see this, remember that the Softmax classifier interprets the scores inside the output vector $f$ as the unnormalized log probabilities. Exponentiating these quantities therefore gives the (unnormalized) probabilities, and the division performs the normalization so that the probabilities sum to one. 

## 5 Practical Issues: Numeric Stability

When you’re writing code for computing the Softmax function in practice, the intermediate terms $e^{f_{y_i}}$ and $\sum_j e^{f_j}$ may be very large due to the exponentials. Dividing large numbers can be numerically unstable, so it is important to use a normalization trick. 
$$
\frac{e^{f_{y_i}}}{\sum_j e^{f_j}}
= \frac{Ce^{f_{y_i}}}{C\sum_j e^{f_j}}
= \frac{e^{f_{y_i} + \log C}}{\sum_j e^{f_j + \log C}}
$$
We are free to choose the value of $C$. This will not change any of the results, but ==we can use this value to improve the numerical stability of the computation==. A common choice for $C$ is to set $\log C = -\max_j f_j$. This simply states that we should shift the values inside the vector $f$ so that the highest value is zero. In code:

```python
f = np.array([123, 456, 789]) # example with 3 classes and each having large scores
p = np.exp(f) / np.sum(np.exp(f)) # Bad: Numeric problem, potential blowup

# instead: first shift the values of f so that the highest number is 0:
f -= np.max(f) # f becomes [-666, -333, 0]
p = np.exp(f) / np.sum(np.exp(f)) # safe to do, gives the correct answer
```

## 5 Softmax VS. Logistic Regression

- Softmax classifier is Logistic Regression classifier’s generalization to multiple classes.

## 6 Softmax VS. SVM

![img](assets/svmvssoftmax.png)

|                           Softmax                            |                             SVM                              |
| :----------------------------------------------------------: | :----------------------------------------------------------: |
|           provides “probabilities” for each class            |                                                              |
| gives a slightly more intuitive output (normalized class probabilities) and also has a probabilistic interpretation | treats the outputs $f(x_i, W)$ as (uncalibrated and possibly difficult to interpret) scores for each class |
| never fully happy with the scores it produces: the correct class could always have a higher probability and the incorrect classes always a lower probability and the loss would always get better | happy once the margins are satisfied and it does not micromanage the exact scores beyond this constraint |

The performance difference between the SVM and Softmax are usually very small, and different people will have different opinions on which classifier works better. 








