# Neural Network

[TOC]

![BP Neural Network](assets/BP Neural Network.png)

|  $X$  |             $W1$             |                           $S1$                            |             $W2$             |             $S2$             |
| :---: | :--------------------------: | :-------------------------------------------------------: | :--------------------------: | :--------------------------: |
| m * n |            n * h             |                           m * h                           |            h * c             |            m * c             |
|       |             $b1$             |                                                           |             $b2$             |                              |
|       |            1 * h             |                                                           |            1 * c             |                              |
|       |                              |                  **Forward Propagation**                  |                              |                              |
|       |                              |           $g(t)=max(0,t)$ (ReLU) $S1=g(XW1+b1)$           |                              |         $S2=S1W2+b2$         |
|       |                              |                           m * h                           |                              | m * c = (m * h)(h * c)+(1*c) |
|       |                              |                 **Backward Propagation**                  |                              |                              |
|       |                              | $\delta1=\delta2{W2}^T.*\frac{\partial g(t)}{\partial t}$ |                              |      $\delta2=(S2-y)/m$      |
|       |                              |             (m * h) = (m * c)(c * h).*(m * h)             |                              |           (m * c)            |
|       |      $dW1={X}^T\delta1$      |                                                           |     $dW2={S1}^T\delta2$      |                              |
|       |   (n * h) = (n * m)(m * h)   |                                                           |   (h * c) = (h * m)(m * c)   |                              |
|       | $db1=np.sum(\delta1,axis=0)$ |                                                           | $db2=np.sum(\delta2,axis=0)$ |                              |
|       |            1 * h             |                                                           |            1 * c             |                              |

---

```python
# delta2
delta2 = probs # N * C
delta2[range(N),y] -= 1 # y: 1 * N
delta2 /= N
# W2 and b2
dW2 = scores1.T.dot(delta2) # H * C
db2 = np.sum(delta2, axis=0) # 1 * C
# delta1
dscores1 = scores1 # N * H
dscores1[dscores1>0] = 1
delta1 = delta2.dot(W2.T)*dscores1 # N * H
# W1 and b1
dW1 = X.T.dot(delta1) # D * H
db1 = np.sum(delta1, axis=0) # 1 * H
# Add regualrization
dW2 += 2 * reg * W2
dW1 += 2 * reg * W1
```





