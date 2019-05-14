# Neural Network

[TOC]





```python
# delta2
delta2 = probs # N * C
delta2[range(N),y] -= 1 # y: 1 * N
delta2 /= N
# W2 and b2
dW2 = scores1.T.dot(delta2) # H * C
db2 = np.sum(delta2, axis=0, keepdims = True) # 1 * C
# delta1
dscores1 = scores1 # N * H
dscores1[dscores1>0] = 1
delta1 = delta2.dot(W2.T)*dscores1 # N * H
# W1 and b1
dW1 = X.T.dot(delta1) # D * H
db1 = np.sum(delta1, axis=0, keepdims = True) # 1 * H
# Add regualrization
dW2 += 2 * reg * W2
dW1 += 2 * reg * W1
```



