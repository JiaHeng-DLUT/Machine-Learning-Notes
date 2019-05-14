# Neural Network

[TOC]

## 1 Forward Propagation

## 2 Loss (Softmax Loss)

### 2.1 Example



### 2.2 Code

```python
max_scores = np.reshape(np.max(scores, axis=1), (N,1))
exp_scores = np.exp(scores-max_scores)
probs = exp_scores/np.sum(exp_scores, axis=1, keepdims=True)
loss = np.sum(-np.log(probs[range(N),y]))
loss /= N
loss += reg * np.sum(W1 * W1) + reg * np.sum(W2 * W2)
```

## 3 Backward Propagation



