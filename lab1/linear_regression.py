import matplotlib.pyplot as plt

X = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
Y = [1, 1, 2, 4, 5, 7, 8, 9, 9, 10]

plt.figure(figsize=(5, 3))
plt.plot(X, Y)
plt.scatter(X, Y)
plt.show()

class H():
    def __init__(self, w): 
        self.w = w
    
    def forward(self, x):
        return self.w * x
    
def cost(h, X, Y):
    error = 0
    for i in range(len(X)):
        error += (h.forward(X[i]) - Y[i]) **2
    error = error / len(X)
    return error

h = H(4)
pred_y = h.forward(5)
print('value of f(5) :', pred_y)
print('value of w :', h.w)

list_w = []
list_c = []
for i in range(-100, 100):
    w = i * 0.1
    h = H(w)
    c = cost(h, X, Y)
    list_w.append(w)
    list_c.append(c)
print('list of w', list_w)
print('list of c', list_c)
plt.xlabel('w')
plt.ylabel('cost')
plt.scatter(list_w, list_c, s = 3)
plt.show()

def cal_grad(w, cost):
    h = H(w)
    cost1 = cost(h, X, Y)
    eps = 0.001
    h = H(w+eps)
    cost2 = cost(h, X, Y)
    dcost = cost2- cost1
    dw = eps
    grad = dcost / dw
    return grad, (cost1 + cost2)*0.5

def cal_grad2(w, cost):
    h = H(w)
    grad = 0
    for i in range(len(X)):
        grad += 2 * (h.forward(X[i]) - Y[i]) * X[i]
    grad = grad / len(X)
    c = cost(h, X, Y)
    return grad, c

w1 = 4
w2 = 4
lr = 0.1

list_w1 = []
list_c1 = []
list_w2 = []
list_c2 = []

for i in range(10):
    grad, mean_cost = cal_grad(w1, cost)
    grad2, mean_cost2 = cal_grad2(w2, cost)

    w1 -= lr *grad
    w2 -= lr * grad2
    list_w1.append(w1)
    list_w2.append(w2)
    list_c1.append(mean_cost)
    list_c2.append(mean_cost2)

plt.scatter(list_w1, list_c1, label='analytic', marker='*', color='blue')
plt.scatter(list_w2, list_c2, label='formula', color='red')
plt.show()