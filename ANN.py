import numpy as np
input_set=np.array([[0,1,0],
                    [0,0,1],
                    [1,0,0],
                    [1,1,0],
                    [1,1,1],
                    [0,1,1],
                    [0,1,0]])#depedent varaible
labels=np.array([[1,
                  0,
                  0,
                  1,
                  1,
                  0,
                  1]])#actual outputs
labels=labels.reshape(7,1)#converting labels to vectors
np.random.seed(42)
weights=np.random.rand(3,1)
bias=np.random.rand(1)
lr=0.05 #learning rate
def sigmoid(x):
    return 1/(1+np.exp(-x))
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))
for epoch in range (25000):
    inputs =input_set
    XW=np.dot(inputs,weights)+bias
    z=sigmoid(XW)#predicted outputs
    err=z-labels#predicted outputs - actual output
    if (epoch % 1000 == 0):
        print(f"Epoch {epoch}, Error {err.sum()}")
    #print(err.sum())
    dcost=err
    depred=sigmoid_derivative(z)#derivavtive of output from activation function
    z_del=dcost*depred#gradident formula loss* dervative of output wrt to input (chain rule)
    inputs=input_set.T
    weights=weights-lr*np.dot(inputs,z_del)
    for num in z_del:
        bias=bias-lr*num

single_pt = np.array([1,0,0])
result = sigmoid(np.dot(single_pt, weights) + bias)
print(result)

