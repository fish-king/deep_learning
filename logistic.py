import numpy as np
from sklearn.datasets import make_classification
def init_size(X,Y,hide_unit):
    n_0 = X.shape[0] #输入层的节点数量
    n_1 = 4 #隐藏层层的节点数量
    n_2 = Y.shape[0] #输出的节点数量
    return (n_0,n_1,n_2)

def init_para(n_0,n_1,n_2):#参数初始化
    W1 = np.random.randn(n_1,n_0) 
    B1 = np.zeros(shape=(n_1, 1))
    W2 = np.random.randn(n_2,n_1)
    B2 = np.zeros(shape=(n_2, 1))
    
    assert(W1.shape == ( n_1 , n_0 ))
    assert(B1.shape == ( n_1 , 1 ))
    assert(W2.shape == ( n_2 , n_1 ))
    assert(B2.shape == ( n_2 , 1 ))
    
    parameters = {"W1" : W1,"B1" : B1,"W2" : W2,"B2" : B2 }
    
    return parameters
def sigmoid(x):
    s = 1/(1+np.exp(-x))
    return s
def forward(X,parameters):#向后传播
    W1 = parameters["W1"]
    B1 = parameters["B2"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]
    
    Z1 = np.dot(W1,X) + B1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2,A1) + B2
    A2 = sigmoid(Z2)
    
    d_pre = {"Z1": Z1,"A1": A1,"Z2": Z2,"A2": A2}
    return (A2,d_pre);
def loss_func(A2,Y):#损失函数
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply((1 - Y), np.log(1 - A2))
    cost = -np.sum(logprobs) / m
    cost = float(np.squeeze(cost))
    return cost

def backward(parameters,d_pre,X,Y):#向前递推
    m = X.shape[1]
    
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    A1 = d_pre["A1"]
    A2 = d_pre["A2"]
    
    dZ2= A2 - Y
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    dB2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.multiply(np.dot(W2.T, dZ2), 1 - np.power(A1, 2))
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    dB1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)
    grads = {"dW1": dW1,
             "dB1": dB1,
             "dW2": dW2,
             "dB2": dB2 }
    
    return grads
def updatePara(parameters,grads,learning_rate):#更新参数
    W1,W2 = parameters["W1"],parameters["W2"]
    B1,B2 = parameters["B1"],parameters["B2"]
    
    dW1,dW2 = grads["dW1"],grads["dW2"]
    dB1,dB2 = grads["dB1"],grads["dB2"]
    
    W1 = W1 - learning_rate * dW1
    B1 = B1 - learning_rate * dB1
    W2 = W2 - learning_rate * dW2
    B2 = B2 - learning_rate * dB2
    
    parameters = {"W1": W1,
                  "B1": B1,
                  "W2": W2,
                  "B2": B2}
    
    return parameters
def logistic_model(X,Y,hide_unit,num_iterations):
    size = init_size(X,Y,hide_unit)
    parameters = init_para(size[0],size[1],size[2])
    W1 = parameters["W1"]
    B1 = parameters["B1"]
    W2 = parameters["W2"]
    B2 = parameters["B2"]
    for i in range(num_iterations):
        A2 , d_pre = forward(X,parameters)
        cost = loss_func(A2,Y)
        grads = backward(parameters,d_pre,X,Y)
        parameters = updatePara(parameters,grads,learning_rate = 0.5)
        if i%1000 == 0:
            print(i,"    "+str(cost))
    return parameters
def predict(parameters,X):#预测
    A2 , d_pre = forward(X,parameters)
    predictions = np.round(A2)
    
    return predictions

X,Y=make_classification(n_samples=200,n_features=2,n_redundant=0,n_informative=2,random_state=1,n_clusters_per_class=2)#测试数据
X = X.T
Y = Y.reshape((200,1)).T

parameters = logistic_model(X,Y,4,10000)
predictions = predict(parameters, X)
print ('准确率: %d' % float((np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')