import numpy as np
from operation import Operation
from node import Node

class ReLU(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = np.maximum(0, x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad * (self.x.data > 0)
        self.x.accumulate_grad(dx)
        return dx
    
class Sigmoid(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = 1 / (1 + np.exp(-x.data))
        self.sigmoid_value = value
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad * self.sigmoid_value * (1-self.sigmoid_value)
        self.x.accumulate_grad(dx)
        return dx
    
class Softmax(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        exp_values = np.exp(x.data - np.max(x.data, axis=-1, keepdims=True))
        value = exp_values / np.sum(exp_values, axis=-1, keepdims=True)
        self.softmax_value = value
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = np.empty_like(upstream_grad)
        for i, (s_val, u_grad) in enumerate(zip(self.softmax_value, upstream_grad)):
            s_val = s_val.reshape(-1, 1)
            jacobian = np.diagflat(s_val) - np.dot(s_val, s_val.T)
            dx[i] = np.dot(jacobian, u_grad)
        
        self.x.accumulate_grad(dx)
        return dx