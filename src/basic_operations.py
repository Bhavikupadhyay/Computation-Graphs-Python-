from operation import Operation
from node import Node
import numpy as np


class Add(Operation):
    def forward(self, x: Node, y: Node) -> Node:
        self.x = x
        self.y = y
        value = x.data + y.data
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad
        dy = upstream_grad

        dx = np.broadcast_to(dx, self.x.data.shape)
        dy = np.broadcast_to(dy, self.y.data.shape)

        self.x.accumulate_grad(dx)
        self.y.accumulate_grad(dy)
        return dx, dy
    

class Subtract(Operation):
    def forward(self, x: Node, y: Node) -> Node:
        self.x = x
        self.y = y
        value = x.data - y.data
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad
        dy = -upstream_grad

        dx = np.broadcast_to(dx, self.x.data.shape)
        dy = np.broadcast_to(dy, self.y.data.shape)

        self.x.accumulate_grad(dx)
        self.y.accumulate_grad(dy)
        return dx, dy


class HadamardMultiply(Operation):
    def forward(self, x: Node, y: Node) -> Node:
        self.x = x
        self.y = y
        value = x.data * y.data
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad * self.y.data
        dy = upstream_grad * self.x.data

        dx = np.broadcast_to(dx, self.x.data.shape)
        dy = np.broadcast_to(dy, self.y.data.shape)

        self.x.accumulate_grad(dx)
        self.y.accumulate_grad(dy)

        return dx, dy
    

class Multiply(Operation):
    def forward(self, x: Node, y: Node) -> Node:
        if x.data.shape[1] != y.data.shape[0]:
            raise ValueError(f"Multiply operation: Shapes of x {x.data.shape} and y {y.data.shape} do not match")
        
        self.x = x
        self.y = y
        value = np.dot(x.data, y.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = np.dot(upstream_grad, self.y.data.T)
        dy = np.dot(self.x.data.T, upstream_grad)

        self.x.accumulate_grad(dx)
        self.y.accumulate_grad(dy)

        return dx, dy
    

class Divide(Operation):
    def forward(self, x: Node, y: Node) -> Node:
        self.x = x
        self.y = y

        value = x.data / y.data
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad / self.y.data
        dy = -upstream_grad * self.x.data / (self.y.data ** 2)

        dx = np.broadcast_to(dx, self.x.data.shape)
        dy = np.broadcast_to(dy, self.y.data.shape)

        self.x.accumulate_grad(dx)
        self.y.accumulate_grad(dy)

        return dx, dy