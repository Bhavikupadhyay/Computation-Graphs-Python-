import numpy as np
from operation import Operation
from node import Node

class Sin(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = np.sine(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad * np.cos(self.x.data)
        self.x.accumulate_grad(dx)
        return dx
    
class Cos(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = np.cos(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = -upstream_grad * np.sin(self.x.data)
        self.x.accumulate_grad(dx)
        return dx
    
class Tan(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = np.tan(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad / (np.cos(self.x.data) ** 2)
        self.x.accumulate_grad(dx)
        return dx
    
class Cot(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = 1 / np.tan(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = -upstream_grad / (np.sin(self.x.data) ** 2)
        self.x.accumulate_grad(dx)
        return dx
    
class Sec(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = 1 / np.cos(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad * np.sin(self.x.data) / (np.cos(self.x.data)**2)
        self.x.accumulate_grad(dx)
        return dx
    
class Cosec(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = 1 / np.cos(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = -upstream_grad * np.cos(self.x.data) / (np.sin(self.x.data)**2)
        self.x.accumulate_grad(dx)
        return dx