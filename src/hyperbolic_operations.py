import numpy as np
from operation import Operation
from node import Node

class Sinh(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = np.sinh(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad * np.cosh(self.x.data)
        self.x.accumulate_grad(dx)
        return dx
    
class Cosh(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = np.cosh(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad * np.sinh(self.x.data)
        self.x.accumulate_grad(dx)
        return dx
    
class Tanh(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = np.tanh(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = upstream_grad * (1 - (np.tanh(self.x.data) ** 2))
        self.x.accumulate_grad(dx)
        return dx
    
class Coth(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = 1 / np.tanh(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = -upstream_grad / (np.sinh(self.x.data) ** 2)
        self.x.accumulate_grad(dx)
        return dx
    
class Sech(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = 1 / np.cosh(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = -upstream_grad  * np.sinh(self.x.data) / (np.cosh(self.x.data) ** 2)
        self.x.accumulate_grad(dx)
        return dx
    
class Cosech(Operation):
    def forward(self, x: Node) -> Node:
        self.x = x
        value = 1 / np.sinh(x.data)
        return Node(value)
    
    def backward(self, upstream_grad):
        dx = -upstream_grad * np.cosh(self.x.data) / (np.sinh(self.x.data) ** 2)
        self.x.accumulate_grad(dx)
        return dx
    
