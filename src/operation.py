class Operation:
    def forward(self, *inputs):
        raise NotImplementedError
    
    def backward(self, upstream_grad):
        raise NotImplementedError