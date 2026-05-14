import torch
import torch.nn as nn

class Fully_Connected(nn.Module):

    def __init__(self, in_features, out_features):
        super().__init__()

        weight = torch.randn(out_features, in_features)
        bias = torch.randn(out_features)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

    def forward(self, x):
        output = (x @ self.weight.T) + self.bias
        return output

Layer = Fully_Connected(3,2)

x = torch.tensor([[1.0,2.0,3.0]])

y = Layer(x)

print(y)
print(Layer.weight)
print(Layer.bias)