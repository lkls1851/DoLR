import torch

a=torch.tensor([[1,2],[3,4]])
b=torch.tensor(([[5,6],[7,8]]))
print(a)
print(b)
print((a*b).sum())