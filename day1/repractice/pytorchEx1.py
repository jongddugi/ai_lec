import torch

t1 = torch.FloatTensor([[1,2],[3,4]])
print(t1)
print(type(t1))
print(t1.size())
print()


t2 = torch.tensor([[4,6],[9,10]],dtype=torch.float32)
print(t2)
print(type(t2))
print(t2.size())
print(t2.dtype)

print(t2. numpy())
print()

import numpy as np

ndata = np.array([[1,2,3,4],[5,6,7,8]], dtype = np.float32)
t3=torch.from_numpy(ndata)
print(t3)
