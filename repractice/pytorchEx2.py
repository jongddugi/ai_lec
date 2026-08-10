import torch

t1 = torch.tensor([1,2,3])
t2 = torch.tensor([5,6,7])

print(t1)
print(t2)
print()

t3 = t1+ t2
print(t3)
print()

t4 = torch.tensor([[10, 20, 30],[50, 60, 70]])
print(t4)
print()
print(t4+t1)
print()
#브로드 캐스팅 연산으로 진행

t5 = torch.linspace(0, 3, 10)
print(t5)
print(0)