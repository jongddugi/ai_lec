import torch

# t1= torch.tensor([1,2,3])
# t2 = torch.tensor([5,6,7])

# print(t1)
# print(t2)
# print()

# t3 = t1+ t2
# print(t3)
# print()

# t4 = torch.tensor([[10, 20, 30],[50, 60, 70]])
# print(t4)
# print()
# print(t4+t1)
# print()
# #브로드캐스팅 연산

# t5 = torch.linspace(0, 3, 10)
# print(t5)
# print()

# print(torch.exp(t5))
# print()
# print(torch.log(t5))
# print()
# print(torch.cos(t5))
# print()
# print(torch.sin(t5))
# print()
# print(torch.sqrt(t5))
# print()
# print(torch.mean(t5))
# print()

t6 = torch.tensor([[2,3,0],[90, 50,70]])
print(t6)
print()
print(torch.max(t6))
print()
print(torch.max(t6, dim=1))
# value는 각각 제일 큰 값을 반환, 각 index위치 
#이터레이터 제네레이터 공부 필
print()
print(torch.max(t6,dim=1)[1])

print()
print([3, 90])












