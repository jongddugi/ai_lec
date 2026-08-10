import torch

# t1 = torch.tensor([1,2,3,4,5,6])
# print(t1)
# print()

# t2 = t1.view(2,3)#reshape와 유사 -> view는 참조만 논리적으로 바꾸어줌
# print(t2)
# print()

# print(t1.reshape(2,3))#실제로 바꿈
# print()

# # 데이터를 우리가 원하는 형태로 바꾸어 주기 위해서 필요함
# t3 = torch.tensor([[1,2], [3,4], [5,6]])
# print(t3)
# print()
# print(t3.view(-1))#풀거임
# print()

# print(t3.view(1,-1))#풀건데 1열로 만들기
# print()

# print(t3.view(2,-1))#풀건데 2열로 만들기
# print()

# print(t3.view(3,-1))#풀건데 3열로 만들기
# print()

# print(t3.view(6,-1))#풀건데 3열로 만들기
# print()


t3 = torch.tensor([[1,2,3],[4,5,6]])
t4 = torch.tensor([[10, 20, 30],[40, 50, 60]])
print(t3)
print()
print(t4)
print()

print(torch.cat([t3,t4], dim=0))
print()
print(torch.cat([t3,t4], dim=1))
print()
print(torch.cat([t3,t4], dim=-1))

t5 = torch.tensor([[1,2],[3,4]])
print()
print(torch.cat([t3,t5], dim=1))
print()
t6 = torch.tensor([[1,2,3]])
print(torch.cat([t3,t6],dim=0))














