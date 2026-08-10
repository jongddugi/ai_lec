import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt  # [추가] 시각화 라이브러리

# 1. 데이터 준비 및 스케일링
X_raw = torch.tensor([[50.0, 1.0], [80.0, 2.0], [100.0, 3.0], [120.0, 4.0]], dtype=torch.float32)
y = torch.tensor([[2.5], [4.1], [5.1], [6.1]], dtype=torch.float32)

X_mean = X_raw.mean(dim=0)
X_std = X_raw.std(dim=0)
X = (X_raw - X_mean) / X_std

# 2. 모델 및 최적화 설정
model = nn.Linear(in_features=2, out_features=1)
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# 3. 학습 및 Loss 기록
epochs = 100
loss_history = []  # [추가] 에포크별 손실(Loss)을 기록할 리스트

for epoch in range(1, epochs + 1):
    prediction = model(X)
    loss = criterion(prediction, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # [추가] 매 에포크의 Loss 값을 리스트에 저장 (.item()으로 스칼라 값만 추출)
    loss_history.append(loss.item())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}/{epochs} | Loss: {loss.item():.4f}")

# 4. [추가] Matplotlib을 활용한 Loss 감소 시각화
plt.figure(figsize=(8, 4.5))
plt.plot(range(1, epochs + 1), loss_history, color='royalblue', linewidth=2, label='Training Loss')

plt.title('Gradient Descent: Loss Reduction over Epochs', fontsize=13, fontweight='bold')
plt.xlabel('Epoch', fontsize=11)
plt.ylabel('MSE Loss', fontsize=11)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend()

# 그래프 출력
plt.show()