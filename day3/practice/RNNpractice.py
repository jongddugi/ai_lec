import torch
import torch.optim as optim
import numpy as np

sentence = ("if you want to build a ship, don't drum up people together to "
            "collect wood and don't assign them tasks and work, but rather "
            "teach them to long for the endless immensity of the sea")
print(sentence)
char_set = list(set(sentence))
print(char_set)
char_dic = {c:i for i, c in enumerate(char_set)}
print(char_dic)

dic_size = len(char_dic)
print(dic_size)#유니크한 문자의 개수

hidden_size = 128#dic_size보다 넉넉하게 키워서 표현력을 높임
sequence_length = 10
learning_rate = 0.01#Adam 기준 0.1은 너무 커서 학습이 불안정해짐 -> 0.01로 완화

x_data = []
y_data = []
for i in range(0, len(sentence)- sequence_length):
    x_str = sentence[i:i+sequence_length]#바로 직전 값을 가지고 하기 때문에 i->f->ws(white space)-> y
    y_str = sentence[i+1: i+sequence_length+1]#하나뒤에 정답값
    #print(i, x_str, '->', y_str)

    x_data.append([char_dic[c] for c in x_str])
    y_data.append([char_dic[c] for c in y_str])

print(x_data[0])#10개인 이유는 sequence_length를 10으로 맞춰줬기 때문임
print(y_data[0])
print()

#print(np.eye(10))
x_one_hot = [np.eye(dic_size)[x] for x in x_data]

# eye : [[1000]
#        [0100]
#        [0010]
#        [0001]] eye가 이런 행렬을 만드는 역할을 함
print(x_one_hot[0])
x = torch.FloatTensor(x_one_hot)#실수 : 딥러닝 형태는 입력은 실수여야함
y = torch.LongTensor(y_data)#정수 : 결과값인 인덱스는 정수값임
print(x.shape)
print(y.shape)

import torch.nn as nn

class RNNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, layers):
        super().__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, num_layers=layers, batch_first=True)#RNN -> LSTM으로 교체 (장기 의존성 학습에 유리)
        self.fc = nn.Linear(hidden_size, output_size)#hidden_size로 커진 만큼 출력은 dic_size로 다시 맞춰줌

    def forward(self, x):
        output, _ = self.rnn(x)
        y = self.fc(output)
        return y

model = RNNet(dic_size, hidden_size, dic_size, layers=2)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

outputs = model(x)
print(outputs.shape)
print(outputs.view(-1, dic_size).shape)

for epoch in range(500):#100 -> 500으로 늘려서 충분히 수렴하도록 함
    optimizer.zero_grad()
    hypothesis = model(x)
    loss = loss_func(hypothesis.view(-1, dic_size), y.view(-1)) #i는 f가 정답이고 f는 white space가 정답이 되게 설정
    loss.backward()
    optimizer.step()

    predictions = hypothesis.argmax(dim=2)
    #print(predictions.shape)
    predict_str = ''
    for j, result in enumerate(predictions):
        if j == 0:
            predict_str += ''.join([char_set[t] for t in result])
        else:
            predict_str += char_set[result[-1]]

    if epoch % 50 == 0 or epoch == 499:
        print(epoch, loss.item())#loss 수치를 함께 찍어서 수렴 여부를 눈으로 확인
    print(predict_str)
