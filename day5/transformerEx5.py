import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 1. Multi-Head--Attention 클래스
# ============================================================

class SimpleMultiHeadAttention(nn.Module):

    def __init__(
        self,
        embedding_dim,
        num_heads
    ):
        super().__init__()

        assert embedding_dim % num_heads ==0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        #하나의 head가 담당하는 차원
        self.head_dim = embedding_dim // num_heads
        

        # 입력 벡터를 Query로 변환하는 Linear
        self.query = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        # 입력 벡터를 Key로 변환하는 Linear
        self.key = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False,
        )

        # 입력 벡터를 Value로 변환하는 Linear
        self.value = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False, #각 노드에 있는 bias를 안쓰겠다. 노드편향을 줄이겠다. 기본값은 True이므로
        )


        #여러 head의 결과를 다시 합친 후 적용할 linear

        self.output_linear = nn.Linear(
            embedding_dim,
            embedding_dim,
            bias=False
        )


    def forward(
        self,
        x,
    ):
        # ----------------------------------------------------
        # x shape
        #[batch_size, sequence_length, embedding_dim]
        # ----------------------------------------------------

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        # ----------------------------------------------------
        # 1. 입력 X로부터 Q, K, V 생성
        # ----------------------------------------------------

        Q = self.query(x)
        K = self.key(x)
        V = self.value(x)

        #현재 shape
        '''
        Q : [batch, seq, embedding]
        K : [batch, seq, embedding]
        V : [batch, seq, embedding]
        '''


        # ----------------------------------------------------
        # 여러 Head로 분리
        # ----------------------------------------------------

        Q = Q.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )

        K = K.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )


        V = V.reshape(
            batch_size,
            sequence_length,
            self.num_heads,
            self.head_dim
        )


        # [batch, sequence, head, head_dim]


        # ----------------------------------------------------
        # Head를 앞으로 이동
        # ----------------------------------------------------

        Q = Q.transpose(1,2) #1번 2번 바꾸기 batch :0, sequence : 1, head : 2 ;;;
        K = K.transpose(1,2) 
        V = V.transpose(1,2) 

        # ----------------------------------------------------
        # key의 마지막 두 차원 교환
        # ----------------------------------------------------

        K_T = K.transpose(
            -2,
            -1,
        )


        # ----------------------------------------------------
        # 2. Q와 K의 유사도 계산
        # ----------------------------------------------------

        attention_score = (
            Q @ K_T
        )

        # [batch, sequence, head, head_dim]



        # ----------------------------------------------------
        # 3. Scaling
        # ----------------------------------------------------





        scaled_score = (
            attention_score
            / (self.head_dim ** 0.5)
        )

        # ----------------------------------------------------
        # 4. Softmax
        # ----------------------------------------------------

        attention_weights = F.softmax(
            scaled_score,
            dim=-1,
        )


        # ----------------------------------------------------
        # 5. 각 Head의 Attention 결과
        # ----------------------------------------------------

        output = (
            attention_weights @ V
        )

        #[batch, head, sequence, head_dim]


        # ----------------------------------------------------
        # Head 위치를 다시 원래대로 이동
        # ----------------------------------------------------
        
        head_output = output.transpose(1,2)


        #[batch, sequence, head, head_dim]


        # ----------------------------------------------------
        # 여러 Head를 하나로 합치기
        # ----------------------------------------------------

        concat_output = head_output.reshape(
            batch_size, sequence_length, self.embedding_dim
        )

        #[batch, sequence, embedding]

        output = self.output_linear(concat_output)

        return (
            output,
            attention_weights,
            Q,
            K,
            V,
            head_output,
            concat_output
        )


# ============================================================
# 2. 입력 데이터
# ============================================================

words = [
    "나는",
    "사과를",
    "먹는다",
]



sentence =  [
            [1.0, 0.0, 1.0, 0.0],   # 나는
            [0.0, 2.0, 0.0, 2.0],   # 사과를
            [1.0, 1.0, 1.0, 1.0],   # 먹는다
            ]
    
X = torch.tensor(
    [
        sentence
    ],
    dtype = torch.float32
)


print("=" * 60)
print("입력 X")
print("=" * 60)

print(X)

print(
    "\nX shape:",
    X.shape,
)


# ============================================================
# 3. 모델 생성
# ============================================================

torch.manual_seed(42)

embedding_dim = 4
num_heads = 2

model = SimpleMultiHeadAttention(
    embedding_dim=embedding_dim,
    num_heads = num_heads
)


# ============================================================
# 4. 실행
# ============================================================

(
    output,
    attention_weights,
    Q,
    K,
    V,
    head_output,
    concat_output
) = model(X)


# ============================================================
# 5. Q 확인
# ============================================================

print("\n" + "=" * 60)
print("Query")
print("=" * 60)

print(Q)


# ============================================================
# 6. K 확인
# ============================================================

print("\n" + "=" * 60)
print("Key")
print("=" * 60)

print(K)


# ============================================================
# 7. V 확인
# ============================================================

print("\n" + "=" * 60)
print("Value")
print("=" * 60)

print(V)


# ============================================================
# 8. Attention Weight 확인
# ============================================================

print("\n" + "=" * 60)
print("Attention Weights")
print("=" * 60)

print(attention_weights)
print(f'attention weight shape: {attention_weights.shape}')

# ============================================================
# Head별 결과
# ============================================================
print("\n" + "=" * 60)
print('Head output')
print("=" * 60)
print(head_output)
print(f'attention weight shape: {head_output.shape}')

# ============================================================
# Head별 결과
# ============================================================
print("\n" + "=" * 60)
print('Head output')
print("=" * 60)
print(concat_output)
print(f'concat_output shape: {concat_output.shape}')



# ============================================================
# 9. Attention Weight 행의 합 확인
# ============================================================

print("\n각 행의 합")

print(
    attention_weights.sum(
        dim=-1,
    )
)


# ============================================================
# 10. 최종 결과
# ============================================================

print("\n" + "=" * 60)
print("final output")
print("=" * 60)
print(output)
print(f'final output shape: {output.shape}')

