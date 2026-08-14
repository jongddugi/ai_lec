from transformers import AutoTokenizer

MODEL_NAME = 'klue/bert-base'

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)#잘 맞는거 알아서 찾아야함 목적에 맞게

text = '나는 오늘 학교에서 인공지능을 공부한다.'
print('\n원문 : {text}')

'''
tokens = tokenizer.tokenize(text)
print('\nToken:')
print(tokens)

token_ids = tokenizer.convert_tokens_to_ids(tokens)
print('\nToken ID:')
print(token_ids)
'''

inputs = tokenizer(
    text,
    return_tensors = 'pt'
)

print('\nTokenizer 결과 : ')
print(inputs)
