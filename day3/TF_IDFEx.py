from sklearn.feature_extraction.text import TfidfVectorizer

corpus = [
    'I love deep learning i love apple',
    'I love NLP',
    'I enjoy flying '
]

vectorizer = TfidfVectorizer()

vectorizer.fit(corpus) #fit : 이것을 바탕으로 학습, 적응시키겠다
print(vectorizer.vocabulary_)

x = vectorizer.transform(corpus)
print(x)#이렇게 보면 불편함
print()
print(x.toarray())