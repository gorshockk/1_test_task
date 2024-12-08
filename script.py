import pandas as pd
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
def count_words_by_language(text):
    russian_pattern = r'\b[а-яА-ЯёЁ]+\b'  # Регулярное выражение для русских слов
    english_pattern = r'\b[a-zA-Z]+\b'    # Регулярное выражение для английских слов

    russian_count = len(re.findall(russian_pattern, text))
    english_count = len(re.findall(english_pattern, text))

    return russian_count, english_count

def get_sbert_vector(text):
    return model.encode([text])

def sbert_find_most_similar_entity(text,entity_vectors,entity_names,top_n=3):
    text_vector = get_sbert_vector(text)
    similarities = cosine_similarity(text_vector, entity_vectors)[0]
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    res = [(entity_names[i],similarities[i]) for i in top_indices]
    return res



df=pd.read_csv("test_items.csv")
df.loc[-1] = df.columns  # Переносим текущие заголовки в данные
df.index = df.index + 1  # Сдвигаем индексы
df = df.sort_index()  # Переставляем строки
df.columns = ["id","name","description"]  # Задаем новые заголовки


df['russian_count'], df['english_count'] = zip(*df['name'].apply(count_words_by_language))
eng_df=df[df["russian_count"]==0]
ru_df=df[df["english_count"]==0]

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')



sbert_entity_vectors = []
sbert_entity_names = []

for i in range(len(ru_df)):
    vector = get_sbert_vector(ru_df.iloc[i,2])
    sbert_entity_vectors.append(vector)
    sbert_entity_names.append(ru_df.iloc[i,1])

sbert_entity_vectors = np.array(sbert_entity_vectors).squeeze()

text=input("Enter the text:")

res=sbert_find_most_similar_entity(text,sbert_entity_vectors,sbert_entity_names)

for i in res:
  print(f"{i[0]} : {i[1]}")