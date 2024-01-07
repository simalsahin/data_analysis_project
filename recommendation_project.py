import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

pd.set_option('display.max_columns', None)

data = pd.read_csv("christmas_movies.csv")
df = pd.DataFrame(data)
df.drop(["img_src"],axis= 1,inplace=True)

df["gross"] = df["gross"].str.replace("$","")
df["gross"] = df["gross"].str.replace(".","")
df["gross"] = df["gross"].str.replace("M","0000").astype(float)


df['clean_describe'] = df['description'].str.lower()
df['clean_describe'] = df['clean_describe'].apply(lambda x: re.sub('[^a-zA-Z]', ' ', x))
df['clean_describe'] = df['clean_describe'].apply(lambda x: re.sub('\s+', ' ', x))

df['clean_describe'] = df['clean_describe'].apply(lambda x: nltk.word_tokenize(x))
#print(df['clean_describe'])

stop_words = nltk.corpus.stopwords.words('english')
description = []
for sentence in df['clean_describe']:
    temp = []
    for word in sentence:
        if word not in stop_words and len(word) >= 3:
            temp.append(word)
    description.append(temp)
    
df['clean_describe'] = description

df['genre'] = df['genre'].str.split(",")
df["stars"] = df['stars'].str.split(",")
df["director"] = df['director'].str.split(",")

df.drop(["rating","runtime","imdb_rating","meta_score","release_year","votes","gross","type"],axis=1,inplace=True)
df.dropna(inplace=True)


def clean(sentence):
    temp = []
    for word in sentence:
        temp.append(word.lower().replace(' ', ''))
    return temp
df['genre'] = [clean(x) for x in df['genre']]
df['stars'] = [clean(x) for x in df['stars']]
df['director'] = [clean(x) for x in df['director']]

df.reset_index(inplace = True,drop = True)

columns = ['clean_describe', 'genre', 'stars', 'director']
l = []
for i in range(len(df)):
    words = ''
    for col in columns:
        words += ' '.join(df[col][i]) + ' '
    l.append(words)

df['clean_input'] = l
df = df[['title', 'clean_input']]

tfidf = TfidfVectorizer()
features = tfidf.fit_transform(df['clean_input'])

cosine_sim = cosine_similarity(features, features)

index = pd.Series(df['title'])

def recommend_movies(title):
    movies = []
    idx = index[index == title].index[0]
    #print(idx)
    score = pd.Series(cosine_sim[idx]).sort_values(ascending=False)
    top5 = list(score.iloc[1:6].index)
    #print(top5)
    
    for i in top5:       
        recommended_title = index.iloc[i]
        movies.append(recommended_title)
    return movies

movie = input("enter the movie you like: ")
recommended_movies = recommend_movies(movie)
print(f"if you like {movie};")
print("Recommended movies are:")
for i in recommended_movies:
    print(i)
