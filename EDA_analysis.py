# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


data = pd.read_csv("christmas_movies.csv")
df = pd.DataFrame(data)

#Data cleaning 

df["votes"] = df["votes"].str.replace(",","").astype(float)
mean_votes = df["votes"].median()
df["votes"].fillna(mean_votes,inplace=True)

df["gross"] = df["gross"].str.replace("$","")
df["gross"] = df["gross"].str.replace(".","")
df["gross"] = df["gross"].str.replace("M","0000").astype(float)
mean_gross = df["gross"].median()
df["gross"].fillna(mean_gross,inplace=True)

mean_runtime = df["runtime"].median()
df["runtime"].fillna(mean_runtime,inplace=True)

mean_imdb_rating = df["imdb_rating"].median()
df["imdb_rating"].fillna(mean_imdb_rating,inplace=True)

mean_meta_score = df["meta_score"].median()
df["meta_score"].fillna(mean_meta_score,inplace=True)

df.dropna(axis=0,inplace=True)

df["release_year"] = df["release_year"].astype(int)

for i in range(len(df)):   #not rated ve unrated sonuçları aynı olduğundan birleştirdik
    if df["rating"].iloc[i] == "Unrated":
        df["rating"].iloc[i] = "Not Rated"

#print(df.rating.unique())



#Analysis
#top10 movies by imbd ratings
top_movies  =  df.sort_values("imdb_rating" ,  ascending  =  False)
print(top_movies[['title' , 'imdb_rating' , 'release_year']].head(10))

#Top grossing movies
top_gross =  df.sort_values("gross" ,  ascending   =  False).reset_index()
print(top_gross[['title' , "gross" , "imdb_rating"]].head(10))

#movies by years
movies_by_year  =  df.groupby('release_year')['title'].count().to_frame(name = "Number_of_Christmas_Movies").reset_index()
movies_by_year.sort_values('Number_of_Christmas_Movies' , ascending  =  False).head(10)
print(movies_by_year)
#grafiği
sns.countplot(x = df["release_year"])
plt.xticks(rotation = 75)
plt.show()

#imdb geliri nasıl etkiler:
plt.figure(figsize = (10,8))
sns.regplot(data  =  df ,  y = "imdb_rating" , x =  "gross")
plt.title("Rating VS Earning" , fontsize = 15)
plt.ylabel("imdb_rating" , fontsize = 12)
plt.xlabel("Earnings" ,  fontsize =  12)
plt.show()

#yıllara göre sürenin değişimi:
plt.figure(figsize = (10,8))
sns.lineplot(data  = df , x = "release_year" ,y = "runtime")
plt.title("Runtime over Years" , fontsize = 15)
plt.show()

#most common ratings:
plt.figure(figsize = (15,8))
sns.countplot(data = df , x = 'rating')
plt.show()

#süre imdbyi nasıl etkler:
plt.figure(figsize = (10,8))
sns.regplot(data  =  df ,  y = "runtime" , x =  "imdb_rating")
plt.title("Runtime VS IMDB" , fontsize = 15)
plt.ylabel("runtime" , fontsize = 12)
plt.xlabel("imdb_rating" ,  fontsize =  12)
plt.show()

#Yıllara göre oylama sayıları:
plt.figure(figsize = (10,8))
sns.lineplot(data  = df , x = "release_year" ,y = "votes")
plt.title("Votes over Years" , fontsize = 15)
plt.show()