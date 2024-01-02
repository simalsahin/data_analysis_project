import pandas as pd

data = pd.read_csv("christmas_movies.csv")
df = pd.DataFrame(data)

result1 = df.isnull().sum()

# rating kategorisinde çok fazla nan değer var ama bu kategorinin tipi string, zaten kullanmayacağımızdan 1-2-3 şeklinde değiştirip değer atadım:
df["rating"].replace('R',1,inplace=True)
df["rating"].replace('PG-13',2,inplace=True)
df["rating"].replace('Not Rated',0,inplace=True)
df["rating"].replace('G',3,inplace=True)
df["rating"].replace('TV-G',4,inplace=True)
df["rating"].replace('TV-PG',5,inplace=True)
df["rating"].replace('TV-MA',7,inplace=True)
df["rating"].replace('Passed',8,inplace=True)
df["rating"].replace('Approved',9,inplace=True)
df["rating"].replace('6',6,inplace=True)
df["rating"].replace('AL',10,inplace=True)
df["rating"].replace('TV-14',11,inplace=True)
df["rating"].replace('Unrated',0,inplace=True)
df["rating"].replace('7+',12,inplace=True)
df["rating"].replace('TV-Y',13,inplace=True)
df["rating"].replace('PG',14,inplace=True)


#votes sütunundaki virgülleri kaldırıp float hale getirdim:
df["votes"] = df["votes"].str.replace(",","").astype(float)
mean_votes = df["votes"].mean()
df["votes"].fillna(mean_votes,inplace=True)

#gross sütunundaki $,M ve noktayı temizleyip float hale getirdim:
df["gross"] = df["gross"].str.replace("$","")
df["gross"] = df["gross"].str.replace(".","")
df["gross"] = df["gross"].str.replace("M","0000").astype(float)
mean_gross = df["gross"].mean()
df["gross"].fillna(mean_gross,inplace=True)

#print(df.dtypes) data types kontrolü

#rating runtime imdb_rating meta_score release_year sütunlarındaki nan değerleri ortalamalarla doldurdum:
mean_rating = df["rating"].mean()
df["rating"].fillna(mean_rating,inplace=True)

mean_runtime = df["runtime"].mean()
df["runtime"].fillna(mean_runtime,inplace=True)

mean_imdb_rating = df["imdb_rating"].mean()
df["imdb_rating"].fillna(mean_imdb_rating,inplace=True)

mean_meta_score = df["meta_score"].mean()
df["meta_score"].fillna(mean_meta_score,inplace=True)

#print(df.isna().sum()) işlemlerden sonra datasetteki nan değer kontrolü

#kalan nan değerleri str olduklarından atıyorum, sayıları da az:
df.dropna(inplace=True)

#print(df) #sonda 771 değerle bitirdim sadece 18 değer kaybettik.



