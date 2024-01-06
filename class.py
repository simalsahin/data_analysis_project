import pandas as pd
import numpy as np
data = pd.read_csv("christmas_movies.csv")
df = pd.DataFrame(data)


df["gross"] = df["gross"].str.replace("$","")
df["gross"] = df["gross"].str.replace(".","")
df["gross"] = df["gross"].str.replace("M","0000").astype(float)
df["gross"].fillna(df["gross"].median(),inplace=True)
df["votes"] = df["votes"].str.replace(",","").astype(float)
mean_votes = df["votes"].median()
df["votes"].fillna(mean_votes,inplace=True)
mean_runtime = df["runtime"].median()
df["runtime"].fillna(mean_runtime,inplace=True)
mean_imdb_rating = df["imdb_rating"].median()
df["imdb_rating"].fillna(mean_imdb_rating,inplace=True)
mean_meta_score = df["meta_score"].median()
df["meta_score"].fillna(mean_meta_score,inplace=True)

# print(df["genre"].value_counts())
        
# Assuming df is your DataFrame
genres_dummies = df['genre'].str.join('|').str.get_dummies()
from sklearn.preprocessing import StandardScaler

# Assuming df_numeric contains your numerical features
scaler = StandardScaler()
df_numeric_scaled = scaler.fit_transform(df[["runtime","votes","meta_score","imdb_rating","gross"]])
# Assuming df_numeric_scaled contains your numerical features
df_numeric_scaled = pd.DataFrame(df_numeric_scaled)
df_combined = pd.concat([df_numeric_scaled, genres_dummies], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

# Assuming 'target' is your target variable

X_train, X_test, y_train, y_test = train_test_split(df_combined, genres_dummies, test_size=0.15, random_state=42)

# Create KNN classifier
knn = KNeighborsClassifier(n_neighbors=5)
X_train.columns = X_train.columns.astype(str)
# Train the classifier
knn.fit(X_train, y_train)

# Make predictions
X_test = X_test.values
X_test = np.ascontiguousarray(X_test)

predictions = knn.predict(X_test)

# Evaluate the model
from sklearn.metrics import accuracy_score,mean_squared_error
accuracy = accuracy_score(y_test, predictions)
mean_squared = mean_squared_error(y_test, predictions)
from sklearn.metrics import hamming_loss, jaccard_score, precision_recall_fscore_support

# Evaluate the model
hamming_loss_value = hamming_loss(y_test, predictions)
jaccard_score_value = jaccard_score(y_test, predictions, average='micro')  # You can choose 'macro' or 'weighted' as well
precision, recall, _, _ = precision_recall_fscore_support(y_test, predictions, average='micro')

print(f"Hamming Loss: {hamming_loss_value:.4f}")
print(f"Jaccard Score: {jaccard_score_value:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")

print(f"Accuracy: {accuracy} \nMSE: {mean_squared}")
#print(genres_dummies)
