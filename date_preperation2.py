import pandas as pd
from sklearn.model_selection import train_test_split
movies = pd.read_csv('movie.csv') 
filtered_movies = movies[['Series_Title', 'Genre', 'Released_Year', 'IMDB_Rating']]  
filtered_movies = filtered_movies.dropna()  
filtered_movies = filtered_movies[filtered_movies['IMDB_Rating'] >= 7.0] 
filtered_movies['dataset_split'] = 'train'  
train_data, test_data = train_test_split(filtered_movies, test_size=0.2, random_state=42)
filtered_movies.loc[test_data.index, 'dataset_split'] = 'test'
filtered_movies.to_csv('filtered_movies_with_split.csv', index=False)

print("Filtered dataset with train-test split saved as 'filtered_movies_with_split.csv'")
