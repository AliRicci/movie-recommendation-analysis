import re
import string
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from nltk.stem import PorterStemmer
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn import preprocessing
from scipy.stats import pearsonr


# Load movie data
movies = pd.read_csv('..\\Data\\movies.csv', ',', engine='python')
links = pd.read_csv('..\\Data\\links.csv', ',', engine='python')
movies = movies.merge(links, on='movieId')

# Load reviews data
reviews = pd.read_csv('..\\Data\\cg_reviews.csv', ',', engine='python')
reviews['imdbId'] = 'imdb' + reviews['imdb_id'].astype(str)
print('reviews shape', reviews.shape)

# Load ratings data
ratings = pd.read_csv('..\\Data\\ratings.csv', ',', engine='python')
ratings = ratings.merge(links, on='movieId')
ratings['imdbId'] = 'imdb' + ratings['imdbId'].astype(str)
print('ratings shape', ratings.shape)

# Number of reviews per movie
df = pd.DataFrame()
num_reviews_by_movie = reviews['imdbId'].value_counts()
agg_num_reviews = num_reviews_by_movie.value_counts()
df = agg_num_reviews.reset_index()
df.columns = ['num_reviews', 'num_movies']
df = df.sort_values(by=['num_reviews'], ascending=False)

# Plot distribution of movie reviews
plt.figure(figsize=(10,5))
sns.barplot(y=df['num_movies'].values,x=df['num_reviews'].values, order=df['num_reviews'])
plt.title('Distribution of IMDb Movie Reviews')
plt.xlabel('Number of Reviews', fontsize=12)
plt.ylabel('Number of Movies', fontsize=12)
plt.yscale('log')
plt.show()

# Preprocessing on movie reviews

# Strip any leading or trailing spaces
reviews['content'] = reviews['content'].str.strip()

# Remove numbers
reviews['content'] = reviews['content'].replace('\d+', '', regex=True)

# Convert to lowercase
reviews['content'] = reviews['content'].apply(lambda x: x.lower())

# Split each movie review into tokens
reviews['content_tokenized'] = reviews.apply(lambda x: wordpunct_tokenize(x['content']), axis=1)

# Remove punctuation
exclude = set(string.punctuation)
exclude.add('–')
exclude.add('",')
reviews['content_tokenized'] = reviews['content_tokenized'].apply(lambda x: [word for word in x if word not in exclude])

# Remove stopwords
sws = set(stopwords.words('english'))
reviews['content_tokenized'] = reviews['content_tokenized'].apply(lambda x: [word for word in x if word not in sws])

# Perform word stemming
stemmer = PorterStemmer()
reviews['content_tokenized'] = reviews['content_tokenized'].apply(lambda x : [stemmer.stem(word) for word in x])

# Create content_cleaned field as a string object
reviews['content_cleaned'] = reviews['content_tokenized'].apply(lambda x: ' '.join(word for word in x))

# For each movie review, get the polarity scores
sid = SentimentIntensityAnalyzer()
reviews['scores'] = reviews['content_cleaned'].apply(lambda x: sid.polarity_scores(x))

# For each movie review, get the compound score
reviews['compound']  = reviews['scores'].apply(lambda score_dict: score_dict['compound'])

# Label each movie review as neg / pos / neu
reviews['label'] = reviews['compound'].apply(lambda x: 'pos' if x > .5 else ('neg' if x < -1 * .5 else 'neu'))

# Get the average rating for each movie
ratings_averaged = ratings.drop(['userId', 'movieId', 'timestamp', 'tmdbId'], axis=1)
ratings_averaged = ratings.groupby(['imdbId'], as_index=False).mean()

# Get the average compound score for each movie
reviews_averaged = reviews.drop(['imdb_id', 'content', 'helpful', 'title', 'author', 'date', 'rating', 'not_helpful'], axis=1)
reviews_averaged = reviews_averaged.groupby(['imdbId'], as_index=False).mean()

# Label each movie as neg / pos / neu
reviews_averaged['label'] = reviews_averaged['compound'].apply(lambda x: 'pos' if x > .5 else ('neg' if x < -1 * .5 else 'neu'))

# Get number of reviews for each label
reviews_averaged['label'].value_counts()

# Merge ratings and reviews
df_comparison = ratings_averaged.merge(reviews_averaged, on='imdbId')
df_comparison = df_comparison.sort_values('rating', ascending=False)
df_comparison.shape

# Graph to show sentiment scores and ratings for all movies
df_comparison.plot(x = 'imdbId', y = ['rating', 'compound'], kind = 'bar')
ax = plt.gca()
plt.legend(loc='best')
ax.axes.xaxis.set_visible(False)
plt.title('Ratings and Sentiment Scores for All Movies')

corr, _ = pearsonr(df_comparison['rating'], df_comparison['compound'])
print('Pearsons correlation: %.3f' % corr)

# Keep the first movie for each unique rating
# This prevents having multiple instances of movies with the same rating, allowing for a range of movies to be selected
ratings_averaged = ratings_averaged.drop_duplicates(subset='rating', keep='first')

# Get the top 10 rated movies
top_10_rated = ratings_averaged.sort_values(by=['rating'], ascending=False).iloc[:10]

# Get the bottom 10 rate movies
bottom_10_rated = ratings_averaged.sort_values(by=['rating'], ascending=True).iloc[:10]

# Select top 10 and bottom 10 movies
reviews_top_10_rated = df_comparison[df_comparison['imdbId'].isin(top_10_rated['imdbId'])]
reviews_bottom_10_rated = df_comparison[df_comparison['imdbId'].isin(bottom_10_rated['imdbId'])]
reviews_all = pd.concat([reviews_top_10_rated, reviews_bottom_10_rated], ignore_index=True, sort=False)

# Top 10 rated movies by avg sentiment
plt.hist(reviews_top_10_rated['compound'], bins=3, color='skyblue', ec='black')
plt.xlabel('Average Compound Sentiment Score')
plt.title('Sentiment Score Distribution for Top 10 Rated Movies')
plt.show()

# Top 10 rated movies by avg rating
plt.hist(reviews_top_10_rated['rating'], bins=5, color='maroon', ec='black')
plt.xlabel('Average Rating')
plt.title('Rating Distribution for Top 10 Rated Movies')
plt.show()

# Bottom 10 rated movies by avg sentiment
plt.hist(reviews_bottom_10_rated['compound'], bins=3, color='skyblue', ec='black')
plt.xlabel('Average Compound Sentiment Score')
plt.title('Sentiment Score Distribution for Bottom 10 Rated Movies')
plt.show()

# Bottom 10 rated movies by avg rating
plt.hist(reviews_bottom_10_rated['rating'], bins=5, color='maroon', ec='black')
plt.xlabel('Average Rating')
plt.title('Rating Distribution for Bottom 10 Rated Movies')
plt.show()

# Display full text
pd.set_option('display.max_colwidth', None)

# Get movie name
reviews_top_10_rated = reviews_top_10_rated.merge(movies, on='movieId')
reviews_bottom_10_rated = reviews_bottom_10_rated.merge(movies, on='movieId')

high_rated_movie_example = reviews[reviews['imdbId']=='imdb244316']
print(high_rated_movie_example)

low_rated_movie_example = reviews[reviews['imdbId']=='imdb101492']
print(low_rated_movie_example)
