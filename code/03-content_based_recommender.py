import re
import random
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer

# Load ratings data
ratings = pd.read_csv('..\\Data\\ratings.csv', ',', engine='python')

# Filter for top 200 movies based on the number of ratings (reviews)
most_reviewed = ratings['movieId'].value_counts()[:200].index.tolist()
top_reviewed_movies = ratings.loc[ratings['movieId'].isin(most_reviewed)]
top_200_movies = top_reviewed_movies['movieId'].unique().tolist()

# Load movies data and filter by top 200 movies
movies = pd.read_csv('..\\Data\\movies.csv', ',', engine='python')
movies = movies[movies['movieId'].isin(top_200_movies)]

# Merge movies with links
links = pd.read_csv('..\\Data\\links.csv', ',', engine='python')
movies = movies.merge(links, on='movieId')

# Load content based data
tags = pd.read_csv('..\\Data\\tags.csv', ',', engine='python')
keywords = pd.read_csv('..\\Data\\cg_keywords.csv', ',', engine='python')
plots = pd.read_csv('..\\Data\\cg_plots.csv', ',', engine='python')
cast = pd.read_csv('..\\Data\\cg_cast.csv', ',', engine='python')

# Prep columns for merging and concatenation
movies['imdbId'] = 'imdb' + movies['imdbId'].astype(str)
keywords['imdbId'] = 'imdb' + keywords['imdb_id'].astype(str)
plots['imdbId'] = 'imdb' + plots['imdb_id'].astype(str)
cast['imdbId'] = 'imdb' + cast['imdb_id'].astype(str)

# Load tools for text preprocessing
exclude = set(string.punctuation)
wnl = WordNetLemmatizer()
nltk.download('wordnet')

# Title - remove parenthesis containing year from movie title
movies['title'] = movies['title'].str.replace(r"\(.*\)", '', regex=True)
movies['title'] = movies['title'].str.strip()

# Cast - remove punctuation
cast['cast'] = cast['cast'].apply(lambda x: word_tokenize(x))
cast['cast'] = cast['cast'].apply(lambda x: [word for word in x if word not in exclude])
cast['cast'] = [','.join(i) for i in cast['cast']]

# Genres - convert to comma delimited list of genres
movies['genres'] = movies['genres'].str.replace('-','')
movies['genres'] = movies['genres'].str.replace('(no genres listed)','',regex=True)
movies['genres'] = movies['genres'].replace('\|', ',', regex=True)

# Keywords
# Remove dashes that separate tags (tag format: dog-cat-funny) and remove select punctuation
keywords['keywords'] = keywords['keywords'].str.replace('-',' ', regex=True)
keywords['keywords'] = keywords['keywords'].str.replace('\'','', regex=True)
keywords['keywords'] = keywords['keywords'].str.replace('.','', regex=True)

# Keywords

def get_keywords(imdb_id):
    df = pd.DataFrame()
    keywords_list = keywords[keywords['imdbId'] == imdb_id] 
    k = ','.join(keywords_list['keywords'])
    df.insert(0, 'imdbId', pd.Series(imdb_id))
    df.insert(1, 'keywords', pd.Series(k))
    return df

# Create a new dataframe to hold keywords for each movie
df_keywords = pd.DataFrame()

# Get keywords for each movie and append to dataframe
for index, row in movies.iterrows():
    imdb_id = row['imdbId']
    k = get_keywords(imdb_id)
    df_keywords = df_keywords.append(k)

# Merge keywords
movies = movies.merge(df_keywords, on='imdbId')


# Tags

def get_tags(movie_id):
    df = pd.DataFrame()
    tags_list = tags[tags['movieId'] == movie_id]
    t = ','.join(tags_list['tag'])
    df.insert(0, 'movieId', pd.Series(movie_id))
    df.insert(1, 'tags', pd.Series(t))
    return df

# Create a new dataframe to hold tags for each movie
df_tags = pd.DataFrame()

# Get tags for each movie and append to dataframe
for m in movies.values:
    movie_id = m[0]
    t = get_tags(movie_id)
    df_tags = df_tags.append(t)
    
# Merge tags 
movies = movies.merge(df_tags, on='movieId')

# Text preprocessing on tags
movies['tags'] = movies['tags'].str.replace('-','')
movies['tags'] = movies['tags'].str.replace(',',' ')


# Plots
    
def get_plots(imdb_id):
    df = pd.DataFrame()
    plots_list = plots[plots['imdbId'] == imdb_id]
    
    #Get first 10 plots
    #plots_list = plots_list[:10]
    
    p = ' '.join(plots_list['plot'])
    df.insert(0, 'imdbId', pd.Series(imdb_id))
    df.insert(1, 'plots', pd.Series(p))
    return df

# Create a new dataframe to hold plots for each movie
df_plots = pd.DataFrame()

# Get plots for each movie and append to dataframe
for index, row in movies.iterrows():
    imdb_id = row['imdbId']
    p = get_plots(imdb_id)
    df_plots = df_plots.append(p)
    
#Remove emails associated with some plots provided by IMDb viewers
regex_email = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
df_plots['plots'] = df_plots['plots'].apply(lambda x: re.sub(regex_email,'', x))
    
# Merge plots
movies = movies.merge(df_plots, on='imdbId')

# Preprocessing on features
features = ['tags', 'keywords', 'plots']

for f in features:
    movies[f] = movies[f].apply(lambda x: x.lower())
    movies[f] = movies[f].apply(lambda x: word_tokenize(x))
    movies[f] = movies[f].apply(lambda x:[wnl.lemmatize(word) for word in x])
    movies[f] = [','.join(i) for i in movies[f]]

# Concatenate features into one column
movies['combined_features'] = movies['genres'] + movies['tags'] + movies['keywords'] + movies['plots'] + cast['cast']

# Build TF-IDF vectors
tfidf = TfidfVectorizer(stop_words='english', min_df=0.05)
tfidf_matrix = tfidf.fit_transform(movies['combined_features'])
tfidf_matrix.shape
#print(tfidf_matrix)
print('Number of features:', len(tfidf.get_feature_names_out()))

# Compute cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
cosine_sim.shape

#Construct a reverse map of indices and movie titles
indices = pd.Series(movies.index, index=movies['title']).drop_duplicates()

# Recommender unction that takes in movie title as input and outputs the most similar movies
def get_recommendations(title, cosine_sim=cosine_sim):
    
    # Get the index of the movie that matches the title
    index = indices[title]

    # Get the similarity scores of all movies compared to input movie
    sim_scores = list(enumerate(cosine_sim[index]))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar movies excluding the input movie at position 0
    sim_scores = sim_scores[1:11]

    # Get the movie indices
    movie_indices = [i[0] for i in sim_scores]

    # Return the most similar movies
    return movies['title'].iloc[movie_indices]


# Recommendation for the top 15 movies from the top 3 users in the collaborative approach

input_movies = ['Back to the Future', 'Dr. Strangelove or: How I Learned to Stop Worrying and Love the Bomb',
                'L.A. Confidential', 'Aliens', 'Kill Bill: Vol. 1', 'Matrix, The', 'Inception', 
                'Princess Bride, The', 'Office Space', 'Dead Poets Society', 'Alien', 'Gattaca',
                'Life Is Beautiful', 'Godfather, The', 'When Harry Met Sally...']

# Output each input movie and the recommended movie output
for m in input_movies:
    print('Recommendations for movie input', m)
    result = get_recommendations(m)
    print(result.iloc[0])
    print()