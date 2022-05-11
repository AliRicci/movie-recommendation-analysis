import pandas as pd
import logging
import imdb
from imdb import IMDbError
from imdb import IMDbDataAccessError

def get_infoset_data(imdb_id, feature):
    
    # Prevent critical runtime errors by disabling logger
    logger = logging.getLogger('imdbpy');
    logger.disabled = True
    
    # Create new dataframe for movie feature
    df = pd.DataFrame()
    
    try:
        movie_info = cg.get_movie(imdb_id, info=[feature])
        feature_values = movie_info.get(feature)
        df = df.append(pd.DataFrame(feature_values), ignore_index=True)
        df.insert(0, 'imdb_id', imdb_id)
    
    # Skip movie if imdbId no longer exists in database and/or HTTP 404 page not found error is thrown
    except IMDbDataAccessError:
        pass
    except IMDbError:
        pass
    
    return df


def get_cast_feature(imdb_id):
    
    #Declare new dataframe to hold cast
    df = pd.DataFrame()
    
    #Get first 3 cast members
    cast_list = movie['cast'][0:3]
    
    #Remove spaces to format names as FirstnameLastname
    for c in cast_list:
        cast_member = str(c['name']).replace(' ', '')
        df = df.append(pd.Series(cast_member), ignore_index=True)
        
    df.insert(0, 'imdb_id', imdb_id)   
    
    return df


# Load data
movie_links = pd.read_csv('..\\Data\\links.csv', engine='python')

# Convert to string to match lookup value
movie_links['movieId'] = movie_links['movieId'].astype(str)
movie_links['imdbId'] = movie_links['imdbId'].astype(str)

print(movie_links.shape)
movie_links.head()

# Create an instance of the Cinemagoer class
cg = imdb.Cinemagoer()

# Infosets to extract
features = ['keywords', 'plots', 'reviews']

for f in features:
    
    #Declare new dataframe to hold feature
    df_feature = pd.DataFrame()
    
    #For each movie, extract features and append to dataframe
    for m in movie_links.values:
        imdb_id = m[1]
        df = pd.DataFrame()
        df = get_infoset_data(imdb_id, f)
        df_feature = df_feature.append(df)
       
    #Rename columns and output to csv
    if f == 'keywords':
        df_feature.columns = ['imdb_id', 'keywords']
        df_feature.to_csv('..\\data\\cg_keywords.csv')
    print('keyword extraction completed')
        
    elif f == 'plots':
        df_feature.columns = ['imdb_id', 'plot']
        df_feature.to_csv('..\\data\\cg_plots.csv')
    print('plots extraction completed')
    
    else:
        df_feature.columns = ['imdb_id', 'content', 'helpful', 'title', 'author', 'date', 'rating', 'not_helpful']
        df_feature.to_csv('..\\data\\cg_reviews.csv')
        print('reviews extraction completed')


# Create new dataframe for cast feature
cg_cast = pd.DataFrame()

for m in movie_links.values:
    
    # Prevent critical runtime errors by disabling logger
    logger = logging.getLogger('imdbpy');
    logger.disabled = True
    
    imdb_id = m[1]
    
    try:
        movie = cg.get_movie(imdb_id)
        cast = get_cast_feature(imdb_id)
        
    # Skip movie if imdbId no longer exists in database and/or HTTP 404 page not found error is thrown
    except IMDbDataAccessError:
        pass
    except IMDbError:
        pass    
        
    cg_cast = cg_cast.append(cast)


#Rename columns and output to csv
cg_cast.columns = ['imdb_id', 'cast']
cg_cast['cast'] = cg_cast.groupby(['imdb_id'])['cast'].transform(lambda x : ','.join(x))
cg_cast = cg_cast.drop_duplicates() 
cg_cast.to_csv('..\\data\\cg_cast.csv')
print('cast extraction completed')