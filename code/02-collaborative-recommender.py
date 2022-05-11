import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from community import community_louvain
import time
pd.options.mode.chained_assignment = None
%matplotlib inline


names = ['userId', 'movieId', 'rating', 'timestamp']
data = pd.read_csv('..\\Data\\ratings.csv', ',', names=names, engine='python', skiprows = 1)
data['userId'] = 'u' + data['userId'].astype(str)
data['movieId'] = 'm' + data['movieId'].astype(str)
data['movie_name'] = pd.Series(dtype='str')

movies_names = ['movieId', 'movie_name', 'genres']
movies = pd.read_csv('..\\Data\\movies.csv', ',', names=movies_names, engine='python', skiprows = 1)
movies['movieId'] = 'm' + movies['movieId'].astype(str)
movies.set_index('movieId', inplace=True)

# Pull in movie name
data['movie_name'] = data.apply(lambda row: movies.loc[row['movieId']][0], axis=1)

print(data.shape)
data.describe(include = 'all')
data.head()
print(movies.shape)

# Plot the top 200 movies with the most ratings
most_reviewed = data['movieId'].value_counts()[:200].index.tolist()
top_reviewed_movies = data.loc[data['movieId'].isin(most_reviewed)]
data['movieId'].value_counts()[:200].plot(kind='bar', xticks=[], ylabel='Number Of Ratings')

movie_rating_count = data.groupby('movieId')['userId'].count() 
print (movie_rating_count[movie_rating_count < 10].count())

# Apply formatting
mcr_desc = pd.DataFrame({'Review Count': movie_rating_count.describe()})
mcr_desc = mcr_desc.reset_index()
mcr_desc.to_csv('all_movies_desc.csv')

r_bins = [1, 10, 20, 30, 40, 50, 100, 200, 300]
cnt_plt = movie_rating_count.plot(kind='hist', bins=r_bins, title='Number Of Ratings Per Movie')
cnt_plt.set_xlabel("Review Count")

print (top_reviewed_movies.shape)
top_reviewed_movies.head()

no_movies = len(top_reviewed_movies.movieId.unique())
no_users = len(top_reviewed_movies.userId.unique())
print('Number of movies reviewed in dataset:', no_movies)
print('Number of reviewers in dataset:', no_users)
top_reviewed_movies.shape
top_reviewed_movies.groupby('movieId')['userId'].count().describe()

# Create the ratings matrix
# One row per reviewer and one column per movie with the rating for that movie as the value
# Rating is -1 when the user has not reviewed this movie
ratings = top_reviewed_movies.pivot(index = 'userId', columns='movieId', values='rating').fillna(-1).astype('float')
print(ratings.shape)

# Dataframe for movie commonly reviewed
movies_integer = np.zeros((no_movies, no_movies))
def both_reviewed(x):
    # we are only looking for 'liked' movies
    if x[0] >= 4 and x[1] >= 4:
        return 1
    return -1

# Count how many times each pair of movies has been positively reviewed by the same user
for i in range(no_movies):
    cur_t = time.strftime("%H:%M:%S", time.localtime())
    print (f"processing row i={i} at {cur_t}")
    for j in range(no_movies):
        if i != j:
            df_ij = ratings.iloc[:,[i,j]] #create a temporary df with only i and j movies as columns
            df_ij['both_reviewed'] = df_ij.apply(both_reviewed, axis=1)
            pairings_ij = len(df_ij[df_ij['both_reviewed'] > 0]) #both reviewd positively
            movies_integer[i,j] = pairings_ij
            movies_integer[j,i] = pairings_ij

# Count how many times each individual movie has been reviewed with another movie
times_reviewed = movies_integer.sum(axis = 1)
pd.Series(times_reviewed).hist(bins = 20)

# Construct final weighted matrix of movie reviews
movies_weighted = np.zeros((no_movies, no_movies))
for i in range(no_movies):
    cur_t = time.strftime("%H:%M:%S", time.localtime())
    print (f"processing row i={i} at {cur_t}")
    for j in range(no_movies):
        if i == j:
            continue
        if (times_reviewed[i] + times_reviewed[j]) !=0: 
            mi_ij = movies_integer[i,j]
            tr_i = times_reviewed[i]
            tr_j = times_reviewed[j]
            weighted_rank = (mi_ij) / (tr_i + tr_j)
            movies_weighted[i,j] = weighted_rank

# Get list of movie names instead of ids
nodes_ids = np.array(ratings.columns).astype('str')
movie_lookup_dict = pd.Series(top_reviewed_movies.movie_name.values, index=top_reviewed_movies.movieId).to_dict()
nodes_labels = [movie_lookup_dict[id] for id in nodes_ids]

# Create Graph object using the weighted movie matrix as adjacency matrix
G = nx.from_numpy_matrix(movies_weighted)
pos = nx.random_layout(G)
labels = {}
for idx, node in enumerate(G.nodes()):
    labels[node] = nodes_labels[idx]

# Export graph to Gephi with labels
H = nx.relabel_nodes(G,labels) 
nx.write_gexf(H, "movies.gexf")

print(nx.info(G))
diameter = nx.diameter(G)
nnodes = G.number_of_nodes()
nedges = G.number_of_edges()
deg = sum(d for n, d in G.degree()) / float(nnodes)
comm_g_avg_cluster_coef = nx.average_clustering(G);
avg_shortest_path_length = nx.average_shortest_path_length(G)
density = nx.density(G)
print ('stats for graph \n')
print ('density: {}'.format(density))
print ('diameter: {}'.format(diameter))
print ('average shortest path length: {}'.format(avg_shortest_path_length))
print ('number nodes in graph {}'.format(nnodes))
print ('number edges in graph {}'.format(nedges))
print ('avg degree of graph {}'.format(deg))
print ('avg cluster_coef of graph {}'.format(comm_g_avg_cluster_coef))
print ('\n')

# Community detection using Louvain algorithm
partition = community_louvain.best_partition(G, resolution = 1.15)
values = list(partition.values())
len(np.unique(values))

print('Number of communities:', len(np.unique(values)))
movies_communities = pd.DataFrame(nodes_labels, columns = ['movie_name'])
movies_communities['community_id'] = values
for c in np.unique(values):
	comm_movies = movies_communities[movies_communities['community_id']==c]
        comm_movies.to_csv(f"community{c}_movies.csv")

cplot = pd.Series(partition).value_counts().sort_index().plot(kind='bar', xlabel='Community Number', ylabel='Movie Count', rot=0)

# Divide each element in movies_weighted dataframe with the maximum of each row.
# This will normalize values in the row
movies_weighted_pd = pd.DataFrame(movies_weighted, columns = nodes_labels)
movies_weighted_pd.set_index(movies_weighted_pd.columns, 'movie_name', inplace=True)
movies_prob = movies_weighted_pd.divide(movies_weighted_pd.max(axis = 1), axis = 0)

# Top 5 users
top_movies = []
top_recommendations = []
users_ids = []
no_of_suggestions = 3
already_done = []

top_reviewers = list(top_reviewed_movies['userId'].value_counts().index[:5][0:5])
for r in top_reviewers:
    users_ids.append(r)
    r_favs = top_reviewed_movies[top_reviewed_movies['userId'] == r][['movie_name', 'rating']]
    r_favs.reset_index(drop=True)
    r_favs = r_favs[~r_favs.movie_name.isin(already_done)]
    r_favs.reset_index(drop=True)
    r_favs.sort_values(by=['rating'], ascending=[False], inplace=True)
    r_favs.reset_index(drop=True)
    r_favs_names = list(r_favs['movie_name'].iloc[0:no_of_suggestions])
    top_movies.append(r_favs_names)
    print (f"user {r}'s' highest rated movies: {r_favs_names}")
    movies = r_favs_names

    all_movies = movies_prob[movies]
    suggestions = []
    for m in movies:
        already_done.append(m)
        all_movies = all_movies.sort_values(by = [m], ascending=False)
        suggested_movie = list(all_movies.index[:1])[0]
        suggestions.append(suggested_movie)
        print (f"for movie '{m}' top suggestion was '{suggested_movie}'")
    print(f"user {r} might like: {suggestions}")
    top_recommendations.append(suggestions)
    print('')
    
results_df = df = pd.DataFrame({'user_id':users_ids, 'top_rated':top_movies, 'recommendations':top_recommendations})
results_df.to_csv('results.csv')
results_df.head()