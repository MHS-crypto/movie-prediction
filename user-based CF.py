import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity


# PART A

# importing users data
users = pd.read_csv('u.user', sep="|", names=["user_id", "age", "gender", "occupation", "zip_code"])
print(users.head(2))

"""
user_id  age gender  occupation zip_code
0        1   24      M  technician    85711
1        2   53      F       other    94043

"""

# importing ratings data
ratings = pd.read_csv('u.data',  sep='\t', names = ['user_id', 'movie_id', 'rating', 'timestamp']
, encoding='latin-1')

print(ratings.head(2))


"""
 user_id  movie_id  rating  timestamp
0      196       242       3  881250949
1      186       302       3  891717742
"""

df = pd.DataFrame(ratings)


print("Rows for rating column:", len(df['rating'])) # Rows for rating column: 100000




# Now we will merge the ratings and users data on user_id

data = pd.merge(users, ratings, on="user_id")


# creating a pivot table to summarize the data making movie_id as column, taking ratings as values 
# user_id as rows
user_item_matrix = data.pivot_table(index="user_id", columns="movie_id", values="rating")
user_item_matrix = user_item_matrix.fillna(0)




def pearson_corelation(selected_user):
    
    
    similarities = []
    for user in user_item_matrix.index:
        
        if user != selected_user:
            
            common_movies = set(user_item_matrix.columns[user_item_matrix.loc[selected_user].notna() & user_item_matrix.loc[user].notna()])
            if len(common_movies) == 0:
                 return 0  # Users have no common rated movies
            user1_ratings = user_item_matrix.loc[selected_user, common_movies]
            user2_ratings = user_item_matrix.loc[user, common_movies]
            correlation = np.corrcoef(user1_ratings, user2_ratings)[0, 1]
            
            if np.isnan(correlation):
                similarity =  0
                similarities.append((user,similarity))
            else:
                similarity = correlation
                similarities.append((user,similarity))
            
    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

    return similarities[:10]


def predict_movie_score(selected_user, target_movie_id, similar_users, user_item_matrix):
    # Calculate the mean rating ¯𝒓𝒂 for the selected user
    active_user_ratings = user_item_matrix.loc[selected_user]
    active_user_mean_rating = active_user_ratings.mean()
    
    weighted_sum = 0
    similarity_sum = 0
    
    for similar_user, similarity_score in similar_users:
        if user_item_matrix.at[similar_user, target_movie_id] != 0:
            # Calculate ¯𝒓𝒃 for the similar user 𝒃
            similar_user_ratings = user_item_matrix.loc[similar_user]
            similar_user_mean_rating = similar_user_ratings.mean()
            
            weighted_sum += similarity_score * (user_item_matrix.at[similar_user, target_movie_id] - similar_user_mean_rating)
            
            similarity_sum += abs(similarity_score)
    
    if similarity_sum == 0:
        return active_user_mean_rating
    
    # Calculate the final prediction
    predicted_score = active_user_mean_rating + weighted_sum / similarity_sum
    return predicted_score


def movie_predictor(user_id):
    similar_users = pearson_corelation(user_id)

    unrated_movies = user_item_matrix.columns[user_item_matrix.loc[user_id] == 0]
    predictions = []

    for movie in unrated_movies:
        score = predict_movie_score(user_id, movie, similar_users, user_item_matrix)
        predictions.append((movie, score))
    
    top_10_movies = sorted(predictions, key=lambda x: x[1], reverse=True)[:10]

    print()
    print("Top 10 users similer to User_id:", user_id)

    for users, relation in similar_users:
        print(f"User {users}, Similarity: {relation:.2f}")
    

    print()
    print("Top 10 recommended movies for User_id", user_id)
    for movie_id, predicted_score in top_10_movies:
        print(f"Movie {movie_id}, Predicted Score: {predicted_score:.2f}")




selected_user = 863
movie_predictor(3)
#movie_predictor(863)
#movie_predictor(616)

"""
Top 10 users similer to User_id: 3
User 863, Similarity: 0.47        
User 616, Similarity: 0.47        
User 489, Similarity: 0.47        
User 784, Similarity: 0.45        
User 317, Similarity: 0.45
User 587, Similarity: 0.45
User 752, Similarity: 0.44
User 724, Similarity: 0.44
User 335, Similarity: 0.44
User 772, Similarity: 0.43

Top 10 recommended movies for User_id 3
Movie 306, Predicted Score: 4.92
Movie 360, Predicted Score: 4.85
Movie 984, Predicted Score: 4.85
Movie 1293, Predicted Score: 4.85
Movie 1612, Predicted Score: 4.85
Movie 313, Predicted Score: 4.38
Movie 316, Predicted Score: 4.14
Movie 902, Predicted Score: 4.13
Movie 315, Predicted Score: 4.10
Movie 1025, Predicted Score: 3.97

"""



def cosine_similarity(selected_user):
    similarities = []
    for user in user_item_matrix.index:
        if user != selected_user:
            user1_ratings = user_item_matrix.loc[selected_user].values.reshape(1, -1)
            user2_ratings = user_item_matrix.loc[user].values.reshape(1, -1)
            similarity = sklearn_cosine_similarity(user1_ratings, user2_ratings)[0][0]
            similarities.append((user, similarity))

    similarities = sorted(similarities, key=lambda x: x[1], reverse=True)
    return similarities[:5]


similarity = cosine_similarity(selected_user)
print()
print(f"Top 5 similar users for User {selected_user} using Cosine similarity:")

for user_id, relation in similarity:
    print(f"User {user_id}, Similarity: {relation:.2f}")

"""
Top 5 similar users for User 3 using Cosine similarity:
User 863, Similarity: 0.49
User 489, Similarity: 0.49
User 616, Similarity: 0.48
User 587, Similarity: 0.47
User 784, Similarity: 0.47


The above Cosine Similarity is useful as it is not dependent on the rating scale. Also it works well
even when users have rated only a few movies in common. It calculates meaningful similarities.


"""
