from surprise import Reader, Dataset, SVD, NMF, accuracy
from surprise.model_selection import cross_validate, train_test_split
import pickle

def validate_model(df, model_type = 'SVD'):
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    if model_type == 'SVD':
        algo = SVD()
    elif model_type == 'NMF':
        algo = NMF()

    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)
    mae = accuracy.mae(predictions, verbose=True)

    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    print(f"Model: {model_type}, RMSE: {rmse}, MAE: {mae}")
    return algo

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

def load_model(filename):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded from {filename}")
    return model

def generate_recommendations(user_id, algo, df):
    """Generate and print top 10 movie recommendations for a user."""
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    user_predictions = [pred for pred in predictions if pred.uid == user_id]
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    top_10_recommendations = user_predictions[:10]

    # print(f"Top 10 recommendations for User {user_id}:")
    # for pred in top_10_recommendations:
    #     print(f"Movie ID: {pred.iid}, Estimated Rating: {pred.est:.2f}")
    return top_10_recommendations
        

def get_movie_details(tmdb_api_key, title):
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={tmdb_api_key}&query={title}"
    response = requests.get(search_url)
    if response.status_code == 200:
        results = response.json()['results']
        if results:
            movie = results[0]
            return {
                'title': movie['title'],
                'poster': f"https://image.tmdb.org/t/p/w500{movie['poster_path']}" if movie['poster_path'] else None,
                'link': f"https://www.themoviedb.org/movie/{movie['id']}"
            }
    return None