import pandas as pd
import traceback
from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# Load movie list and similarity data
movies_list = pd.read_pickle("movies.pkl")
similarity = pd.read_pickle("similarity.pkl")

# Create a function to get movie recommendations
def get_recommendations(movie_name):
    # Get the index of the movie in the movies_list
    movie_index = movies_list[movies_list['title'] == movie_name].index[0]

    # Get the similarity scores of the movie with all other movies
    similarity_scores = similarity[movie_index]

    # Convert the similarity_scores to a pandas Series
    similarity_scores = pd.Series(similarity_scores)

    # Sort the similarity scores in descending order
    sorted_similarity_scores = similarity_scores.sort_values(ascending=False)

    # Get the top 10 most similar movies
    top_10_movies = movies_list.iloc[sorted_similarity_scores.index[:10]]

    # Return the top 10 most similar movies with ratings and description
    return top_10_movies

# Create a Flask app
app = Flask(__name__, static_folder='')

# Add a route to serve the index.html file
@app.route('/')
def index():
    return app.send_static_file('index.html')

# Add a route to the app to get movie recommendations
@app.route('/recommendations')
def recommendations():
    try:
        # Get the movie name from the query parameters
        movie_name = request.args.get('movie_name')

        # Get the top 10 most similar movies with ratings and description
        top_10_movies = get_recommendations(movie_name)

        # Convert the DataFrame to a dictionary
        recommendations_dict = top_10_movies.to_dict(orient='records')

        # Return the recommendations as a JSON response
        return jsonify(recommendations_dict)

    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': 'An error occurred while processing the request'}), 500



if __name__ == '__main__':
    app.run()