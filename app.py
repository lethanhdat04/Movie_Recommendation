from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv 
import requests

app = Flask(__name__)

load_dotenv()
tmdb_api_key = os.getenv('TMDB_API_KEY')

def get_movie_details(title):
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

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/search', methods=['POST'])
def search():
    try:
        user_input = request.form['user_input']
        movie_details = get_movie_details(user_input)
        if movie_details:
            return jsonify(movie_details)
        else:
            return jsonify({'error': 'Movie not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    
if __name__ == '__main__':
    app.run(debug=True)