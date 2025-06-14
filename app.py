from flask import Flask, request, render_template, jsonify
import os
from dotenv import load_dotenv 
import requests
from utils import get_movie_details, load_model, generate_recommendations
import pandas as pd

app = Flask(__name__)

load_dotenv()
tmdb_api_key = os.getenv('TMDB_API_KEY')

# Load data and model
df = pd.read_csv('dataset/the-movies-dataset/ratings_small.csv')
links_df = pd.read_csv('dataset/the-movies-dataset/links_small.csv')
svd_model = load_model('models/svd_model.pkl')

def get_movie_details_from_tmdb(movie_id):
    """
    Lấy thông tin chi tiết của phim từ TMDB API bằng movie ID
    """
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}"
        params = {
            'api_key': tmdb_api_key,
            'language': 'vi-VN'  # Hoặc 'en-US' tùy ý
        }
        response = requests.get(url, params=params)
        
        if response.status_code == 200:
            movie_data = response.json()
            return {
                'id': movie_data.get('id'),
                'title': movie_data.get('title'),
                'overview': movie_data.get('overview'),
                'release_date': movie_data.get('release_date'),
                'poster_path': f"https://image.tmdb.org/t/p/w500{movie_data.get('poster_path')}" if movie_data.get('poster_path') else None,
                'vote_average': movie_data.get('vote_average'),
                'genres': [genre['name'] for genre in movie_data.get('genres', [])]
            }
        else:
            return None
    except Exception as e:
        print(f"Error fetching movie details for ID {movie_id}: {str(e)}")
        return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    try:
        # Lấy user ID từ form
        user_id = request.form.get('user_id')
        
        if not user_id:
            return jsonify({'error': 'Vui lòng nhập User ID'}), 400
        
        try:
            user_id = int(user_id)
        except ValueError:
            return jsonify({'error': 'User ID phải là một số nguyên'}), 400
        
        # Kiểm tra user_id có tồn tại trong dataset không
        if user_id not in df['userId'].values:
            return jsonify({'error': f'User ID {user_id} không tồn tại trong hệ thống'}), 404
        
        # Tạo recommendations
        recommendations = generate_recommendations(user_id, svd_model, df)
        
        # Lấy thông tin chi tiết của từng phim từ TMDB
        movie_recommendations = []
        
        for pred in recommendations:
            movie_id = pred.iid
            estimated_rating = pred.est
            movie_details = get_movie_details_from_tmdb(movie_id)
            
            if movie_details:
                movie_details['estimated_rating'] = round(estimated_rating, 2)
                movie_recommendations.append(movie_details)
            else:
                # Nếu không tìm thấy trên TMDB, vẫn hiển thị thông tin cơ bản
                movie_recommendations.append({
                    'id': movie_id,
                    'title': f'Movie ID: {movie_id}',
                    'overview': 'Không có thông tin chi tiết',
                    'estimated_rating': round(estimated_rating, 2),
                    'poster_path': None,
                    'vote_average': None,
                    'genres': []
                })
        
        return jsonify({
            'user_id': user_id,
            'recommendations': movie_recommendations,
            'total_recommendations': len(movie_recommendations)
        })
        
    except Exception as e:
        return jsonify({'error': f'Đã xảy ra lỗi: {str(e)}'}), 500

@app.route('/search', methods=['POST'])
def search():
    """
    Route cũ để tìm kiếm phim (giữ lại để tương thích)
    """
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