import os
from flask import Flask, render_template, request, jsonify, redirect, url_for
from recommender import BookRecommender

app = Flask(__name__)
app.config['SECRET_KEY'] = 'book-recommender-super-secret-key-2026'

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
recommender = BookRecommender(BASE_DIR)

@app.context_processor
def inject_global_vars():
    """Inject global statistics and search helpers to all templates."""
    stats = recommender.get_stats()
    return {
        'app_stats': stats,
        'cf_books_count': stats.get('collaborative_books_count', 679)
    }

@app.route('/')
def index():
    """Home Page displaying Top 50 Popular Books & Hero Section."""
    popular_books = recommender.get_popular_books(limit=50)
    featured_picks = recommender.get_random_famous_books(count=6)
    stats = recommender.get_stats()
    return render_template(
        'index.html',
        popular_books=popular_books,
        featured_picks=featured_picks,
        stats=stats
    )

@app.route('/recommend', methods=['GET', 'POST'])
def recommend_page():
    """Interactive Book Recommendation Studio."""
    book_title = request.args.get('book', '').strip()
    top_n = request.args.get('top_n', 6, type=int)
    top_n = max(3, min(top_n, 12))  # clamp between 3 and 12
    
    if request.method == 'POST':
        book_title = request.form.get('book_name', '').strip()
        top_n = request.form.get('top_n', 6, type=int)
        top_n = max(3, min(top_n, 12))

    recommendation_data = None
    if book_title:
        recommendation_data = recommender.recommend(book_title, top_n=top_n)

    sample_books = recommender.get_random_famous_books(count=8)

    return render_template(
        'recommend.html',
        selected_book=book_title,
        top_n=top_n,
        rec_data=recommendation_data,
        sample_books=sample_books
    )

@app.route('/explore')
def explore_page():
    """Catalog Explorer with Search, Filters, and Pagination."""
    page = request.args.get('page', 1, type=int)
    search_query = request.args.get('q', '').strip()
    author_filter = request.args.get('author', '').strip()
    
    catalog_data = recommender.explore_catalog(
        page=page,
        per_page=24,
        search=search_query if search_query else None,
        author=author_filter if author_filter else None
    )
    
    return render_template(
        'explore.html',
        books=catalog_data['books'],
        total=catalog_data['total'],
        pages=catalog_data['pages'],
        page=catalog_data['page'],
        search_query=search_query,
        author_filter=author_filter
    )

@app.route('/reading-list')
def reading_list():
    """Personal Reading List & Bookmarks page."""
    return render_template('reading_list.html')

@app.route('/about')
def about():
    """Architecture and Machine Learning Algorithm Details."""
    stats = recommender.get_stat
    s()
    return render_template('about.html', stats=stats)

# ==========================================
# REST API ENDPOINTS
# ==========================================

@app.route('/api/search')
def api_search():
    """API for real-time live search autocomplete."""
    query = request.args.get('q', '').strip()
    limit = request.args.get('limit', 8, type=int)
    results = recommender.search_books(query, limit=limit)
    return jsonify({'results': results})

@app.route('/api/recommend')
def api_recommend():
    """API endpoint for fetching recommendations asynchronously."""
    book_title = request.args.get('book', '').strip()
    top_n = request.args.get('top_n', 6, type=int)
    top_n = max(1, min(top_n, 20))
    
    if not book_title:
        return jsonify({'success': False, 'message': 'Please provide a book title.'}), 400
        
    result = recommender.recommend(book_title, top_n=top_n)
    return jsonify(result)

@app.route('/api/random')
def api_random():
    """API endpoint for random famous book suggestions."""
    count = request.args.get('count', 6, type=int)
    random_books = recommender.get_random_famous_books(count=count)
    return jsonify({'books': random_books})

@app.route('/api/stats')
def api_stats():
    """API endpoint returning dataset and model statistics."""
    return jsonify(recommender.get_stats())

@app.route('/api/popular')
def api_popular():
    """API endpoint returning Top 50 Popular Books."""
    limit = request.args.get('limit', 50, type=int)
    popular = recommender.get_popular_books(limit=limit)
    return jsonify({'books': popular})

@app.route('/api/book/<path:title>')
def api_book_details(title):
    """API endpoint for specific book metadata."""
    meta = recommender.get_book_metadata(title)
    return jsonify(meta)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"[*] Starting Book Recommender Web Application at http://127.0.0.1:{port}")
    app.run(host='0.0.0.0', port=port)
