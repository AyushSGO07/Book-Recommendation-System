import os
import time
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def build_artifacts(base_dir=None):
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    model_dir = os.path.join(base_dir, 'model')
    os.makedirs(model_dir, exist_ok=True)
    
    print("=" * 60)
    print("[*] Starting Artifact Generation for Book Recommender System")
    print("=" * 60)
    start_time = time.time()
    
    books_csv = os.path.join(base_dir, 'Books.csv')
    ratings_csv = os.path.join(base_dir, 'Ratings.csv')
    users_csv = os.path.join(base_dir, 'Users.csv')
    
    print(f"[*] Reading Books dataset from {books_csv}...")
    books = pd.read_csv(books_csv, low_memory=False)
    
    print(f"[*] Reading Ratings dataset from {ratings_csv}...")
    ratings = pd.read_csv(ratings_csv)
    
    print(f"[*] Reading Users dataset from {users_csv}...")
    users = pd.read_csv(users_csv)
    
    # 1. Clean and normalize URLs to HTTPS
    print("[*] Normalizing image URLs to HTTPS...")
    for col in ['Image-URL-S', 'Image-URL-M', 'Image-URL-L']:
        if col in books.columns:
            books[col] = books[col].astype(str).str.replace('http://', 'https://', regex=False)
    
    # Fill missing values
    books['Book-Author'] = books['Book-Author'].fillna('Unknown Author')
    books['Publisher'] = books['Publisher'].fillna('Unknown Publisher')
    
    # Clean Ratings
    ratings = ratings.dropna()
    
    # Merge ratings with books metadata
    print("[*] Merging ratings with book metadata...")
    rating_with_name = ratings.merge(books, on='ISBN')
    
    # ==========================================
    # 2. POPULARITY-BASED MODEL (TOP 50 BOOKS)
    # ==========================================
    print("[*] Calculating Popularity-based Top 50 Books...")
    num_rating_df = rating_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'Num-Rating'}, inplace=True)
    
    avg_rating_df = rating_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'Avg-Rating'}, inplace=True)
    
    popularity_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    # Filter books with at least 250 ratings, sorted by highest average rating
    popular_filtered = popularity_df[popularity_df['Num-Rating'] >= 250].sort_values(
        by='Avg-Rating', ascending=False
    ).reset_index(drop=True).head(50)
    
    popular_df = popular_filtered.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
        ['Book-Title', 'Book-Author', 'Image-URL-M', 'Image-URL-L', 'Image-URL-S', 'Num-Rating', 'Avg-Rating', 'Year-Of-Publication', 'Publisher', 'ISBN']
    ].reset_index(drop=True)
    popular_df['Avg-Rating'] = popular_df['Avg-Rating'].round(2)
    
    print(f"    -> Found {len(popular_df)} top popular books.")
    
    # ==========================================
    # 3. COLLABORATIVE FILTERING MODEL
    # ==========================================
    print("[*] Filtering frequent raters (> 200 ratings) and famous books (> 50 ratings)...")
    user_counts = rating_with_name.groupby('User-ID').count()['Book-Rating']
    active_users = user_counts[user_counts > 200].index
    filtered_rating = rating_with_name[rating_with_name['User-ID'].isin(active_users)]
    
    book_counts = filtered_rating.groupby('Book-Title').count()['Book-Rating']
    famous_books = book_counts[book_counts > 50].index
    final_rating = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]
    
    print(f"[*] Building user-item pivot table ({len(famous_books)} books x {len(active_users)} users)...")
    pt = final_rating.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating').fillna(0)
    
    print("[*] Computing Cosine Similarity matrix...")
    similarity_scores = cosine_similarity(pt)
    
    # ==========================================
    # 4. OPTIMIZED BOOK LOOKUP METADATA
    # ==========================================
    print("[*] Creating fast book metadata lookup table...")
    all_unique_books = books.drop_duplicates('Book-Title')[
        ['Book-Title', 'Book-Author', 'Image-URL-M', 'Image-URL-L', 'Image-URL-S', 'Year-Of-Publication', 'Publisher', 'ISBN']
    ]
    
    book_dict = {}
    for _, row in all_unique_books.iterrows():
        book_dict[row['Book-Title']] = {
            'title': str(row['Book-Title']),
            'author': str(row['Book-Author']),
            'image_m': str(row['Image-URL-M']),
            'image_l': str(row['Image-URL-L']),
            'image_s': str(row['Image-URL-S']),
            'year': str(row['Year-Of-Publication']),
            'publisher': str(row['Publisher']),
            'isbn': str(row['ISBN'])
        }
    
    book_titles_list = list(pt.index)
    
    # Save artifacts
    print("[*] Serializing and saving artifacts to disk...")
    with open(os.path.join(model_dir, 'popular.pkl'), 'wb') as f:
        pickle.dump(popular_df, f)
        
    with open(os.path.join(model_dir, 'pt.pkl'), 'wb') as f:
        pickle.dump(pt, f)
        
    with open(os.path.join(model_dir, 'similarity_scores.pkl'), 'wb') as f:
        pickle.dump(similarity_scores, f)
        
    with open(os.path.join(model_dir, 'book_dict.pkl'), 'wb') as f:
        pickle.dump(book_dict, f)
        
    with open(os.path.join(model_dir, 'book_names.pkl'), 'wb') as f:
        pickle.dump(book_titles_list, f)
        
    with open(os.path.join(model_dir, 'books.pkl'), 'wb') as f:
        pickle.dump(all_unique_books, f)
    
    # Summary stats
    stats = {
        'total_books_in_raw': len(books),
        'total_ratings_in_raw': len(ratings),
        'total_users_in_raw': len(users),
        'top_popular_count': len(popular_df),
        'collaborative_books_count': len(pt.index),
        'active_users_count': len(active_users),
        'generation_time_seconds': round(time.time() - start_time, 2)
    }
    
    with open(os.path.join(model_dir, 'stats.pkl'), 'wb') as f:
        pickle.dump(stats, f)
        
    print("=" * 60)
    print(f"[OK] Successfully built and saved all artifacts in {stats['generation_time_seconds']}s")
    print(f"   - Popular Books: {stats['top_popular_count']}")
    print(f"   - Collaborative Filtering Books: {stats['collaborative_books_count']}")
    print(f"   - Active Users: {stats['active_users_count']}")
    print("=" * 60)
    return stats

if __name__ == '__main__':
    build_artifacts()
