# 📚 BookVerse - Intelligent Book Recommender System

A modern, production-grade Web Application and Recommender System engineered with **Collaborative Filtering (Cosine Similarity)** and **Popularity-Based Ranking**, trained on **1.14+ Million reader ratings** across **271,000+ books** from the Book-Crossing dataset based on `Project_3.ipynb`.

---

## 🌟 Key Features

### 1. 🏆 Top 50 Most Popular Books
- Curated using a statistical threshold of at least **250 verified reader ratings** to eliminate statistical noise.
- Ranked by weighted average rating score.
- Displays high-resolution book covers, star ratings (out of 10), author, publication year, publisher, and total votes.
- Real-time client-side search/filter bar to filter top 50 books instantly.

### 2. 🔮 Interactive Recommendation Studio
- Powered by **Item-Item Collaborative Filtering** using **Cosine Similarity** across a 679 &times; 811 reader pivot matrix.
- **Real-time Live Autocomplete Search**: Start typing any letter to get instant book suggestions with thumbnails.
- **Customizable Recommendations Count**: Select Top 4, 6, 8, 10, or 12 similar books.
- **Similarity Affinity Score Meter**: Displays match percentage (e.g., `94.2% Match`) with animated progress bars.
- **Deep Recommendation Chaining**: Click "Recommend Next" on any recommended book to immediately generate recommendations based on it.
- **"Surprise Me" Lucky Picker**: Randomly selects a classic from the collaborative matrix for quick inspiration.

### 3. 🧭 Catalog Explorer (271,000+ Books)
- Search and browse through the entire catalog by title or author.
- Integrated pagination and quick book preview modal with Amazon & Goodreads links.

### 4. 🔖 Personal Reading List / Bookmarks
- Bookmark favorite books with a single click.
- Stored locally in your browser session (`localStorage`).
- Interactive Bookshelf view with one-click recommendation links and text export.

### 5. 🎨 Polished Modern Glassmorphism UI
- Seamless **Dark & Light Mode** toggle with local persistence.
- 3D card tilt & hover lift animations, glowing badges, and custom scrollbars.
- **Bulletproof Image Fallback**: Automatically renders an elegant SVG book cover placeholder with book title & author if external Amazon CDN images are blocked or missing.
- Global keyboard shortcut: `Ctrl + K` to focus search instantly.

---

## 🏗️ Project Architecture & File Structure

```
Projects/Books_Recommender/
├── Project_3.ipynb             # Original Jupyter Notebook (EDA, Data Prep, Cosine Similarity)
├── Books.csv                   # Books metadata dataset (271,360 books)
├── Ratings.csv                 # Ratings dataset (1,149,780 ratings)
├── Users.csv                   # Users demographic dataset (278,858 users)
├── generate_artifacts.py       # Script to clean datasets & build model pickle files
├── recommender.py              # Recommender Engine module (CF, Popularity, Search)
├── app.py                      # Flask Application server with web routes & REST APIs
├── requirements.txt            # Python dependencies
├── model/                      # Serialized ML artifacts directory
│   ├── popular.pkl             # Top 50 Popular Books DataFrame
│   ├── pt.pkl                  # 679 x 811 User-Item Pivot Table
│   ├── similarity_scores.pkl   # Pre-computed Cosine Similarity Matrix
│   ├── book_dict.pkl           # Fast O(1) book metadata lookup dictionary
│   ├── book_names.pkl          # List of 679 matrix titles
│   ├── books.pkl               # Unique catalog DataFrame
│   └── stats.pkl               # Model summary metrics
├── static/
│   ├── css/
│   │   └── style.css           # Custom Glassmorphism Dark/Light CSS
│   └── js/
│       └── app.js              # Real-time search, bookmarks, theme switcher JS
└── templates/
    ├── base.html               # Base layout with navbar, search, modal & footer
    ├── index.html              # Home page with Top 50 showcase & Hero
    ├── recommend.html          # Recommendation Studio & Similarity cards
    ├── explore.html            # 271K+ Catalog Browser with pagination
    ├── reading_list.html       # Personal Reading List bookshelf
    └── about.html              # ML Architecture & REST API documentation
```

---

## 🚀 How to Run the Project

### 1. Prerequisites
Ensure you have Python 3.9+ (Python 3.13 tested) installed.

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Generate ML Artifacts (Already pre-generated)
If you modify `Books.csv` or `Ratings.csv`, regenerate the models:
```bash
python generate_artifacts.py
```

### 4. Start the Web Server
```bash
python app.py
```
Open your browser and navigate to:
```
http://127.0.0.1:5000
```

---

## 📡 REST API Documentation

| Endpoint | Method | Description |
|---|---|---|
| `/api/recommend?book=<title>&top_n=<n>` | GET | Get top N recommendations for a book with similarity scores |
| `/api/search?q=<query>&limit=<n>` | GET | Autocomplete live search across titles and authors |
| `/api/popular?limit=50` | GET | Returns the Top 50 most popular books |
| `/api/random?count=6` | GET | Returns random famous books from the CF matrix |
| `/api/book/<title>` | GET | Returns full metadata for a specific book |
| `/api/stats` | GET | Returns summary dataset and model metrics |

---

## 🧠 Machine Learning Methodology (from `Project_3.ipynb`)

1. **Popularity-Based Filtering**:
   $$\text{Threshold: } \text{Count}(\text{Ratings}) \ge 250$$
   Sorted descending by $\text{Mean}(\text{Rating})$, generating a robust cold-start Top 50 list.

2. **Collaborative Filtering Dimensionality Reduction**:
   - Filter active readers with $> 200$ ratings (811 users).
   - Filter famous books with $> 50$ ratings from active readers (679 books).
   - Construct Pivot Table $P_{679 \times 811}$.

3. **Cosine Similarity Vector Metric**:
   $$\text{Similarity}(A, B) = \frac{A \cdot B}{\|A\|_2 \|B\|_2} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}$$
   Enables sub-millisecond retrieval of the closest books in reader vector space!
