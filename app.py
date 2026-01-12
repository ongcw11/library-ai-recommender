from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
import pandas as pd
import numpy as np
from content_lsa_recommender import FinalOptimizedRecommender
from data_preparation import get_cleaned_data, EnhancedTextProcessor, EnhancedFeatureEngineer
from hybrid_recommender import HybridRecommender
from collaborative_recommender import CollaborativeRecommender
from lsa_ann_recommender import initialize_lsa_ann_recommender
try:
	from dl_simple_recommender import initialize_simple_deep_learning
	DEEP_LEARNING_AVAILABLE = True
	DEEP_LEARNING_TYPE = "simple"
	print("Using simplified neural network system...")
except ImportError as e:
	print(f"Simple neural network not available: {e}")
	try:
		from dl_lightweight_recommender import initialize_lightweight_deep_learning
		DEEP_LEARNING_AVAILABLE = True
		DEEP_LEARNING_TYPE = "lightweight"
		print("Using lightweight neural network system...")
	except ImportError as e2:
		print(f"Lightweight neural network not available: {e2}")
		print("Falling back to LSA-only system...")
		DEEP_LEARNING_AVAILABLE = False
		DEEP_LEARNING_TYPE = "none"

HYBRID_RE_RANK_ENABLED = False
import json

app = Flask(__name__)
app.secret_key = 'your-secret-key-here'

user_reading_lists = {}

print("Initializing recommendation system...")
try:
    # Get data only when needed (lazy loading)
    cleaned_data = get_cleaned_data()
    books = cleaned_data['books']
    ratings = cleaned_data['ratings']
    book_tags = cleaned_data['book_tags']
    tags = cleaned_data['tags']
    
    recommender = FinalOptimizedRecommender(books, ratings, book_tags, tags, n_components=100, precision_k=10)
    print("LSA recommendation system ready!")
    try:
        hybrid_recommender = HybridRecommender(weight_cf=0.6, weight_cbf=0.4)
        print("Hybrid recommender ready!")
    except Exception as he:
        print(f"Hybrid recommender init failed: {he}")
        hybrid_recommender = None
    try:
        collaborative_recommender = CollaborativeRecommender(n_components=50)
        print("Collaborative recommender ready!")
    except Exception as ce:
        print(f"Collaborative recommender init failed: {ce}")
        collaborative_recommender = None
    
    # Initialize LSA + ANN Recommender (lazy loading)
    lsa_ann_recommender = None
    lsa_ann_available = False
    print("LSA + ANN recommender will be initialized on first use")

    if DEEP_LEARNING_AVAILABLE:
        print("Initializing deep learning system...")
        try:
            if DEEP_LEARNING_TYPE == "simple":
                deep_learning_system = initialize_simple_deep_learning(
                    books, ratings, book_tags, tags
                )
                print("Simple neural network system ready!")
            else:  # lightweight
                deep_learning_system = initialize_lightweight_deep_learning(
                    books, ratings, book_tags, tags
                )
                print("Lightweight neural network system ready!")
            use_deep_learning = True
        except Exception as dl_error:
            print(f"Deep learning system failed to initialize: {dl_error}")
            print("Using LSA-only system...")
            deep_learning_system = None
            use_deep_learning = False
    else:
        deep_learning_system = None
        use_deep_learning = False
        
except Exception as e:
    print(f"Error initializing recommender: {e}")
    print("Using fallback data loading...")
    import pandas as pd
    books = pd.read_csv('data/books.csv')
    ratings = pd.read_csv('data/ratings.csv')
    book_tags = pd.read_csv('data/book_tags.csv')
    tags = pd.read_csv('data/tags.csv')
    
    recommender = None
    deep_learning_system = None
    use_deep_learning = False
    lsa_ann_recommender = None
    lsa_ann_available = False
    print("Fallback data loaded!")

def initialize_lsa_ann_system():
    """Initialize LSA + ANN system when needed"""
    global lsa_ann_recommender, lsa_ann_available
    
    if lsa_ann_recommender is None:
        try:
            print("Initializing LSA + ANN system (first time use)...")
            lsa_ann_recommender = initialize_lsa_ann_recommender(
                books, ratings, book_tags, tags,
                enable_tuning=False,  # Skip tuning for faster startup
                tuning_mode='fast'
            )
            lsa_ann_available = True
            print("LSA + ANN recommender ready!")
        except Exception as lae:
            print(f"LSA + ANN recommender init failed: {lae}")
            lsa_ann_recommender = None
            lsa_ann_available = False
    
    return lsa_ann_recommender, lsa_ann_available

def get_trending_books(n=10):
    """Get trending books based on average rating and number of ratings"""
    complete_books = books.dropna(subset=['title', 'authors'])
    complete_books = complete_books[complete_books['title'].str.strip() != '']
    complete_books = complete_books[complete_books['authors'].str.strip() != '']
    
    book_stats = ratings.groupby('book_id').agg({
        'rating': ['mean', 'count']
    }).round(2)
    book_stats.columns = ['avg_rating', 'rating_count']
    
    complete_book_ids = set(complete_books['book_id'])
    complete_book_stats = book_stats[book_stats.index.isin(complete_book_ids)]
    complete_book_stats = complete_book_stats[complete_book_stats['rating_count'] >= 5]  # Lower threshold
    
    trending = complete_book_stats.sort_values(['avg_rating', 'rating_count'], ascending=[False, False]).head(n)
    
    trending = trending.reset_index()
    
    trending_books = trending.merge(complete_books, on='book_id', how='inner')
    
    result = trending_books[['book_id', 'title', 'authors', 'average_rating', 'ratings_count', 'original_publication_year', 'small_image_url']].head(n)
    
    result['average_rating'] = result['average_rating'].fillna(0)
    result['ratings_count'] = result['ratings_count'].fillna(0)
    result['original_publication_year'] = result['original_publication_year'].fillna(0)
    result['small_image_url'] = result['small_image_url'].fillna('')
    
    return result

def _normalize_scores(series):
    """Normalize a pandas Series to 0-1; return zeros if constant or empty."""
    if series is None or len(series) == 0:
        return series
    s_min = series.min()
    s_max = series.max()
    if pd.isna(s_min) or pd.isna(s_max) or s_max <= s_min:
        return pd.Series([0.0] * len(series), index=series.index)
    return (series - s_min) / (s_max - s_min)

def hybrid_recommend_for_user(user_id, n_recommendations=20, candidate_size=300, alpha=0.5):
    lsa_df = recommender.recommend_for_user(
        user_id,
        n_recommendations=candidate_size,
        use_percentage=False
    )
    if lsa_df is None or lsa_df.empty:
        return lsa_df

    if not (HYBRID_RE_RANK_ENABLED and use_deep_learning and deep_learning_system is not None):
        return lsa_df.head(n_recommendations)

    try:
        dl_df = deep_learning_system.recommend_for_user(
            user_id,
            n_recommendations=candidate_size,
            use_percentage=False
        )
    except Exception:
        return lsa_df.head(n_recommendations)

    lsa_tmp = lsa_df.copy()
    dl_tmp = dl_df.copy() if dl_df is not None else pd.DataFrame()

    if 'recommendation_score' not in lsa_tmp.columns:
        lsa_tmp['recommendation_score'] = 0.0
    if not dl_tmp.empty and 'recommendation_score' not in dl_tmp.columns:
        dl_tmp['recommendation_score'] = 0.0

    lsa_tmp['lsa_norm'] = _normalize_scores(lsa_tmp['recommendation_score'])
    if not dl_tmp.empty:
        dl_tmp['dl_norm'] = _normalize_scores(dl_tmp['recommendation_score'])
        dl_tmp = dl_tmp[['book_id', 'dl_norm']]
    else:
        dl_tmp = pd.DataFrame(columns=['book_id', 'dl_norm'])

    blended = lsa_tmp.merge(dl_tmp, on='book_id', how='left')
    blended['dl_norm'] = blended['dl_norm'].fillna(0.0)
    blended['blended_score'] = alpha * blended['lsa_norm'] + (1.0 - alpha) * blended['dl_norm']

    blended_sorted = blended.sort_values('blended_score', ascending=False)
    cols_to_return = [c for c in lsa_df.columns if c in blended_sorted.columns]
    return blended_sorted[cols_to_return].head(n_recommendations)

def search_books(query, limit=20):
    """Search books by title, author, ISBN, or tags with comprehensive genre matching"""
    if not query:
        return pd.DataFrame()
    
    query = query.lower()
    
    SPACE_TO_HYPHEN_VARIATIONS = {
        'science fiction': 'science-fiction',
        'sci fi': 'sci-fi', 
        'high fantasy': 'high-fantasy',
        'urban fantasy': 'urban-fantasy',
        'epic fantasy': 'epic-fantasy',
        'love story': 'love-story',
        'romantic fiction': 'romantic-fiction',
        'young adult': 'young-adult',
        'historical fiction': 'historical-fiction',
        'non fiction': 'non-fiction',
        'classical literature': 'classical-literature',
        'self help': 'self-help',
        'self improvement': 'self-improvement',
        'adventure travel': 'adventure-travel',
        'mental health': 'mental-health',
        'audio book': 'audio-book'
    }
    
    genre_books = pd.DataFrame()
    try:
        exact_tags = tags[tags['tag_name'].str.lower() == query]
        
        if exact_tags.empty:
            if query in SPACE_TO_HYPHEN_VARIATIONS:
                hyphen_query = SPACE_TO_HYPHEN_VARIATIONS[query]
                exact_tags = tags[tags['tag_name'].str.lower() == hyphen_query]
                if not exact_tags.empty:
                    print(f"Found match using hyphen variation: '{query}' → '{hyphen_query}'")
            
            elif ' ' in query:
                hyphen_query = query.replace(' ', '-')
                exact_tags = tags[tags['tag_name'].str.lower() == hyphen_query]
                if not exact_tags.empty:
                    print(f"Found match using general hyphen conversion: '{query}' → '{hyphen_query}'")
            
            elif '-' in query:
                space_query = query.replace('-', ' ')
                exact_tags = tags[tags['tag_name'].str.lower() == space_query]
                if not exact_tags.empty:
                    print(f"Found match using space conversion: '{query}' → '{space_query}'")
        
        if not exact_tags.empty:
            matching_tag_ids = exact_tags['tag_id'].tolist()
            if 'goodreads_book_id' in book_tags.columns:
                matching_book_ids = book_tags[book_tags['tag_id'].isin(matching_tag_ids)]['goodreads_book_id'].tolist()
            else:
                matching_book_ids = book_tags[book_tags['tag_id'].isin(matching_tag_ids)]['book_id'].tolist()
            
            if matching_book_ids:
                genre_books = books[books['book_id'].isin(matching_book_ids)]
                print(f"Found {len(genre_books)} books with exact genre match: '{query}'")
                
                if query == 'romance':
                    romance_subgenres = ['contemporary-romance', 'historical-romance', 'paranormal-romance', 
                                       'romance-novels', 'adult-romance', 'new-adult-romance']
                    
                    for subgenre in romance_subgenres:
                        subgenre_tags = tags[tags['tag_name'].str.lower() == subgenre]
                        if not subgenre_tags.empty:
                            subgenre_tag_ids = subgenre_tags['tag_id'].tolist()
                            if 'goodreads_book_id' in book_tags.columns:
                                subgenre_book_ids = book_tags[book_tags['tag_id'].isin(subgenre_tag_ids)]['goodreads_book_id'].tolist()
                            else:
                                subgenre_book_ids = book_tags[book_tags['tag_id'].isin(subgenre_tag_ids)]['book_id'].tolist()
                            
                            if subgenre_book_ids:
                                subgenre_books = books[books['book_id'].isin(subgenre_book_ids)]
                                genre_books = pd.concat([subgenre_books, genre_books]).drop_duplicates(subset=['book_id'])
                                print(f"Added {len(subgenre_books)} books from subgenre: {subgenre}")
    except Exception as e:
        print(f"Genre search error: {e}")
    
    if not genre_books.empty:
        if query == 'romance':
            primary_other_genres = ['fantasy', 'young-adult', 'science-fiction', 'mystery', 'thriller', 'horror', 'action']
            
            filtered_books = []
            for _, book in genre_books.iterrows():
                book_id = book['book_id']
                if 'goodreads_book_id' in book_tags.columns:
                    book_tag_ids = book_tags[book_tags['goodreads_book_id'] == book_id]['tag_id'].tolist()
                else:
                    book_tag_ids = book_tags[book_tags['book_id'] == book_id]['tag_id'].tolist()
                
                # Get tag names
                book_tag_names = tags[tags['tag_id'].isin(book_tag_ids)]['tag_name'].tolist()
                
                # Check if book is primarily other genres
                is_primarily_other_genre = any(genre in ' '.join(book_tag_names).lower() for genre in primary_other_genres)
                
                # Only include if it's not primarily other genres
                if not is_primarily_other_genre:
                    filtered_books.append(book)
            
            if filtered_books:
                results = pd.DataFrame(filtered_books).sort_values(['average_rating', 'ratings_count'], ascending=[False, False])
                print(f"Filtered to {len(results)} books that are primarily romance (excluded {len(genre_books) - len(results)} books that are primarily other genres)")
            else:
                results = genre_books.sort_values(['average_rating', 'ratings_count'], ascending=[False, False])
        else:
            # For other genres, sort by average rating first, then by ratings count
            results = genre_books.sort_values(['average_rating', 'ratings_count'], ascending=[False, False])
    else:
        title_mask = books['title'].str.lower().str.contains(query, na=False)
        author_mask = books['authors'].str.lower().str.contains(query, na=False)
        isbn_mask = books['isbn'].str.contains(query, na=False)
        
        # Search in tags (partial matches)
        tag_mask = pd.Series([False] * len(books), index=books.index)
        
        try:
            # Find books with matching tags (partial match)
            matching_tags = tags[tags['tag_name'].str.contains(query, case=False, na=False)]
            if not matching_tags.empty:
                matching_tag_ids = matching_tags['tag_id'].tolist()
                if 'goodreads_book_id' in book_tags.columns:
                    matching_book_ids = book_tags[book_tags['tag_id'].isin(matching_tag_ids)]['goodreads_book_id'].tolist()
                else:
                    matching_book_ids = book_tags[book_tags['tag_id'].isin(matching_tag_ids)]['book_id'].tolist()
                tag_mask = books['book_id'].isin(matching_book_ids)
        except Exception as e:
            print(f"Tag search error: {e}")
            tag_mask = pd.Series([False] * len(books), index=books.index)
        
        # Combine all search criteria
        combined_mask = title_mask | author_mask | isbn_mask | tag_mask
        results = books[combined_mask].head(limit*2)  # Get more to filter
    
    # Filter out books with missing essential data
    results = results.dropna(subset=['title', 'authors'])
    results = results[results['title'].str.strip() != '']
    results = results[results['authors'].str.strip() != '']
    
    # Sort by rating count (popularity) for better results
    results = results.sort_values('ratings_count', ascending=False)
    
    result = results[['book_id', 'title', 'authors', 'average_rating', 'ratings_count', 'original_publication_year', 'small_image_url']].head(limit)
    
    # Fill NaN values for optional fields only
    result['average_rating'] = result['average_rating'].fillna(0)
    result['ratings_count'] = result['ratings_count'].fillna(0)
    result['original_publication_year'] = result['original_publication_year'].fillna(0)
    result['small_image_url'] = result['small_image_url'].fillna('')
    
    return result

def get_user_reading_list(user_id):
    if 'user_id' not in session:
        return pd.DataFrame()
    
    # Get user's to_read list from the to_read table
    to_read_data = get_cleaned_data()['to_read']
    user_to_read = to_read_data[to_read_data['user_id'] == user_id]['book_id'].tolist()
    
    # Also include books from in-memory storage (for newly added books)
    if user_id in user_reading_lists:
        user_to_read.extend(user_reading_lists[user_id])
        user_to_read = list(set(user_to_read))  # Remove duplicates
    
    if not user_to_read:
        return pd.DataFrame()
    
    # Get book details for the reading list
    reading_list = books[books['book_id'].isin(user_to_read)]
    result = reading_list[['book_id', 'title', 'authors', 'average_rating', 'ratings_count', 'small_image_url']]
    
    # Fill NaN values
    result['title'] = result['title'].fillna('Unknown Title')
    result['authors'] = result['authors'].fillna('Unknown Author')
    result['average_rating'] = result['average_rating'].fillna(0)
    result['ratings_count'] = result['ratings_count'].fillna(0)
    
    return result

# Routes
@app.route('/')
def home():
    # Get trending books
    trending_books = get_trending_books(10)
    
    # Get personalized recommendations if user is logged in
    personalized_recommendations = pd.DataFrame()
    if 'user_id' in session:
        user_id = session['user_id']
        try:
            if recommender is not None:
                personalized_recommendations = recommender.recommend_for_user(user_id, n_recommendations=10, use_percentage=True)
            else:
                # Fallback: get trending books as recommendations
                personalized_recommendations = get_trending_books(10)
        except Exception as e:
            print(f"Home recommendation error: {e}")
            personalized_recommendations = pd.DataFrame()
        
        # Fill NaN values in recommendations
        if not personalized_recommendations.empty:
            personalized_recommendations['title'] = personalized_recommendations['title'].fillna('Unknown Title')
            personalized_recommendations['authors'] = personalized_recommendations['authors'].fillna('Unknown Author')
            personalized_recommendations['average_rating'] = personalized_recommendations['average_rating'].fillna(0)
            
            # Ensure recommendation_score column exists
            if 'recommendation_score' not in personalized_recommendations.columns:
                personalized_recommendations['recommendation_score'] = 0.0
    
    return render_template('home.html', 
                         trending_books=trending_books,
                         personalized_recommendations=personalized_recommendations)

@app.route('/search')
def search():
    query = request.args.get('q', '')
    algorithm = request.args.get('algorithm', 'content-based')
    page = int(request.args.get('page', 1))
    per_page = 25
    
    # Unified search with optional re-ranking
    if True:
        # Get all results first to calculate pagination
        all_results = search_books(query, limit=1000)  # Get more results for pagination
        algo = request.args.get('algorithm')
        # Optional hybrid re-rank
        if algo == 'hybrid' and 'user_id' in session and hybrid_recommender is not None:
            try:
                user_id = session['user_id']
                hybrid_scores = hybrid_recommender.get_hybrid_recommendations(user_id, top_n=200)
                if not hybrid_scores.empty:
                    hybrid_scores = hybrid_scores[['book_id', 'final_score']]
                    all_results = all_results.merge(hybrid_scores, on='book_id', how='left')
                    all_results['final_score'] = all_results['final_score'].fillna(0.0)
                    all_results = all_results.sort_values(['final_score','ratings_count','average_rating'], ascending=[False, False, False])
            except Exception as e:
                print(f"Hybrid re-rank error: {e}")
        # Optional collaborative re-rank
        if algo == 'collaborative' and 'user_id' in session and collaborative_recommender is not None:
            try:
                user_id = session['user_id']
                cf_scores = collaborative_recommender.recommend_for_user(user_id, top_n=200, mode='svd')
                if not cf_scores.empty:
                    cf_scores = cf_scores[['book_id', 'CF_score']]
                    all_results = all_results.merge(cf_scores, on='book_id', how='left')
                    all_results['CF_score'] = all_results['CF_score'].fillna(0.0)
                    all_results = all_results.sort_values(['CF_score','ratings_count','average_rating'], ascending=[False, False, False])
            except Exception as e:
                print(f"Collaborative re-rank error: {e}")
        total_results = len(all_results)
        total_pages = (total_results + per_page - 1) // per_page  # Ceiling division
        # Apply pagination
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        results = all_results.iloc[start_idx:end_idx]
        message = None
    
    # Calculate page numbers to show (max 5 pages)
    start_page = max(1, page - 2)
    end_page = min(total_pages, start_page + 4)
    if end_page - start_page < 4:
        start_page = max(1, end_page - 4)
    
    page_numbers = list(range(start_page, end_page + 1))
    
    return render_template('search.html', 
                         query=query, 
                         results=results, 
                         algorithm=algorithm, 
                         message=message,
                         page=page,
                         total_pages=total_pages,
                         total_results=total_results,
                         page_numbers=page_numbers)

@app.route('/book/<int:book_id>')
def book_details(book_id):
    book = books[books['book_id'] == book_id]
    if book.empty:
        return "Book not found", 404
    
    book = book.iloc[0]
    
    # Fill NaN values for book
    book['title'] = book['title'] if pd.notna(book['title']) else 'Unknown Title'
    book['authors'] = book['authors'] if pd.notna(book['authors']) else 'Unknown Author'
    book['average_rating'] = book['average_rating'] if pd.notna(book['average_rating']) else 0
    book['ratings_count'] = book['ratings_count'] if pd.notna(book['ratings_count']) else 0
    
    # Get similar books using the recommender
    similar_books = recommender.get_similar_books(book_id, n_recommendations=6, use_percentage=True)
    if not similar_books.empty:
        similar_books['title'] = similar_books['title'].fillna('Unknown Title')
        similar_books['authors'] = similar_books['authors'].fillna('Unknown Author')
        similar_books['average_rating'] = similar_books['average_rating'].fillna(0)
    
    # Get book tags
    book_tag_mapping = recommender._create_book_tag_mapping()
    book_tags_list = book_tag_mapping.get(book_id, [])
    
    return render_template('book_details.html', 
                         book=book, 
                         similar_books=similar_books,
                         book_tags=book_tags_list)

@app.route('/profile')
def profile():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    # Get user's reading history - only books that exist in the books table
    user_ratings = ratings[ratings['user_id'] == user_id].sort_values('rating', ascending=False)
    
    # Filter to only include ratings for books that exist in the books table
    existing_book_ids = set(books['book_id'])
    user_ratings = user_ratings[user_ratings['book_id'].isin(existing_book_ids)]
    
    # Merge with books table
    user_books = user_ratings.merge(books, on='book_id', how='inner')
    
    # Filter out books with missing essential data
    user_books = user_books.dropna(subset=['title', 'authors'])
    user_books = user_books[user_books['title'].str.strip() != '']
    user_books = user_books[user_books['authors'].str.strip() != '']
    
    # Fill NaN values for optional fields only
    if not user_books.empty:
        user_books['average_rating'] = user_books['average_rating'].fillna(0)
        user_books['ratings_count'] = user_books['ratings_count'].fillna(0)
        user_books['small_image_url'] = user_books['small_image_url'].fillna('')
    
    # Get personalized recommendations
    recommendations = recommender.recommend_for_user(user_id, n_recommendations=10, use_percentage=True)
    if not recommendations.empty:
        recommendations['title'] = recommendations['title'].fillna('Unknown Title')
        recommendations['authors'] = recommendations['authors'].fillna('Unknown Author')
        recommendations['average_rating'] = recommendations['average_rating'].fillna(0)
        recommendations['small_image_url'] = recommendations['small_image_url'].fillna('')
    
    # Get reading list
    reading_list = get_user_reading_list(user_id)
    if not reading_list.empty:
        reading_list['title'] = reading_list['title'].fillna('Unknown Title')
        reading_list['authors'] = reading_list['authors'].fillna('Unknown Author')
        reading_list['average_rating'] = reading_list['average_rating'].fillna(0)
        reading_list['small_image_url'] = reading_list['small_image_url'].fillna('')
    
    return render_template('profile.html', 
                         user_books=user_books,
                         recommendations=recommendations,
                         reading_list=reading_list)

@app.route('/reading-list')
def reading_list():
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    reading_list = get_user_reading_list(user_id)
    if not reading_list.empty:
        reading_list['title'] = reading_list['title'].fillna('Unknown Title')
        reading_list['authors'] = reading_list['authors'].fillna('Unknown Author')
        reading_list['average_rating'] = reading_list['average_rating'].fillna(0)
        reading_list['small_image_url'] = reading_list['small_image_url'].fillna('')
    
    # Get additional recommendations
    additional_recommendations = recommender.recommend_for_user(user_id, n_recommendations=5, use_percentage=True)
    if not additional_recommendations.empty:
        additional_recommendations['title'] = additional_recommendations['title'].fillna('Unknown Title')
        additional_recommendations['authors'] = additional_recommendations['authors'].fillna('Unknown Author')
        additional_recommendations['average_rating'] = additional_recommendations['average_rating'].fillna(0)
        additional_recommendations['small_image_url'] = additional_recommendations['small_image_url'].fillna('')
    
    return render_template('reading_list.html', 
                         reading_list=reading_list,
                         additional_recommendations=additional_recommendations)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page - only allows existing user IDs"""
    if request.method == 'POST':
        user_id = request.form.get('user_id')
        if user_id:
            try:
                user_id = int(user_id)
                # Check if user ID exists in ratings data
                if user_id in ratings['user_id'].values:
                    session['user_id'] = user_id
                    flash(f'Welcome back, User {user_id}!', 'success')
                    return redirect(url_for('home'))
                else:
                    flash(f'User ID {user_id} does not exist. Please enter a valid user ID.', 'error')
            except ValueError:
                flash('Please enter a valid numeric User ID', 'error')
        else:
            flash('Please enter a User ID', 'error')
    
    return render_template('login.html')

@app.route('/recommendations')
def recommendations():
    """Personalized recommendations page"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    algorithm = request.args.get('algorithm', 'content-based')
    
    try:
        if recommender is not None:
            if algorithm == 'lsa-ann':
                try:
                    # Initialize LSA + ANN system if not already done
                    lsa_ann_system, lsa_ann_available = initialize_lsa_ann_system()
                    
                    if lsa_ann_available and lsa_ann_system is not None:
                        recommendations = lsa_ann_system.recommend_for_user(
                            user_id, n_recommendations=20
                        )
                        system_type = "LSA + ANN (Advanced)"
                    else:
                        recommendations = pd.DataFrame()
                        system_type = "LSA + ANN Not Available"
                except Exception as lae:
                    print(f"LSA + ANN recommendations error: {lae}")
                    recommendations = pd.DataFrame()
                    system_type = "LSA + ANN Error"
            elif algorithm == 'hybrid' and hybrid_recommender is not None:
                try:
                    recommendations = hybrid_recommender.get_hybrid_recommendations(user_id, top_n=20, mode='svd')
                    system_type = "Hybrid (CF + CBF)"
                except Exception as he:
                    print(f"Hybrid recommendations error: {he}")
                    recommendations = pd.DataFrame()
                    system_type = "Hybrid Error"
            elif algorithm == 'collaborative' and collaborative_recommender is not None:
                recommendations = collaborative_recommender.recommend_for_user(user_id, top_n=20, mode='svd')
                system_type = "Collaborative Filtering"
            elif HYBRID_RE_RANK_ENABLED and use_deep_learning and deep_learning_system is not None:
                recommendations = hybrid_recommend_for_user(user_id, n_recommendations=20, candidate_size=300, alpha=0.6)
                system_type = "Hybrid (LSA+DL)"
            elif use_deep_learning and deep_learning_system is not None:
                recommendations = deep_learning_system.recommend_for_user(
                    user_id, n_recommendations=20, use_percentage=True
                )
                system_type = "Neural Network"
            else:
                recommendations = recommender.recommend_for_user(
                    user_id, n_recommendations=20, use_percentage=True
                )
                system_type = "LSA-Enhanced"
        else:
            # Fallback: get trending books as recommendations
            recommendations = get_trending_books(20)
            system_type = "Trending Books"
            flash('Using trending books as recommendations (advanced system unavailable)', 'info')
        
        if recommendations.empty:
            flash('No recommendations available. Please rate some books first!', 'info')
            recommendations = pd.DataFrame()
    except Exception as e:
        print(f"Recommendation error: {e}")
        flash('Unable to generate recommendations at this time.', 'error')
        recommendations = pd.DataFrame()
        system_type = "Error"
    
    # Convert recommendations to list of dictionaries for template
    if not recommendations.empty:
        recommendations_list = recommendations.to_dict('records')
    else:
        recommendations_list = []
    
    return render_template('recommendations.html', 
                         recommendations=recommendations_list,
                         user_id=user_id,
                         system_type=system_type,
                         algorithm=algorithm)

@app.route('/logout')
def logout():
    session.pop('user_id', None)
    return redirect(url_for('home'))

@app.route('/api/add-to-reading-list', methods=['POST'])
def add_to_reading_list():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    book_id = data.get('book_id')
    
    if not book_id:
        return jsonify({'error': 'Book ID required'}), 400
    
    user_id = session['user_id']
    
    # Check if book exists
    if book_id not in books['book_id'].values:
        return jsonify({'error': 'Book not found'}), 404
    
    # Initialize user's reading list if it doesn't exist
    if user_id not in user_reading_lists:
        user_reading_lists[user_id] = []
    
    # Add book to reading list if not already there
    if book_id not in user_reading_lists[user_id]:
        user_reading_lists[user_id].append(book_id)
        return jsonify({'success': True, 'message': 'Book added to reading list'})
    else:
        return jsonify({'success': False, 'message': 'Book already in reading list'})

@app.route('/api/remove-from-reading-list', methods=['POST'])
def remove_from_reading_list():
    if 'user_id' not in session:
        return jsonify({'error': 'Not logged in'}), 401
    
    data = request.get_json()
    book_id = data.get('book_id')
    
    if not book_id:
        return jsonify({'error': 'Book ID required'}), 400
    
    user_id = session['user_id']
    
    # Remove book from reading list
    if user_id in user_reading_lists and book_id in user_reading_lists[user_id]:
        user_reading_lists[user_id].remove(book_id)
        return jsonify({'success': True, 'message': 'Book removed from reading list'})
    else:
        return jsonify({'success': False, 'message': 'Book not found in reading list'})

@app.route('/api/recommendations/<int:user_id>')
def api_recommendations(user_id):
    """API endpoint for getting recommendations"""
    recommendations = recommender.recommend_for_user(user_id, n_recommendations=10, use_percentage=True)
    return jsonify(recommendations.to_dict('records'))

@app.route('/api/similar-books/<int:book_id>')
def api_similar_books(book_id):
    """API endpoint for getting similar books"""    
    # Use LSA system directly (deep learning disabled)
    if recommender is not None:
        similar_books = recommender.get_similar_books(book_id, n_recommendations=10, use_percentage=True)
    else:
        similar_books = pd.DataFrame()
    
    return jsonify(similar_books.to_dict('records'))

@app.route('/compare-systems')
def compare_systems():
    """Compare different recommendation systems"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    
    user_id = session['user_id']
    
    try:
        flash('Deep learning system disabled - using LSA-only system', 'info')
        return redirect(url_for('recommendations'))
            
    except Exception as e:
        print(f"System comparison error: {e}")
        flash('Unable to compare systems at this time.', 'error')
        return redirect(url_for('recommendations'))

@app.route('/api/deep-learning-recommendations/<int:user_id>')
def api_deep_learning_recommendations(user_id):
    """API endpoint for deep learning recommendations"""
    return jsonify({'error': 'Deep learning system disabled - using LSA-only system'}), 503

@app.route('/api/lsa-ann-recommendations/<int:user_id>')
def api_lsa_ann_recommendations(user_id):
    """API endpoint for LSA + ANN recommendations"""
    try:
        # Initialize LSA + ANN system if not already done
        lsa_ann_system, lsa_ann_available = initialize_lsa_ann_system()
        
        if not lsa_ann_available or lsa_ann_system is None:
            return jsonify({'error': 'LSA + ANN system not available'}), 503
        
        recommendations = lsa_ann_system.recommend_for_user(
            user_id, n_recommendations=10
        )
        
        if recommendations.empty:
            return jsonify({'recommendations': [], 'message': 'No recommendations available'})
        
        # Convert to JSON-serializable format
        rec_list = []
        for _, rec in recommendations.iterrows():
            rec_list.append({
                'book_id': int(rec['book_id']),
                'title': rec['title'],
                'authors': rec['authors'],
                'predicted_rating': float(rec.get('predicted_rating', 0)),
                'confidence': float(rec.get('confidence', 0))
            })
        
        return jsonify({
            'recommendations': rec_list,
            'system_type': 'LSA + ANN (Advanced)',
            'user_id': user_id
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
