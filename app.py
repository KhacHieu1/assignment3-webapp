from flask import Flask, render_template, request
import pandas as pd
import re

# Import machine learning tools for review rating prediction
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    SKLEARN_READY = True
except ImportError:
    SKLEARN_READY = False

app = Flask(__name__)

# Load processed.csv for search functionality
# Load original CSV to get review_rating which wasn't saved in processed.csv
df = pd.read_csv('processed.csv')
df_original = pd.read_csv('cosmetics_beauty_products_reviews.csv')

# Merge review_rating from original into our main dataframe
df = df.merge(df_original[['review_id', 'review_rating']], 
              on='review_id', 
              how='left')
# Total number of product images available in static folder
NUM_IMAGES = 6

# Store new reviews while the app is running
new_reviews = []


def clean_review_text(review_text):
    # Clean new review text before sending it to the ML model
    review_text = str(review_text).lower()
    review_text = re.sub(r'[^a-z0-9\s]', ' ', review_text)
    review_text = re.sub(r'\s+', ' ', review_text).strip()
    return review_text


def rating_to_sentiment(rating):
    # Convert predicted star rating into an easy label for the web page
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'


def train_review_model():
    # Build a simple ML model to predict review rating
    if not SKLEARN_READY:
        print('scikit-learn is not installed. Review prediction will use a simple fallback.')
        return None, None, None

    # Use processed review text as input and review rating as output
    model_df = df[['processed_review', 'review_rating']].copy()
    model_df['processed_review'] = model_df['processed_review'].fillna('').astype(str)
    model_df['review_rating'] = pd.to_numeric(model_df['review_rating'], errors='coerce')
    model_df = model_df.dropna()
    model_df = model_df[model_df['processed_review'].str.strip() != '']

    # Create lists for review text and rating labels
    joined_reviews = model_df['processed_review'].tolist()
    ratings = model_df['review_rating'].astype(int).tolist()

    # Convert review text into count vector features
    seed = 15
    vectorizer = CountVectorizer(analyzer='word', max_features=5000)
    count_features = vectorizer.fit_transform(joined_reviews)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        count_features,
        ratings,
        test_size=0.2,
        random_state=seed
    )

    # Train Logistic Regression model and check accuracy
    model = LogisticRegression(random_state=seed, max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print('Review rating model accuracy:', accuracy)

    # Re-train on all reviews so the web app uses the full dataset
    model.fit(count_features, ratings)
    return vectorizer, model, accuracy


# Train the review prediction model when the app starts
review_vectorizer, review_model, review_model_accuracy = train_review_model()


def fallback_predict_rating(review_text):
    # Backup prediction method if the ML package is not installed
    positive_words = ['good', 'great', 'best', 'love', 'amazing', 'perfect', 'nice', 'excellent']
    negative_words = ['bad', 'worst', 'hate', 'poor', 'dry', 'damage', 'useless', 'disappoint']

    text = review_text.lower()
    score = 3
    for word in positive_words:
        if word in text:
            score += 1
    for word in negative_words:
        if word in text:
            score -= 1

    return max(1, min(5, score))


def predict_review_rating(review_text):
    # Predict rating, sentiment, and confidence for a new review
    processed_review = clean_review_text(review_text)

    # Use backup prediction only if the ML model is not available
    if review_model is None or review_vectorizer is None or processed_review == '':
        predicted_rating = fallback_predict_rating(review_text)
        confidence = None
    else:
        # Transform the new review and predict its rating
        review_features = review_vectorizer.transform([processed_review])
        predicted_rating = int(review_model.predict(review_features)[0])
        prediction_probability = review_model.predict_proba(review_features)[0]
        confidence = round(float(max(prediction_probability)) * 100, 1)

    # Return prediction details to the HTML page
    return {
        'rating': predicted_rating,
        'sentiment': rating_to_sentiment(predicted_rating),
        'confidence': confidence,
        'processed_review': processed_review
    }

@app.route('/')
def home():
    # Render the homepage with the search bar
    return render_template('index.html')

@app.route('/search')
def search():
    # Get the search query from the URL and clean it
    query = request.args.get('query', '').strip()

    if not query:
        return render_template('index.html')

    # Lowercase the query for case-insensitive matching
    query_lower = query.lower()
    keywords = query_lower.split()

    # Combine brand and product title into one searchable string per row
    combined = (
        df['brand_name'].str.lower().fillna('').str.replace("'", '') + ' ' +
        df['product_title'].str.lower().fillna('').str.replace("'", '')
    )

    # Clean the query too for apostrophe handling
    keywords = [k.replace("'", '') for k in keywords]

    # All keywords must match somewhere in the combined string
    mask = pd.Series([True] * len(df), index=df.index)
    for keyword in keywords:
        mask = mask & combined.str.contains(keyword, na=False)

    df_filtered = df[mask]

    # Get unique products by product_title, show up to 20
    results = df_filtered.drop_duplicates(subset=['product_title']).head(20)
    count = len(results)

    # Assign a consistent image to each product based on review_id
    # Modulo ensures the index stays within the number of available images
    results_with_images = results.to_dict('records')
    for r in results_with_images:
        r['image'] = f"img{(r['review_id'] % NUM_IMAGES) + 1}.jpg"

    return render_template('results.html',
                           query=query,
                           results=results_with_images,
                           count=count)

@app.route('/product/<int:review_id>', methods=['GET', 'POST'])
def product_detail(review_id):
    # Find the product by review_id
    product = df[df['review_id'] == review_id]

    if product.empty:
        return "Product not found", 404

    # Get product info from first row
    product_info = product.iloc[0]
    review_result = None

    # Handle the review form submission
    if request.method == 'POST':
        # Get review details entered by the user
        review_title = request.form.get('review_title', '').strip()
        review_text = request.form.get('review_text', '').strip()
        is_a_buyer = request.form.get('is_a_buyer') == 'on'

        # Create a review only when the review text is not empty
        if review_text:
            review_result = predict_review_rating(review_text)
            new_review = {
                'review_title': review_title if review_title else 'New customer review',
                'review_text': review_text,
                'review_rating': review_result['rating'],
                'is_a_buyer': is_a_buyer,
                'product_title': product_info['product_title'],
                'predicted_sentiment': review_result['sentiment'],
                'prediction_confidence': review_result['confidence'],
                'created_by_user': True
            }
            # Add new review to the top of the review list
            new_reviews.insert(0, new_review)

    # Get all reviews for this same product title
    all_reviews = df[df['product_title'] == product_info['product_title']]
    # Find new reviews submitted for this product in this app session
    product_new_reviews = [
        review for review in new_reviews
        if review['product_title'] == product_info['product_title']
    ]
    # Show new submitted reviews before existing reviews
    review_records = product_new_reviews + all_reviews.to_dict('records')

    # Assign a consistent image based on review_id
    product_dict = product_info.to_dict()
    product_dict['image'] = f"img{(int(product_dict['review_id']) % NUM_IMAGES) + 1}.jpg"

    return render_template('product.html',
                           product=product_dict,
                           reviews=review_records,
                           # Send prediction result to the product page
                           review_result=review_result,
                           model_accuracy=review_model_accuracy)

if __name__ == '__main__':
    app.run(debug=True)
