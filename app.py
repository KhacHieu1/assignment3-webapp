from flask import Flask, render_template, request
import pandas as pd

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
    # This satisfies the requirement that "Maybeline" and "maybeline" return same results
    query_lower = query.lower()
    keywords = query_lower.split()

    # Combine brand and product title into one searchable string per row
    # Replacing apostrophes so "loreal" matches "L'Oreal Paris"
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

@app.route('/product/<int:review_id>')
def product_detail(review_id):
    # Find the product by review_id
    product = df[df['review_id'] == review_id]

    if product.empty:
        return "Product not found", 404

    # Get product info from first row
    product_info = product.iloc[0]

    # Get all reviews for this same product title
    all_reviews = df[df['product_title'] == product_info['product_title']]

    # Assign a consistent image based on review_id
    product_dict = product_info.to_dict()
    product_dict['image'] = f"img{(int(product_dict['review_id']) % NUM_IMAGES) + 1}.jpg"

    return render_template('product.html',
                           product=product_dict,
                           reviews=all_reviews.to_dict('records'))

if __name__ == '__main__':
    app.run(debug=True)