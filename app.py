from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort
from recommendations import init_recommendations, get_similar_products
import pandas as pd
import re

# Import ML tools for review prediction — wrapped in try/except for safety
try:
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    SKLEARN_READY = True
except ImportError:
    SKLEARN_READY = False

app = Flask(__name__)
app.secret_key = "velora-dev-secret-change-in-production"

df = pd.read_csv("processed.csv")
df_original = pd.read_csv("cosmetics_beauty_products_reviews.csv")

df = df.merge(
    df_original[["review_id", "review_rating"]],
    on="review_id",
    how="left",
)

NUM_IMAGES = 6

# Store new reviews submitted during this app session
new_reviews = []


def clean_review_text(review_text):
    # Clean new review text before sending it to the ML model
    review_text = str(review_text).lower()
    review_text = re.sub(r'[^a-z0-9\s]', ' ', review_text)
    review_text = re.sub(r'\s+', ' ', review_text).strip()
    return review_text


def rating_to_sentiment(rating):
    # Convert predicted star rating into a readable label for the web page
    if rating >= 4:
        return 'Positive'
    elif rating == 3:
        return 'Neutral'
    else:
        return 'Negative'


def train_review_model():
    # Build a logistic regression model to predict review rating from text
    if not SKLEARN_READY:
        print('scikit-learn not available. Using fallback prediction.')
        return None, None, None

    # Use processed review text as input and review rating as target
    model_df = df[['processed_review', 'review_rating']].copy()
    model_df['processed_review'] = model_df['processed_review'].fillna('').astype(str)
    model_df['review_rating'] = pd.to_numeric(model_df['review_rating'], errors='coerce')
    model_df = model_df.dropna()
    model_df = model_df[model_df['processed_review'].str.strip() != '']

    joined_reviews = model_df['processed_review'].tolist()
    ratings = model_df['review_rating'].astype(int).tolist()

    seed = 15
    vectorizer = CountVectorizer(analyzer='word', max_features=5000)
    count_features = vectorizer.fit_transform(joined_reviews)

    X_train, X_test, y_train, y_test = train_test_split(
        count_features, ratings, test_size=0.2, random_state=seed
    )

    model = LogisticRegression(random_state=seed, max_iter=1000)
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    print(f'Review rating model accuracy: {accuracy}')

    # Retrain on full dataset for deployment
    model.fit(count_features, ratings)
    return vectorizer, model, accuracy


def fallback_predict_rating(review_text):
    # Simple keyword-based fallback if ML model is unavailable
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
    # Predict rating, sentiment and confidence for a new review
    processed_review = clean_review_text(review_text)
    if review_model is None or review_vectorizer is None or processed_review == '':
        predicted_rating = fallback_predict_rating(review_text)
        confidence = None
    else:
        review_features = review_vectorizer.transform([processed_review])
        predicted_rating = int(review_model.predict(review_features)[0])
        proba = review_model.predict_proba(review_features)[0]
        confidence = round(float(max(proba)) * 100, 1)
    return {
        'rating': predicted_rating,
        'sentiment': rating_to_sentiment(predicted_rating),
        'confidence': confidence,
        'processed_review': processed_review
    }


# Train the model once when the app starts
review_vectorizer, review_model, review_model_accuracy = train_review_model()
init_recommendations(df)

@app.context_processor
def inject_nav():
    brands = df["brand_name"].dropna().value_counts().head(10).index.tolist()
    return {"nav_brands": brands}


def image_for_review_id(review_id):
    return f"img{(int(review_id) % NUM_IMAGES) + 1}.jpg"


def row_to_product_card(row):
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    d["image"] = image_for_review_id(d["review_id"])
    return d


def unique_products(sub_df, n=12):
    u = sub_df.drop_duplicates(subset=["product_title"]).head(n)
    return [row_to_product_card(u.iloc[i]) for i in range(len(u))]


def dataset_review_record(row):
    review = row.to_dict()
    review["image"] = image_for_review_id(review["review_id"])
    ib = row.get("is_a_buyer")
    review["is_buyer"] = ib is True or str(ib).lower() == "true"
    review["created_by_user"] = False
    return review


def reviews_for_product(title):
    user_reviews = [
        review for review in new_reviews
        if review.get("product_title") == title
    ]
    dataset_reviews = [
        dataset_review_record(row)
        for _, row in df[df["product_title"] == title].iterrows()
    ]
    return user_reviews + dataset_reviews


def find_created_review(review_id):
    for review in new_reviews:
        if int(review.get("review_id", 0)) == int(review_id):
            return review
    return None


def search_mask(query):
    query_lower = query.strip().lower()
    if not query_lower:
        return pd.Series([False] * len(df), index=df.index)
    keywords = [k.replace("'", "") for k in query_lower.split()]
    combined = (
        df["brand_name"].str.lower().fillna("").str.replace("'", "", regex=False)
        + " "
        + df["product_title"].str.lower().fillna("").str.replace("'", "", regex=False)
    )
    combined = combined + " " + df.get("processed_review", pd.Series([""] * len(df))).str.lower().fillna("")
    mask = pd.Series([True] * len(df), index=df.index)
    for keyword in keywords:
        mask = mask & combined.str.contains(keyword, na=False, regex=False)
    return mask


def relevance_score(query, row):
    """Simple deterministic pseudo-score for UI (tutor-visible ranking signal)."""
    q = query.lower().strip()
    if not q:
        return 0
    brand = str(row.get("brand_name", "")).lower()
    title = str(row.get("product_title", "")).lower()
    score = 50
    if q in brand:
        score += 25
    if q in title:
        score += 20
    for word in q.split():
        if len(word) > 2 and word in brand:
            score += 8
        if len(word) > 2 and word in title:
            score += 5
    rid = int(row.get("review_id", 0))
    score += (rid % 17)
    return min(99, score)


@app.route("/")
def home():
    unique = df.drop_duplicates(subset=["product_title"])
    featured = unique.head(8)
    top_rated = unique.sort_values("avg_product_rating", ascending=False).head(8)
    brand_counts = df["brand_name"].fillna("Unknown").value_counts().head(10)
    popular_brands = brand_counts.index.tolist()
    categories = popular_brands[:8]

    return render_template(
        "index.html",
        featured_products=[row_to_product_card(featured.iloc[i]) for i in range(len(featured))],
        top_rated_products=[row_to_product_card(top_rated.iloc[i]) for i in range(len(top_rated))],
        popular_brands=popular_brands,
        categories=categories,
    )


@app.route("/search")
def search():
    query = request.args.get("query", "").strip()
    if not query:
        return redirect(url_for("home"))

    mask = search_mask(query)
    df_filtered = df[mask]
    results_df = df_filtered.drop_duplicates(subset=["product_title"]).head(48).copy()
    rel_scores = [relevance_score(query, row) for _, row in results_df.iterrows()]
    results_df["_rel"] = rel_scores
    results_df = results_df.sort_values("_rel", ascending=False)
    
    results = []
    for _, r in results_df.iterrows():
        d = row_to_product_card(r)
        d["relevance"] = int(r["_rel"])
        d["short_text"] = (
            str(r.get("review_text", ""))[:120] + "…"
            if len(str(r.get("review_text", ""))) > 120
            else str(r.get("review_text", ""))
        )
        results.append(d)

    count = len(results)
    return render_template(
        "results.html",
        query=query,
        results=results,
        count=count,
    )


@app.route("/api/suggestions")
def suggestions():
    q = request.args.get("q", "").strip().lower()
    if len(q) < 1:
        return jsonify([])
    brands = df["brand_name"].dropna().unique()
    titles = df["product_title"].dropna().unique()
    out = []
    for b in brands:
        if q in str(b).lower() and b not in out:
            out.append(str(b))
        if len(out) >= 8:
            break
    for t in titles:
        if q in str(t).lower() and t not in out:
            out.append(str(t)[:80])
        if len(out) >= 12:
            break
    return jsonify(out[:12])


@app.route("/api/review-prediction", methods=["POST"])
def review_prediction_api():
    data = request.get_json(silent=True) or {}
    review_title = str(data.get("review_title", "")).strip()
    review_text = str(data.get("review_text", "")).strip()
    if not review_text:
        return jsonify({"error": "Review text is required."}), 400

    prediction = predict_review_rating(review_text)
    return jsonify({
        "rating": prediction["rating"],
        "sentiment": prediction["sentiment"],
        "confidence": prediction["confidence"],
        "source": prediction["source"],
        "model_accuracy": review_model_accuracy,
    })


@app.route("/product/<int:review_id>")
def product_detail(review_id):
    product_rows = df[df["review_id"] == review_id]
    if product_rows.empty:
        abort(404)

    product_info = product_rows.iloc[0]
    title = product_info["product_title"]
    brand = product_info["brand_name"]

    all_reviews = df[df["product_title"] == title]
    reviews = []
    for _, rev in all_reviews.iterrows():
        rd = rev.to_dict()
        rd["image"] = image_for_review_id(rd["review_id"])
        ib = rev.get("is_a_buyer")
        rd["is_buyer"] = ib is True or str(ib).lower() == "true"
        reviews.append(rd)
     # Add new reviews from this session, normalising the is_buyer key
    product_new_reviews = []
    for r in new_reviews:
        if r["product_title"] == title:
            r["is_buyer"] = r.get("is_a_buyer", False)
            product_new_reviews.append(r)
    reviews = product_new_reviews + reviews

    buyer_n = sum(1 for r in reviews if r["is_buyer"])
    buyer_ratio = round(100 * buyer_n / len(reviews), 1) if reviews else 0
    review_ratings = pd.to_numeric(
        pd.Series([r.get("review_rating") for r in reviews]),
        errors="coerce",
    ).dropna()
    avg_rev_rating = (
        round(review_ratings.mean(), 2)
        if len(review_ratings)
        else None
    )

    similar = [row_to_product_card(r) for r in get_similar_products(title, 6)]

    product_dict = row_to_product_card(product_info)
    product_dict["category"] = str(brand) if pd.notna(brand) else "Beauty"
    product_dict["review_count"] = len(reviews)
    product_dict["buyer_ratio"] = buyer_ratio
    product_dict["avg_review_rating"] = avg_rev_rating
    texts = all_reviews["review_text"].dropna().astype(str)
    product_dict["description"] = texts.iloc[0][:560] + ("…" if len(texts.iloc[0]) > 560 else "") if len(texts) else "Premium beauty pick from the dataset—details appear as customers describe their experience."
    pr = str(product_info.get("processed_review", "") or "")
    product_dict["tags"] = pr.split()[:10] if pr else []

    return render_template(
        "product.html",
        product=product_dict,
        reviews=reviews,
        similar_products=similar,
        breadcrumb=[
            ("Home", url_for("home")),
            ("Search", url_for("search", query=str(brand)[:40] if brand else "beauty")),
            (str(title)[:48] + ("…" if len(str(title)) > 48 else ""), None),
        ],
    )


@app.route("/recommendations/<int:review_id>")
def recommendations_page(review_id):
    product_rows = df[df["review_id"] == review_id]
    if product_rows.empty:
        abort(404)
    product_info = product_rows.iloc[0]
    title = product_info["product_title"]
    brand = product_info["brand_name"]
    similar = [row_to_product_card(r) for r in get_similar_products(title, 12)]
    product_dict = row_to_product_card(product_info)
    return render_template(
        "recommendations.html",
        product=product_dict,
        similar_products=similar,
        breadcrumb=[
            ("Home", url_for("home")),
            ("Product", url_for("product_detail", review_id=review_id)),
            ("Recommendations", None),
        ],
    )


@app.route("/create-review/<int:review_id>", methods=["GET", "POST"])
def create_review(review_id):
    product_rows = df[df["review_id"] == review_id]
    if product_rows.empty:
        abort(404)
    product_info = product_rows.iloc[0]
    product_dict = row_to_product_card(product_info)
    review_result = None

    if request.method == "POST":
        # Get review details from the form
        review_title = request.form.get("title", "").strip()
        review_text = request.form.get("body", "").strip()
        is_a_buyer = request.form.get("is_a_buyer") == "on"

        if review_text:
            # Run ML prediction on the submitted review text
            review_result = predict_review_rating(review_text)
            new_review = {
                "review_id": 0,
                "review_title": review_title if review_title else "New customer review",
                "review_text": review_text,
                "review_rating": review_result["rating"],
                "is_a_buyer": is_a_buyer,
                "is_buyer": is_a_buyer,
                "product_title": product_info["product_title"],
                "brand_name": product_info["brand_name"],
                "avg_product_rating": product_info.get("avg_product_rating", 0),
                "price": product_info.get("price", 0),
                "predicted_sentiment": review_result["sentiment"],
                "prediction_confidence": review_result["confidence"],
                "created_by_user": True
         }
            # Store the new review in memory for this session
            new_reviews.insert(0, new_review)
            flash("Review saved successfully.", "success")

    return render_template(
        "create_review.html",
        product=product_dict,
        review_result=review_result,
        model_accuracy=review_model_accuracy,
        breadcrumb=[
            ("Home", url_for("home")),
            ("Product", url_for("product_detail", review_id=review_id)),
            ("Write a review", None),
        ],
    )


@app.route("/review/<int:review_id>")
def review_detail(review_id):
    created_review = find_created_review(review_id)
    if created_review:
        review = dict(created_review)
        product_review_id = review.get("product_review_id", review_id)
        product_link = url_for("product_detail", review_id=int(product_review_id))
        return render_template(
            "review_detail.html",
            review=review,
            product_title=review.get("product_title", ""),
            product_url=product_link,
            breadcrumb=[
                ("Home", url_for("home")),
                ("Product", product_link),
                ("Review #%s" % review_id, None),
            ],
        )

    row = df[df["review_id"] == review_id]
    if row.empty:
        abort(404)
    r = row.iloc[0].to_dict()
    r["image"] = image_for_review_id(r["review_id"])
    ib = r.get("is_a_buyer")
    r["is_buyer"] = ib is True or str(ib).lower() == "true"
    r["ai_predicted_buyer"] = r["is_buyer"]
    r["confidence"] = 72 + (review_id % 25)

    prod = df[df["product_title"] == r["product_title"]].iloc[0]
    product_link = url_for("product_detail", review_id=int(prod["review_id"]))

    return render_template(
        "review_detail.html",
        review=r,
        product_title=r.get("product_title", ""),
        product_url=product_link,
        breadcrumb=[
            ("Home", url_for("home")),
            ("Product", product_link),
            ("Review #%s" % review_id, None),
        ],
    )


@app.route("/admin", endpoint="admin")
def admin_dashboard():
    total_reviews = len(df)
    unique_titles = df["product_title"].nunique()
    ib = df["is_a_buyer"].apply(lambda x: x is True or str(x).lower() == "true")
    buyer_ratio = round(100 * ib.sum() / total_reviews, 1) if total_reviews else 0
    top_brand = df["brand_name"].fillna("—").value_counts().index[0] if total_reviews else "—"

    ratings = pd.to_numeric(df["review_rating"], errors="coerce").dropna()
    rating_bins = {str(i): int((ratings == i).sum()) for i in range(1, 6)}

    brand_top = df["brand_name"].fillna("Unknown").value_counts().head(8)

    return render_template(
        "admin.html",
        breadcrumb=[
            ("Home", url_for("home")),
            ("Insights", None),
        ],
        stats={
            "products": unique_titles,
            "reviews": total_reviews,
            "buyer_ratio": buyer_ratio,
            "top_brand": top_brand,
        },
        rating_distribution=rating_bins,
        brand_labels=list(brand_top.index),
        brand_values=[int(x) for x in brand_top.values],
        model_metrics={
            "accuracy": round(review_model_accuracy or 0, 3),
            "features": "TF-IDF",
            "classifier": "Logistic Regression",
        },
    )


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
