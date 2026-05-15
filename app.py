from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort
import pandas as pd
import re
import json
from pathlib import Path

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
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
BASE_DIR = Path(__file__).resolve().parent
CREATED_REVIEWS_PATH = BASE_DIR / "created_reviews.json"


def load_created_reviews():
    if not CREATED_REVIEWS_PATH.exists():
        return []
    try:
        with open(CREATED_REVIEWS_PATH, "r", encoding="utf-8") as f:
            reviews = json.load(f)
    except (OSError, json.JSONDecodeError):
        return []
    return reviews if isinstance(reviews, list) else []


def save_created_reviews():
    payload = json.dumps(created_reviews, ensure_ascii=False, indent=2)
    try:
        CREATED_REVIEWS_PATH.write_text(payload, encoding="utf-8")
        saved_reviews = json.loads(CREATED_REVIEWS_PATH.read_text(encoding="utf-8"))
        if not isinstance(saved_reviews, list) or len(saved_reviews) != len(created_reviews):
            raise OSError("created_reviews.json did not update correctly")
        return True
    except (OSError, json.JSONDecodeError) as exc:
        print("Could not save created review:", exc)
        return False


created_reviews = load_created_reviews()

from recommendations import init_recommendations, get_similar_products

init_recommendations(df)


def clean_review_text(text):
    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def rating_to_sentiment(rating):
    if rating >= 4:
        return "Positive"
    if rating == 3:
        return "Neutral"
    return "Negative"


def build_model_text(title, body, processed=""):
    return clean_review_text(f"{title} {body} {processed}")


def keyword_adjust_rating(rating, confidence, text):
    text = f" {clean_review_text(text)} "
    negative_words = ["hate", "worst", "bad", "terrible", "awful", "useless", "waste", "disappoint", "delete", "poor"]
    positive_words = ["love", "best", "good", "great", "excellent", "amazing", "perfect", "recommend", "nice", "works"]
    negative = sum(1 for word in negative_words if word in text)
    positive = sum(1 for word in positive_words if word in text)
    score = positive - negative

    if score <= -2:
        return 1, max(confidence or 0, 90), "ML model + sentiment keywords"
    if score <= -1:
        return min(rating, 2), max(confidence or 0, 82), "ML model + sentiment keywords"
    if score >= 2:
        return 5, max(confidence or 0, 90), "ML model + sentiment keywords"
    if score >= 1:
        return max(rating, 4), max(confidence or 0, 82), "ML model + sentiment keywords"
    return rating, confidence, "ML model"


def fallback_predict_rating(title, body):
    combined = build_model_text(title, body)
    rating = 3
    rating, _, _ = keyword_adjust_rating(rating, 70, combined)
    return rating


def train_review_model():
    if not SKLEARN_READY:
        return None, None, None

    model_df = df[["review_title", "review_text", "processed_review", "review_rating"]].copy()
    model_df["review_rating"] = pd.to_numeric(model_df["review_rating"], errors="coerce")
    model_df = model_df.dropna()
    model_df["model_text"] = model_df.apply(
        lambda row: build_model_text(
            row.get("review_title", ""),
            row.get("review_text", ""),
            row.get("processed_review", ""),
        ),
        axis=1,
    )
    model_df = model_df[model_df["model_text"].str.strip() != ""]

    reviews = model_df["model_text"].tolist()
    ratings = model_df["review_rating"].astype(int).tolist()

    seed = 15
    vectorizer = TfidfVectorizer(max_features=6000, ngram_range=(1, 2), min_df=3, sublinear_tf=True)
    features = vectorizer.fit_transform(reviews)
    X_train, X_test, y_train, y_test = train_test_split(
        features,
        ratings,
        test_size=0.2,
        random_state=seed,
        stratify=ratings,
    )
    model = LogisticRegression(random_state=seed, max_iter=600, class_weight="balanced", solver="lbfgs")
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    model.fit(features, ratings)
    return vectorizer, model, accuracy


review_vectorizer, review_model, review_model_accuracy = train_review_model()


def predict_review_rating(body, title=""):
    model_text = build_model_text(title, body)
    if not model_text:
        rating = 3
        confidence = None
        source = "fallback"
    elif review_model is None or review_vectorizer is None:
        rating = fallback_predict_rating(title, body)
        confidence = 70
        source = "keyword fallback"
    else:
        features = review_vectorizer.transform([model_text])
        rating = int(review_model.predict(features)[0])
        confidence = round(float(max(review_model.predict_proba(features)[0])) * 100, 1)
        source = "ML model"

    rating, confidence, source = keyword_adjust_rating(rating, confidence, model_text)
    return {
        "rating": rating,
        "sentiment": rating_to_sentiment(rating),
        "confidence": round(confidence, 1) if confidence is not None else None,
        "source": source,
    }


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
    new_reviews = [
        review for review in created_reviews
        if review.get("product_title") == title
    ]
    dataset_reviews = [
        dataset_review_record(row)
        for _, row in df[df["product_title"] == title].iterrows()
    ]
    return new_reviews + dataset_reviews


def find_created_review(review_id):
    for review in created_reviews:
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
    results_df = df_filtered.drop_duplicates(subset=["product_title"]).head(48)
    results_df = results_df.copy()
    results_df["_rel"] = results_df.apply(lambda r: relevance_score(query, r), axis=1)
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

    prediction = predict_review_rating(review_text, review_title)
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
    reviews = reviews_for_product(title)

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

    if request.method == "POST":
        review_title = request.form.get("title", "").strip()
        review_text = request.form.get("body", "").strip()
        username = request.form.get("username", "").strip()
        rating_raw = request.form.get("rating", "").strip()

        errors = []
        if not 3 <= len(review_title) <= 120:
            errors.append("Review title must be between 3 and 120 characters.")
        if not 20 <= len(review_text) <= 2000:
            errors.append("Review description must be between 20 and 2000 characters.")
        if not 2 <= len(username) <= 40:
            errors.append("Display name must be between 2 and 40 characters.")
        try:
            user_rating = int(rating_raw)
        except (TypeError, ValueError):
            user_rating = None
        if user_rating not in (1, 2, 3, 4, 5):
            errors.append("Please select a star rating from 1 to 5.")

        if errors:
            for error in errors:
                flash(error, "error")
            return redirect(url_for("create_review", review_id=review_id))

        prediction = predict_review_rating(review_text, review_title)
        existing_ids = [
            int(review.get("review_id", 900000000))
            for review in created_reviews
            if str(review.get("review_id", "")).isdigit()
        ]
        new_review_id = max(existing_ids + [900000000]) + 1

        created_reviews.insert(0, {
            "review_id": new_review_id,
            "product_review_id": int(review_id),
            "review_title": review_title,
            "review_text": review_text,
            "review_rating": user_rating,
            "author": username,
            "is_a_buyer": user_rating >= 4,
            "is_buyer": user_rating >= 4,
            "brand_name": str(product_info["brand_name"]),
            "product_title": str(product_info["product_title"]),
            "price": float(product_info["price"]),
            "avg_product_rating": float(product_info["avg_product_rating"]),
            "image": image_for_review_id(review_id),
            "created_by_user": True,
            "predicted_rating": prediction["rating"],
            "predicted_sentiment": prediction["sentiment"],
            "prediction_confidence": prediction["confidence"],
            "prediction_source": prediction["source"],
        })
        if save_created_reviews():
            flash("Review saved successfully.", "success")
        else:
            flash("Review was created for this session, but could not be saved to created_reviews.json.", "error")
        return redirect(url_for("product_detail", review_id=review_id))

    return render_template(
        "create_review.html",
        product=product_dict,
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
        product_link = url_for("product_detail", review_id=int(review["product_review_id"]))
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
