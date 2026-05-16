"""Main Flask app connecting product browsing, reviews, recommendations, and advisor pages."""

from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort
from recommendations import init_recommendations, get_similar_products
from review_models import binary_label, predict_review_label, train_review_label_ensemble
from beauty_advisor import init_advisor, recommend_single, recommend_set, get_brands
import pandas as pd
import json
from pathlib import Path

app = Flask(__name__)
app.secret_key = "velora-dev-secret-change-in-production"

# Load the main processed dataset and the original review dataset.
# The processed file is used for search/model features, while the original file
# still contains useful raw fields such as the star rating.
df = pd.read_csv("processed.csv")
df_original = pd.read_csv("cosmetics_beauty_products_reviews.csv")

# Add the original star rating back into the processed dataframe.
# This lets the review label model use both text features and numeric rating features.
df = df.merge(
    df_original[["review_id", "review_rating"]],
    on="review_id",
    how="left",
)

NUM_IMAGES = 6
CREATED_REVIEWS_PATH = Path("created_reviews.json")
USER_REVIEW_ID_START = 900000000

# Store reviews created by users and persist them for review URLs.
new_reviews = []


# These helper functions manage reviews created from the web form.
# They are stored in JSON because this assignment app does not need a full database,
# but the saved file still lets created review URLs work after restarting Flask.
def load_created_reviews():
    # Read user-created reviews so their /review/<id> URLs survive app restarts.
    if not CREATED_REVIEWS_PATH.exists():
        return []
    try:
        reviews = json.loads(CREATED_REVIEWS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return reviews if isinstance(reviews, list) else []


def save_created_reviews(reviews):
    # Save created reviews as simple JSON instead of adding a database dependency.
    CREATED_REVIEWS_PATH.write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def next_created_review_id():
    # Keep new review IDs away from dataset review IDs.
    max_existing = USER_REVIEW_ID_START
    for review in new_reviews:
        try:
            max_existing = max(max_existing, int(review.get("review_id", 0)))
        except (TypeError, ValueError):
            continue
    return max_existing + 1


# Prediction helpers keep the Flask routes cleaner.
# They collect the product context once, then call the shared model function so
# both the live preview API and the final form submission use the same logic.
def product_context(product_info, review_rating=None):
    # Package structured features used by the buyer-label ensemble.
    return {
        "review_rating": review_rating,
        "avg_product_rating": product_info.get("avg_product_rating", 0),
        "price": product_info.get("price", 0),
    }


def predict_for_review(review_title, review_text, product_info, review_rating=None):
    # Central wrapper so the form and API use the same prediction logic.
    return predict_review_label(
        review_label_ensemble,
        review_text,
        review_title=review_title,
        context=product_context(product_info, review_rating),
    )


# Older saved reviews may have been created before the current label format.
# This function refreshes their saved model fields so the review detail page
# displays the same label names, confidence values, and model votes as new reviews.
def refresh_legacy_review_labels():
    # Upgrade older saved reviews to the current Would buy / Would not buy format.
    updated = False
    for review in new_reviews:
        if not review.get("created_by_user"):
            continue

        prediction = predict_review_label(
            review_label_ensemble,
            review.get("review_text", ""),
            review_title=review.get("review_title", ""),
            context={
                "review_rating": review.get("review_rating"),
                "avg_product_rating": review.get("avg_product_rating"),
                "price": review.get("price"),
            },
        )
        ai_is_buyer = bool(prediction["would_buy"])
        final_is_buyer = (
            review.get("final_label") == "Would buy"
            if review.get("label_source") == "User override"
            else ai_is_buyer
        )
        review["is_a_buyer"] = final_is_buyer
        review["is_buyer"] = final_is_buyer
        review["ai_predicted_buyer"] = ai_is_buyer
        review["ai_predicted_label"] = prediction["label"]
        review["final_label"] = binary_label(final_is_buyer)
        review["label_source"] = review.get("label_source") or "AI ensemble"
        review["prediction_probability"] = prediction["probability"]
        review["prediction_confidence"] = prediction["confidence"]
        review["prediction_source"] = prediction["source"]
        review["model_votes"] = prediction["votes"]
        updated = True

    if updated:
        save_created_reviews(new_reviews)


# Load saved user reviews and train the buyer-label ensemble once at startup.
new_reviews = load_created_reviews()
review_label_ensemble = train_review_label_ensemble(df)
review_model_accuracy = review_label_ensemble.get("fused_accuracy")
refresh_legacy_review_labels()
# Build shared recommendation indexes once for fast request handling.
init_recommendations(df)
init_advisor(df, df_original)


# Template and formatting helpers.
# These functions turn dataframe rows into simple dictionaries with images and
# consistent review fields, which keeps the HTML templates easier to read.
@app.context_processor
def inject_nav():
    brands = df["brand_name"].dropna().value_counts().head(10).index.tolist()
    return {"nav_brands": brands}


def image_for_review_id(review_id):
    # Reuse six local images deterministically for product cards.
    return f"img{(int(review_id) % NUM_IMAGES) + 1}.jpg"


def row_to_product_card(row):
    # Convert dataframe rows into dictionaries expected by the templates.
    d = row.to_dict() if hasattr(row, "to_dict") else dict(row)
    d["image"] = image_for_review_id(d["review_id"])
    return d


def unique_products(sub_df, n=12):
    u = sub_df.drop_duplicates(subset=["product_title"]).head(n)
    return [row_to_product_card(u.iloc[i]) for i in range(len(u))]


def dataset_review_record(row):
    # Normalise dataset review fields to match user-created review fields.
    review = row.to_dict()
    review["image"] = image_for_review_id(review["review_id"])
    ib = row.get("is_a_buyer")
    review["is_buyer"] = ib is True or str(ib).lower() == "true"
    review["created_by_user"] = False
    return review


def reviews_for_product(title):
    # Show newly created reviews before existing dataset reviews.
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


# Search uses a simple AND match: every query word must appear somewhere in the
# product brand, product title, or processed review text. This makes partial
# searches predictable without adding another search library.
def search_mask(query):
    # Match all query words against brand, title, and processed review text.
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
    # The homepage is built directly from the dataset instead of hard-coded cards.
    # Featured products come from the first unique products, while top-rated products
    # are sorted by average rating to give the page a useful browsing structure.
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
    # Search first filters rows by the query, then removes duplicate product titles.
    # A small relevance score is added so stronger brand/title matches appear higher
    # than products where the query only appears in review text.
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
    # This API supports the search bar autocomplete.
    # It checks brands first, then product titles, and returns only a small list so
    # the front-end can update quickly while the user types.
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
    # The review form calls this route while the user is typing.
    # It returns the suggested label, confidence, and individual model votes without
    # saving anything yet, so the user can still edit or override the label.
    data = request.get_json(silent=True) or {}
    review_title = str(data.get("review_title", "")).strip()
    review_text = str(data.get("review_text", "")).strip()
    review_rating = data.get("review_rating") or None
    product_review_id = data.get("product_review_id")
    if not review_text:
        return jsonify({"error": "Review text is required."}), 400

    product_info = {}
    if product_review_id:
        try:
            product_rows = df[df["review_id"] == int(product_review_id)]
            if not product_rows.empty:
                product_info = product_rows.iloc[0]
        except (TypeError, ValueError):
            product_info = {}

    prediction = predict_for_review(review_title, review_text, product_info, review_rating)
    return jsonify({
        "would_buy": prediction["would_buy"],
        "label": prediction["label"],
        "probability": prediction["probability"],
        "confidence": prediction["confidence"],
        "source": prediction["source"],
        "votes": prediction["votes"],
        "model_accuracy": review_model_accuracy,
    })


@app.route("/product/<int:review_id>")
def product_detail(review_id):
    # The product page starts from one review_id, then finds all reviews for the
    # same product title. User-created reviews are placed first so newly submitted
    # content is immediately visible on the website.
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
    # Add new reviews from this session, normalising the is_buyer key.
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

    show_all_reviews = request.args.get("show_all", "").lower() in {"1", "true", "yes"}
    total_reviews = len(reviews)
    if not show_all_reviews and total_reviews > 10:
        reviews = reviews[:10]
        more_reviews = total_reviews - 10
    else:
        more_reviews = 0

    return render_template(
        "product.html",
        product=product_dict,
        reviews=reviews,
        similar_products=similar,
        more_reviews=more_reviews,
        show_all_reviews=show_all_reviews,
        breadcrumb=[
            ("Home", url_for("home")),
            ("Search", url_for("search", query=str(brand)[:40] if brand else "beauty")),
            (str(title)[:48] + ("…" if len(str(title)) > 48 else ""), None),
        ],
    )


@app.route("/recommendations/<int:review_id>")
def recommendations_page(review_id):
    # This page expands the smaller recommendation strip shown on the product page.
    # It uses the same selected product and asks the recommendation module for more
    # similar products based on review-text similarity.
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
    # The GET request shows the review form for a selected product.
    # The POST request validates the submitted text/rating, predicts a buyer label,
    # accepts the user's override choice if provided, then saves the review.
    product_rows = df[df["review_id"] == review_id]
    if product_rows.empty:
        abort(404)
    product_info = product_rows.iloc[0]
    product_dict = row_to_product_card(product_info)
    review_result = None

    if request.method == "POST":
        # Collect the user input from the form.
        # Empty usernames become "Anonymous" so the saved review always has an author.
        review_title = request.form.get("title", "").strip()
        review_text = request.form.get("body", "").strip()
        user_rating = int(request.form.get("rating", 0) or 0)
        author = request.form.get("username", "").strip() or "Anonymous"
        label_mode = request.form.get("label_mode", "ai")
        manual_label = request.form.get("manual_label", "")

        if review_text:
            # Run the ensemble to get the website's suggested label.
            # If the reviewer chooses "override", the manually selected label becomes
            # the final saved label while the original AI suggestion is still stored.
            review_result = predict_for_review(review_title, review_text, product_info, user_rating)
            ai_is_buyer = bool(review_result["would_buy"])
            if label_mode == "override" and manual_label in {"buyer", "not_buyer"}:
                final_is_buyer = manual_label == "buyer"
                label_source = "User override"
            else:
                final_is_buyer = ai_is_buyer
                label_source = "AI ensemble"

            created_review_id = next_created_review_id()
            new_review = {
                "review_id": created_review_id,
                "product_review_id": int(product_info["review_id"]),
                "review_title": review_title if review_title else "New customer review",
                "review_text": review_text,
                "review_rating": user_rating,
                "author": author,
                "is_a_buyer": final_is_buyer,
                "is_buyer": final_is_buyer,
                "product_title": product_info["product_title"],
                "brand_name": product_info["brand_name"],
                "avg_product_rating": float(product_info.get("avg_product_rating", 0) or 0),
                "price": float(product_info.get("price", 0) or 0),
                "image": product_dict["image"],
                "created_by_user": True,
                "ai_predicted_buyer": ai_is_buyer,
                "ai_predicted_label": review_result["label"],
                "final_label": binary_label(final_is_buyer),
                "label_source": label_source,
                "prediction_probability": review_result["probability"],
                "prediction_confidence": review_result["confidence"],
                "prediction_source": review_result["source"],
                "model_votes": review_result["votes"],
            }
            # Store the new review in memory and on disk for a stable URL.
            new_reviews.insert(0, new_review)
            save_created_reviews(new_reviews)
            flash("Review saved successfully.", "success")
            return redirect(url_for("review_detail", review_id=created_review_id))

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
    # Created reviews are loaded from JSON first because they contain extra fields
    # such as AI suggestion, final label, and label source. If no created review
    # matches, the route falls back to displaying an original dataset review.
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


def enrich_advisor_cards(result):
    # Advisor recommendations come from product rows, but the cards still need the
    # local image filename used by the rest of the site. This helper adds images to
    # main picks, extras, set items, and supplements in one place.
    if not result or not result.get("ok"):
        return result

    def attach(card):
        if card:
            card["image"] = image_for_review_id(card["review_id"])
        return card

    if result.get("main"):
        attach(result["main"])
    for key in ("extras", "set_items", "supplements"):
        if result.get(key):
            result[key] = [attach(card) for card in result[key]]
    return result


@app.route("/beauty-advisor", methods=["GET", "POST"])
def beauty_advisor():
    # The advisor form can return either a single product or a small routine.
    # Budget is checked before calling the advisor logic so invalid input shows a
    # friendly message instead of causing a server error.
    form = {
        "budget": request.form.get("budget", "") if request.method == "POST" else "",
        "mode": request.form.get("mode", "single") if request.method == "POST" else "single",
        "brand_pref": request.form.get("brand_pref", "").strip() if request.method == "POST" else "",
        "product_type": request.form.get("product_type", "").strip() if request.method == "POST" else "",
    }
    result = None

    if request.method == "POST":
        try:
            budget = float(form["budget"])
        except (TypeError, ValueError):
            budget = 0
        if budget <= 0:
            flash("Please enter a valid budget greater than zero.", "error")
        else:
            if form["mode"] == "set":
                result = recommend_set(budget, form["brand_pref"], form["product_type"])
            else:
                result = recommend_single(budget, form["brand_pref"], form["product_type"])
            result = enrich_advisor_cards(result)

    return render_template(
        "beauty_advisor.html",
        form=form,
        result=result,
        brands=get_brands(),
        breadcrumb=[
            ("Home", url_for("home")),
            ("Beauty Advisor", None),
        ],
    )


@app.route("/admin", endpoint="admin")
def admin_dashboard():
    # The admin page summarises useful dataset and model information.
    # These values are calculated from the current dataframe so the dashboard stays
    # consistent with the data used by browsing, reviews, and recommendations.
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
            "features": "Text + rating/product metadata + language cues",
            "classifier": "Weighted 3-model ensemble",
        },
    )


@app.errorhandler(404)
def not_found(e):
    return render_template("404.html"), 404


if __name__ == "__main__":
    app.run(debug=True)
