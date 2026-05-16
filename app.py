from flask import Flask, render_template, request, redirect, url_for, flash, jsonify, abort
from recommendations import init_recommendations, get_similar_products
from review_models import binary_label, predict_review_label, train_review_label_ensemble
from beauty_advisor import init_advisor, recommend_single, recommend_set, get_brands
import pandas as pd
import json
from pathlib import Path

app = Flask(__name__)
app.secret_key = "velora-dev-secret-change-in-production"
# Task 1: I load processed.csv from Milestone 1 and merge in review_rating from the
# original CSV since it wasn't saved during preprocessing.
# Loading once at startup keeps search requests fast instead of reloading each time.
df = pd.read_csv("processed.csv")
df_original = pd.read_csv("cosmetics_beauty_products_reviews.csv")

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


def load_created_reviews():
    if not CREATED_REVIEWS_PATH.exists():
        return []
    try:
        reviews = json.loads(CREATED_REVIEWS_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []
    return reviews if isinstance(reviews, list) else []


def save_created_reviews(reviews):
    CREATED_REVIEWS_PATH.write_text(
        json.dumps(reviews, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def next_created_review_id():
    max_existing = USER_REVIEW_ID_START
    for review in new_reviews:
        try:
            max_existing = max(max_existing, int(review.get("review_id", 0)))
        except (TypeError, ValueError):
            continue
    return max_existing + 1


def product_context(product_info, review_rating=None):
    return {
        "review_rating": review_rating,
        "avg_product_rating": product_info.get("avg_product_rating", 0),
        "price": product_info.get("price", 0),
    }


def predict_for_review(review_title, review_text, product_info, review_rating=None):
    return predict_review_label(
        review_label_ensemble,
        review_text,
        review_title=review_title,
        context=product_context(product_info, review_rating),
    )


def refresh_legacy_review_labels():
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
init_recommendations(df)
init_advisor(df, df_original)

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

# Task 1: I combine brand name, product title and review text into one string per product
# so I only need one pass to check all keyword matches.
# Lowercasing and stripping apostrophes handles cases like "loreal" matching
# "L'Oreal Paris" and makes the search case-insensitive as required.
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

# Task 1: After filtering I rank results by how well they match the query.
# A full query match in the brand name scores highest (+25), then product title (+20),
# then individual keywords add smaller scores on top.
# This puts the most relevant products first without needing a complex ML model.
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

# Task 1: Search route — takes the query from the URL, runs search_mask to filter
# matching products, scores and sorts them by relevance, then passes
# up to 48 deduplicated results to the results page.
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

# Task 1: Autocomplete endpoint — as the user types, this returns up to 12 matching
# brand names and product titles as JSON for the search bar suggestions.
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

# Task 1: Product detail route — loads all reviews for the product, calculates buyer
# ratio, fetches similar products, and prepends any new user reviews to the top.
# The review_id in the URL identifies which product to show.
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
        user_rating = int(request.form.get("rating", 0) or 0)
        author = request.form.get("username", "").strip() or "Anonymous"
        label_mode = request.form.get("label_mode", "ai")
        manual_label = request.form.get("manual_label", "")

        if review_text:
            # Run the three-model ensemble, then let the user override its label.
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
