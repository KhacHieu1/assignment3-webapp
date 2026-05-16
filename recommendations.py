"""Content-based product recommendations using TF-IDF and cosine similarity."""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

MAX_DOC_CHARS = 8000

# These module-level variables cache the product catalog and TF-IDF matrix.
# Flask builds them once at startup, then every recommendation request can reuse
# the same vectors instead of recomputing them.
_catalog = None
_matrix = None


def build_product_catalog(df):
    # Collapse many reviews per product into one searchable document.
    # Each product can have many review rows, but recommendations should compare
    # products, not individual reviews, so all review text is joined by title.
    text_col = "processed_review" if "processed_review" in df.columns else "review_text"
    records = []

    for title, group in df.groupby("product_title", sort=False):
        texts = group[text_col].fillna("").astype(str)
        doc = " ".join(t for t in texts if t.strip())
        # Limit very long documents so the vectorizer stays fast and memory-friendly.
        if len(doc) > MAX_DOC_CHARS:
            doc = doc[:MAX_DOC_CHARS]

        row = group.sort_values("review_id").iloc[0]
        records.append(
            {
                "product_title": title,
                "review_id": row["review_id"],
                "brand_name": row.get("brand_name"),
                "price": row.get("price"),
                "avg_product_rating": row.get("avg_product_rating"),
                "doc_text": doc.strip(),
            }
        )

    return pd.DataFrame(records).reset_index(drop=True)


def build_similarity_index(catalog):
    # Convert product review documents into TF-IDF vectors for similarity search.
    # TF-IDF gives more weight to words that describe a product well, while cosine
    # similarity later compares the direction of those word-weight vectors.
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)
    matrix = vectorizer.fit_transform(catalog["doc_text"].fillna(""))
    return vectorizer, matrix


def init_recommendations(df):
    # Build the recommendation index once when Flask starts.
    # Returning the catalog/matrix is useful for debugging, but the important part
    # is storing them in the module cache for get_similar_products().
    global _catalog, _matrix
    _catalog = build_product_catalog(df)
    if _catalog.empty:
        _matrix = None
        return _catalog, None
    _, _matrix = build_similarity_index(_catalog)
    return _catalog, _matrix


def get_similar_products(product_title, k=12):
    # Rank products by cosine similarity to the selected product.
    # The selected product is removed from its own results, then the highest scoring
    # different products are returned for the product page and recommendation page.
    if _catalog is None or _matrix is None or _catalog.empty:
        return []

    matches = _catalog.index[_catalog["product_title"] == product_title]
    if len(matches) == 0:
        return []

    idx = int(matches[0])
    sims = cosine_similarity(_matrix[idx], _matrix).ravel()
    sims[idx] = -1.0

    order = np.argsort(-sims)
    out = []
    for i in order:
        if sims[i] <= 0:
            continue
        out.append(_catalog.iloc[int(i)])
        if len(out) >= k:
            break
    return out
