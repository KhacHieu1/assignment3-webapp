"""Buyer-label models for predicting Would buy / Would not buy."""

import re

import pandas as pd

# Scikit-learn is used for the real prediction models.
# The try/except keeps the Flask app usable even if the package is missing on
# another computer; in that case the code falls back to a simple keyword score.
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    SKLEARN_READY = True
except ImportError:
    SKLEARN_READY = False


# These word lists are used by the language-cue model and the fallback predictor.
# They give the model direct signals for obvious review language such as "love",
# "hate", "recommend", or "waste", which helps when the review text is very short.
POSITIVE_WORDS = {
    "amazing", "best", "bright", "buy", "effective", "excellent", "fresh",
    "gentle", "glow", "good", "great", "happy", "hydrating", "impressed",
    "liked", "love", "loved", "nice", "perfect", "recommend", "repurchase",
    "smooth", "soft", "worth",
}
NEGATIVE_WORDS = {
    "allergy", "bad", "breakout", "burn", "burning", "cheap", "damage",
    "disappoint", "dry", "dull", "hate", "irritation", "itchy", "oily",
    "poor", "sticky", "useless", "waste", "worst",
}
BUY_INTENT_WORDS = {"buy", "buyer", "purchase", "repurchase", "recommend", "worth"}
NOT_BUY_WORDS = {"avoid", "never", "refund", "return", "waste", "worthless"}


def clean_review_text(review_text):
    # Normalise free-text reviews before vectorising or counting cue words.
    # Lowercasing, punctuation removal, and repeated-letter cleanup make user input
    # like "I HATEEEE it!!!" easier to compare with the training data.
    review_text = str(review_text or "").lower()
    review_text = re.sub(r"[^a-z0-9\s]", " ", review_text)
    review_text = re.sub(r"([a-z])\1{2,}", r"\1", review_text)
    review_text = re.sub(r"\s+", " ", review_text).strip()
    return review_text


def binary_label(value):
    # Convert a boolean model result into the binary label shown in the interface.
    return "Would buy" if bool(value) else "Would not buy"


def _as_buyer_bool(value):
    return value is True or str(value).lower() == "true"


def _combined_text(row):
    # Use both review title and review body as text-model input.
    # The title often contains strong sentiment, so it is useful evidence together
    # with the longer review description.
    title = str(row.get("review_title", "") or "")
    processed = str(row.get("processed_review", "") or "")
    raw = str(row.get("review_text", "") or "")
    return clean_review_text(f"{title} {processed or raw}")


def _cue_features_from_text(text):
    # Create simple sentiment and purchase-intent features for the cue model.
    # Instead of using the whole text, this model counts meaningful cues such as
    # positive words, negative words, buy intent, and not-buy intent.
    cleaned = clean_review_text(text)
    words = cleaned.split()
    word_set = set(words)
    word_count = max(len(words), 1)
    positive_count = sum(1 for w in words if w in POSITIVE_WORDS)
    negative_count = sum(
        1
        for w in words
        if w in NEGATIVE_WORDS or w.startswith(("hate", "worst", "disappoint", "waste"))
    )
    buy_count = sum(1 for w in words if w in BUY_INTENT_WORDS)
    not_buy_count = sum(1 for w in words if w in NOT_BUY_WORDS or w.startswith(("never", "waste")))
    return {
        "word_count": word_count,
        "positive_ratio": positive_count / word_count,
        "negative_ratio": negative_count / word_count,
        "buy_intent_ratio": buy_count / word_count,
        "not_buy_ratio": not_buy_count / word_count,
        "positive_minus_negative": positive_count - negative_count,
        "has_recommend": int("recommend" in word_set),
        "has_repurchase": int("repurchase" in word_set),
        "has_waste": int("waste" in word_set),
        "has_never": int("never" in word_set),
    }


def _cue_feature_frame(texts):
    return pd.DataFrame([_cue_features_from_text(text) for text in texts]).fillna(0)


def _metadata_feature_frame(rows):
    # Build numeric features from rating, product average rating, and price.
    # Missing review ratings fall back to the product average, then to a neutral 3,
    # so the model can still predict when a field is blank.
    frame = pd.DataFrame(rows).copy()
    for col in ["review_rating", "avg_product_rating", "price"]:
        frame[col] = pd.to_numeric(frame.get(col, 0), errors="coerce")
    frame["review_rating"] = frame["review_rating"].fillna(frame["avg_product_rating"])
    frame["review_rating"] = frame["review_rating"].fillna(3)
    frame["avg_product_rating"] = frame["avg_product_rating"].fillna(frame["review_rating"])
    frame["avg_product_rating"] = frame["avg_product_rating"].fillna(3)
    frame["price"] = frame["price"].fillna(frame["price"].median() if frame["price"].notna().any() else 0)
    frame["rating_gap"] = frame["review_rating"] - frame["avg_product_rating"]
    return frame[["review_rating", "avg_product_rating", "price", "rating_gap"]]


def _fallback_probability(review_text, review_rating=None, avg_product_rating=None):
    # Keyword fallback used only when model training is unavailable.
    # It starts from a neutral buyer probability, then adjusts the score using star
    # rating, positive/negative words, and buy/not-buy phrases.
    features = _cue_features_from_text(review_text)
    rating = pd.to_numeric(pd.Series([review_rating]), errors="coerce").iloc[0]
    avg_rating = pd.to_numeric(pd.Series([avg_product_rating]), errors="coerce").iloc[0]
    score = 0.55
    if pd.notna(rating):
        score += (rating - 3) * 0.12
    elif pd.notna(avg_rating):
        score += (avg_rating - 3) * 0.08
    score += features["positive_minus_negative"] * 0.08
    score += features["buy_intent_ratio"] * 0.9
    score -= features["not_buy_ratio"] * 1.2
    return max(0.05, min(0.95, float(score)))


def train_review_label_ensemble(df):
    # Train three different models and combine them for the final buyer label.
    # Keeping them separate makes the prediction easier to explain: one model reads
    # text, one reads numeric product/review metadata, and one reads language cues.
    if not SKLEARN_READY:
        return {
            "ready": False,
            "models": {},
            "metrics": {},
            "fused_accuracy": None,
        }

    model_df = df.copy()
    # Build the training target.
    # A rating of 4 or 5 is treated as "Would buy"; if a rating is missing, the
    # original buyer flag is used as the backup label.
    ratings = pd.to_numeric(model_df.get("review_rating"), errors="coerce")
    buyer_fallback = model_df["is_a_buyer"].apply(_as_buyer_bool)
    model_df["target"] = (ratings >= 4).where(ratings.notna(), buyer_fallback).astype(int)
    model_df["combined_text"] = model_df.apply(_combined_text, axis=1)
    model_df = model_df[model_df["combined_text"].str.strip() != ""].copy()

    y = model_df["target"]
    train_idx, test_idx = train_test_split(
        model_df.index,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    train_df = model_df.loc[train_idx]
    test_df = model_df.loc[test_idx]
    y_train = train_df["target"]
    y_test = test_df["target"]

    text_model = Pipeline(
        [
            ("tfidf", TfidfVectorizer(max_features=9000, ngram_range=(1, 2), min_df=2)),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    # Model 1 uses review text; model 2 uses numeric metadata; model 3 uses cues.
    metadata_model = Pipeline(
        [
            ("scale", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )
    cue_model = RandomForestClassifier(
        n_estimators=120,
        min_samples_leaf=8,
        random_state=42,
        n_jobs=-1,
    )

    text_model.fit(train_df["combined_text"], y_train)
    metadata_model.fit(_metadata_feature_frame(train_df), y_train)
    cue_model.fit(_cue_feature_frame(train_df["combined_text"]), y_train)

    # Each model produces a probability on the validation split.
    # These probabilities are used both for individual model accuracy and for the
    # weighted ensemble that becomes the final prediction.
    text_prob = text_model.predict_proba(test_df["combined_text"])[:, 1]
    meta_prob = metadata_model.predict_proba(_metadata_feature_frame(test_df))[:, 1]
    cue_prob = cue_model.predict_proba(_cue_feature_frame(test_df["combined_text"]))[:, 1]

    metrics = {
        "text": float(accuracy_score(y_test, text_prob >= 0.5)),
        "metadata": float(accuracy_score(y_test, meta_prob >= 0.5)),
        "language_cues": float(accuracy_score(y_test, cue_prob >= 0.5)),
    }
    # Give rating/cue models more voice because text labels can be noisy.
    # This prevents one strange text prediction from overpowering a clear low or
    # high star rating and obvious sentiment words.
    weights = {
        "text": max(metrics["text"], 0.01),
        "metadata": max(metrics["metadata"] * 3, 0.01),
        "language_cues": max(metrics["language_cues"] * 2, 0.01),
    }
    total_weight = sum(weights.values())
    fused_prob = (
        text_prob * weights["text"]
        + meta_prob * weights["metadata"]
        + cue_prob * weights["language_cues"]
    ) / total_weight
    fused_accuracy = float(accuracy_score(y_test, fused_prob >= 0.5))

    return {
        "ready": True,
        "models": {
            "text": text_model,
            "metadata": metadata_model,
            "language_cues": cue_model,
        },
        "metrics": metrics,
        "weights": weights,
        "fused_accuracy": fused_accuracy,
    }


def predict_review_label(ensemble, review_text, review_title="", context=None):
    # Return a template-friendly prediction with final label and model votes.
    # The Flask routes use this one function for both the live preview and final
    # saved review so the displayed suggestion matches the saved result.
    context = context or {}
    combined_text = clean_review_text(f"{review_title} {review_text}")

    if not combined_text:
        return {
            "would_buy": False,
            "label": "Would not buy",
            "probability": 0,
            "confidence": 0,
            "source": "Empty review",
            "votes": [],
        }

    if not ensemble or not ensemble.get("ready"):
        # If trained models are unavailable, return the keyword fallback in the
        # same structure as the normal ensemble result.
        probability = _fallback_probability(
            combined_text,
            context.get("review_rating"),
            context.get("avg_product_rating"),
        )
        would_buy = probability >= 0.5
        return {
            "would_buy": would_buy,
            "label": binary_label(would_buy),
            "probability": round(probability, 4),
            "confidence": round(max(probability, 1 - probability) * 100, 1),
            "source": "Keyword fallback",
            "votes": [
                {
                    "name": "Keyword fallback",
                    "probability": round(probability, 4),
                    "label": binary_label(would_buy),
                    "data_used": "review text and rating",
                    "accuracy": None,
                }
            ],
        }

    row = {
        "review_rating": context.get("review_rating"),
        "avg_product_rating": context.get("avg_product_rating"),
        "price": context.get("price"),
    }
    models = ensemble["models"]
    metrics = ensemble.get("metrics", {})
    weights = ensemble.get("weights", {})

    model_outputs = [
        # Each model returns a probability that the reviewer would buy.
        # The model names and data descriptions are shown in the review form so the
        # prediction is not just a black box.
        (
            "Text TF-IDF Logistic Regression",
            "review title and description",
            "text",
            float(models["text"].predict_proba([combined_text])[0][1]),
        ),
        (
            "Numeric Metadata Logistic Regression",
            "star rating, average rating, price",
            "metadata",
            float(models["metadata"].predict_proba(_metadata_feature_frame([row]))[0][1]),
        ),
        (
            "Language Cue Random Forest",
            "sentiment and purchase-intent word features",
            "language_cues",
            float(models["language_cues"].predict_proba(_cue_feature_frame([combined_text]))[0][1]),
        ),
    ]

    total_weight = sum(weights.get(key, 1) for _, _, key, _ in model_outputs)
    fused_probability = sum(
        probability * weights.get(key, 1)
        for _, _, key, probability in model_outputs
    ) / total_weight
    would_buy = fused_probability >= 0.5

    # Save each model's vote separately for display.
    # The final label comes from the fused probability, but the vote list helps the
    # reviewer understand why the website suggested that label.
    votes = []
    for name, data_used, key, probability in model_outputs:
        vote_buy = probability >= 0.5
        votes.append(
            {
                "name": name,
                "probability": round(probability, 4),
                "label": binary_label(vote_buy),
                "data_used": data_used,
                "accuracy": round(metrics.get(key), 4) if metrics.get(key) is not None else None,
            }
        )

    return {
        "would_buy": would_buy,
        "label": binary_label(would_buy),
        "probability": round(fused_probability, 4),
        "confidence": round(max(fused_probability, 1 - fused_probability) * 100, 1),
        "source": "Weighted ensemble of 3 models",
        "votes": votes,
        "fused_accuracy": round(ensemble.get("fused_accuracy") or 0, 4),
        "weights": {k: round(v, 4) for k, v in weights.items()},
    }
