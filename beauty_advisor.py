"""Task 4: budget-aware beauty product and set recommendations."""

import re
import pandas as pd

SET_CATEGORIES = ["cleanser", "toner", "serum", "moisturiser", "sunscreen"]

CATEGORY_KEYWORDS = {
    "cleanser": ["cleanser", "face wash", "facewash", "cleansing", "micellar", "wash"],
    "toner": ["toner", "tonique"],
    "serum": ["serum", "essence", "ampoule"],
    "moisturiser": [
        "moisturiser",
        "moisturizer",
        "moisture",
        "moisturising",
        "moisturizing",
        "cream",
        "whip",
        "lotion",
        "hydrat",
    ],
    "sunscreen": ["sunscreen", "sun screen", "spf", "sunblock"],
    "foundation": ["foundation", "bb cream", "cc cream", "compact"],
    "lip": ["lipstick", "lip balm", "lip gloss", "lipstick", "lip colour", "lip color"],
    "eye": ["mascara", "eyeliner", "eye cream", "kajal", "eyeshadow", "brow"],
    "mask": ["mask", "pack"],
}

PRODUCT_TYPE_MAP = {
    "skincare": {"cleanser", "toner", "serum", "moisturiser", "sunscreen", "mask"},
    "makeup": {"foundation", "lip", "eye"},
}

_catalog = None
_max_reviews = 1


def infer_category(title, tags=""):
    text = f"{title} {tags}".lower()
    for category, keywords in CATEGORY_KEYWORDS.items():
        if any(keyword in text for keyword in keywords):
            return category
    return "other"


def build_catalog(df, df_original=None):
    work = df.copy()
    if df_original is not None and "product_rating_count" in df_original.columns:
        counts = (
            df_original.groupby("product_title", sort=False)
            .agg(
                product_rating_count=("product_rating_count", "max"),
                product_tags=("product_tags", "first"),
            )
            .reset_index()
        )
        work = work.merge(counts, on="product_title", how="left")

    rows = []
    for title, group in work.groupby("product_title", sort=False):
        row = group.sort_values("review_id").iloc[0]
        tags = str(row.get("product_tags", "") or "")
        if tags.lower() == "nan":
            tags = ""
        review_count = int(len(group))
        rating_count = row.get("product_rating_count")
        if pd.notna(rating_count):
            review_count = max(review_count, int(rating_count))

        rows.append(
            {
                "review_id": int(row["review_id"]),
                "product_title": str(title),
                "brand_name": str(row.get("brand_name", "") or "Unknown"),
                "price": float(row.get("price", 0) or 0),
                "avg_product_rating": float(row.get("avg_product_rating", 0) or 0),
                "review_count": review_count,
                "category": infer_category(str(title), tags),
                "processed_review": str(row.get("processed_review", "") or ""),
            }
        )

    catalog = pd.DataFrame(rows)
    if catalog.empty:
        return catalog
    catalog["price"] = pd.to_numeric(catalog["price"], errors="coerce").fillna(0)
    catalog["avg_product_rating"] = pd.to_numeric(
        catalog["avg_product_rating"], errors="coerce"
    ).fillna(0)
    return catalog.reset_index(drop=True)


def init_advisor(df, df_original=None):
    global _catalog, _max_reviews
    _catalog = build_catalog(df, df_original)
    _max_reviews = int(_catalog["review_count"].max()) if len(_catalog) else 1
    return _catalog


def get_brands():
    if _catalog is None or _catalog.empty:
        return []
    return sorted(_catalog["brand_name"].dropna().unique().tolist())


def _normalize_scores(candidates, budget):
    if candidates.empty:
        return candidates

    rating = candidates["avg_product_rating"] / 5.0
    pop = candidates["review_count"] / max(_max_reviews, 1)
    pop = pop.clip(0, 1)
    price = candidates["price"].clip(lower=1)
    value = (candidates["avg_product_rating"] / 5.0) * (budget / price)
    value = value.clip(0, 1)
    if value.max() > 0:
        value = value / value.max()

    candidates = candidates.copy()
    candidates["_score"] = 0.5 * rating + 0.3 * pop + 0.2 * value
    candidates["_score_pct"] = (candidates["_score"] * 100).round(1)
    return candidates


def _filter_candidates(budget, brand_pref="", product_type=""):
    if _catalog is None or _catalog.empty:
        return _catalog.iloc[0:0]

    candidates = _catalog[_catalog["price"] <= budget].copy()
    brand_pref = brand_pref.strip()
    if brand_pref:
        brand_lower = brand_pref.lower()
        candidates = candidates[
            candidates["brand_name"].str.lower().str.contains(brand_lower, na=False)
        ]

    product_type = product_type.strip().lower()
    if product_type in PRODUCT_TYPE_MAP:
        allowed = PRODUCT_TYPE_MAP[product_type]
        candidates = candidates[candidates["category"].isin(allowed)]

    category_pref = product_type if product_type in CATEGORY_KEYWORDS else ""
    if category_pref:
        candidates = candidates[candidates["category"] == category_pref]

    return candidates


def _row_to_result(row, role=""):
    return {
        "review_id": int(row["review_id"]),
        "product_title": row["product_title"],
        "brand_name": row["brand_name"],
        "price": float(row["price"]),
        "avg_product_rating": float(row["avg_product_rating"]),
        "review_count": int(row["review_count"]),
        "category": row["category"],
        "score": float(row.get("_score_pct", row.get("_score", 0) * 100)),
        "role": role,
    }


def recommend_single(budget, brand_pref="", product_type=""):
    budget = float(budget)
    candidates = _filter_candidates(budget, brand_pref, product_type)
    if candidates.empty:
        return {
            "ok": False,
            "message": "No products match your budget and preferences. Try a higher budget or broader filters.",
        }

    ranked = _normalize_scores(candidates, budget).sort_values("_score", ascending=False)
    main = ranked.iloc[0]
    remaining = budget - float(main["price"])

    extras = []
    if remaining > 0:
        extra_pool = ranked.iloc[1:].copy()
        extra_pool = extra_pool[extra_pool["price"] <= remaining]
        extra_pool = extra_pool.sort_values("_score", ascending=False).head(3)
        for _, row in extra_pool.iterrows():
            extras.append(_row_to_result(row, role="extra"))
            remaining -= float(row["price"])

    explanation = (
        f"Top pick balances rating ({main['avg_product_rating']:.1f}★), "
        f"popularity ({int(main['review_count'])} reviews), and value within ₹{budget:.0f}."
    )
    if extras:
        explanation += f" With ₹{max(0, remaining):.0f} left, we added budget-friendly add-ons."

    return {
        "ok": True,
        "mode": "single",
        "main": _row_to_result(main, role="main"),
        "extras": extras,
        "remaining_budget": max(0, remaining),
        "budget": budget,
        "explanation": explanation,
        "formula": "Score = 0.5×Rating + 0.3×Popularity + 0.2×Budget efficiency",
    }


def _pick_best_for_category(pool, category, spent, budget, used_titles):
    options = pool[
        (pool["category"] == category)
        & (~pool["product_title"].isin(used_titles))
        & (pool["price"] <= (budget - spent))
    ]
    if options.empty:
        return None
    options = _normalize_scores(options, budget - spent)
    return options.sort_values("_score", ascending=False).iloc[0]


def _build_set_for_brand(brand, pool, budget):
    brand_pool = pool[pool["brand_name"] == brand]
    if brand_pool.empty:
        return None, set()

    picked = []
    used = set()
    spent = 0.0
    covered = set()

    for category in SET_CATEGORIES:
        row = _pick_best_for_category(brand_pool, category, spent, budget, used)
        if row is None:
            continue
        picked.append(row)
        used.add(row["product_title"])
        spent += float(row["price"])
        covered.add(category)

    if not picked:
        return None, covered

    frame = pd.DataFrame(picked)
    completeness = len(covered) / len(SET_CATEGORIES)
    avg_score = frame["_score"].mean() if "_score" in frame.columns else 0
    set_score = avg_score * (0.7 + 0.3 * completeness)
    return {
        "brand": brand,
        "items": frame,
        "spent": spent,
        "covered": covered,
        "completeness": completeness,
        "set_score": set_score,
    }, covered


def recommend_set(budget, brand_pref="", product_type=""):
    budget = float(budget)
    pool = _filter_candidates(budget, brand_pref="", product_type=product_type or "skincare")
    if brand_pref.strip():
        preferred = brand_pref.strip().lower()
        pool = pool.sort_values(
            by="brand_name",
            key=lambda s: s.str.lower().ne(preferred),
        )

    if pool.empty:
        return {
            "ok": False,
            "message": "No skincare products found for this budget. Try increasing your budget.",
        }

    best_bundle = None
    best_meta = None

    for brand in pool["brand_name"].unique():
        bundle, covered = _build_set_for_brand(brand, pool, budget)
        if bundle and (best_meta is None or bundle["set_score"] > best_meta["set_score"]):
            best_bundle = bundle
            best_meta = bundle

    if best_bundle is None:
        return {
            "ok": False,
            "message": "Could not build a set within budget. Try a higher budget or single-product mode.",
        }

    items = []
    used = set()
    spent = 0.0
    covered = set(best_bundle["covered"])
    supplements = []

    for _, row in best_bundle["items"].iterrows():
        items.append(_row_to_result(row, role="set"))
        used.add(row["product_title"])
        spent += float(row["price"])

    missing = [c for c in SET_CATEGORIES if c not in covered]
    for category in missing:
        row = _pick_best_for_category(pool, category, spent, budget, used)
        if row is None:
            continue
        row = row.copy()
        row = _normalize_scores(pd.DataFrame([row]), budget - spent).iloc[0]
        supplements.append(_row_to_result(row, role="supplement"))
        used.add(row["product_title"])
        spent += float(row["price"])
        covered.add(category)

    completeness = len(covered) / len(SET_CATEGORIES)
    primary_brand = best_bundle["brand"]
    explanation = (
        f"Built a {completeness * 100:.0f}% complete routine (completeness = "
        f"{len(covered)}/{len(SET_CATEGORIES)} categories). "
        f"Primary brand: {primary_brand}."
    )
    if supplements:
        explanation += " Missing categories were filled from other brands."
    if brand_pref.strip() and primary_brand.lower() != brand_pref.strip().lower():
        explanation += f" Your preferred brand had gaps; we optimised for completeness."

    return {
        "ok": True,
        "mode": "set",
        "set_items": items,
        "supplements": supplements,
        "missing_categories": [c for c in SET_CATEGORIES if c not in covered],
        "covered_categories": list(covered),
        "completeness": round(completeness * 100, 1),
        "spent": spent,
        "remaining_budget": max(0, budget - spent),
        "budget": budget,
        "primary_brand": primary_brand,
        "explanation": explanation,
        "formula": "Set score = avg(product score) × (0.7 + 0.3 × completeness)",
    }
