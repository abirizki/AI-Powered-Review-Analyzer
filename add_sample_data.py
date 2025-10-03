# add_sample_data.py
from database import db
import sqlite3
from datetime import datetime, timedelta
import random

def add_sample_data():
    """Add sample data for training"""
    sample_reviews = [
        # Positive reviews
        ("This product is amazing! I love it.", 5, "PRODUCT_001"),
        ("Great value for the price. Highly recommend.", 5, "PRODUCT_001"),
        ("Works perfectly and looks beautiful.", 5, "PRODUCT_002"),
        ("Fast shipping and excellent quality.", 5, "PRODUCT_002"),
        ("Very satisfied with my purchase.", 4, "PRODUCT_003"),
        ("Good product, meets my expectations.", 4, "PRODUCT_003"),
        ("Better than I expected. Will buy again.", 5, "PRODUCT_004"),
        ("Excellent customer service and product.", 5, "PRODUCT_004"),
        ("I use this every day and it's fantastic.", 5, "PRODUCT_005"),
        ("Top notch quality and durability.", 5, "PRODUCT_005"),

        # Negative reviews
        ("Terrible product. Broke after one use.", 1, "PRODUCT_001"),
        ("Poor quality. Do not recommend.", 2, "PRODUCT_001"),
        ("Not as described. Very disappointed.", 2, "PRODUCT_002"),
        ("Stopped working within a week.", 1, "PRODUCT_002"),
        ("Waste of money. Avoid this product.", 1, "PRODUCT_003"),
        ("Cheaply made and doesn't work well.", 2, "PRODUCT_003"),
        ("Customer service was unhelpful.", 1, "PRODUCT_004"),
        ("Product arrived damaged and used.", 1, "PRODUCT_004"),
        ("Does not function as advertised.", 2, "PRODUCT_005"),
        ("I regret buying this product.", 1, "PRODUCT_005"),

        # Neutral reviews
        ("It's okay, nothing special.", 3, "PRODUCT_001"),
        ("Average product. Does the job.", 3, "PRODUCT_002"),
        ("Not bad, but there are better options.", 3, "PRODUCT_003"),
        ("Mediocre quality. It's acceptable.", 3, "PRODUCT_004"),
        ("It works, but I expected more.", 3, "PRODUCT_005"),
    ]

    conn = db.get_connection()
    cursor = conn.cursor()

    for i, (review_text, rating, product_id) in enumerate(sample_reviews):
        cursor.execute(
            "INSERT INTO reviews (product_id, user_id, review_text, rating, created_date) VALUES (?, ?, ?, ?, ?)",
            (product_id, f"user_{i}", review_text, rating, (datetime.now() - timedelta(days=random.randint(0, 30))).strftime('%Y-%m-%d'))
        )

    conn.commit()
    print(f"Added {len(sample_reviews)} sample reviews.")

if __name__ == "__main__":
    add_sample_data()