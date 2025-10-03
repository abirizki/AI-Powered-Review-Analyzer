from database import db
import pandas as pd
from config import DATA_RAW
import os

print("Initializing database...")

dataset_path = os.path.join(DATA_RAW, "amazon_reviews.csv")

if not os.path.exists(dataset_path):
    print("Dataset not found. Please run create_data.py first.")
    exit()

try:
    df = pd.read_csv(dataset_path)
    print(f"Loaded {len(df)} reviews")
    
    for _, row in df.iterrows():
        review_data = (
            str(row["ProductId"]),
            str(row["UserId"]),
            str(row["Text"]),
            int(row["Score"]),
            int(row["HelpfulnessNumerator"]),
            int(row["HelpfulnessDenominator"]),
            pd.to_datetime(row["Time"], unit="s").strftime("%Y-%m-%d")
        )
        db.insert_review(review_data)
    
    print("Database initialized successfully!")
    
    # Show sample
    sample = db.get_reviews(2)
    print("Sample data in database:")
    for _, row in sample.iterrows():
        print(f"- {row['review_text'][:50]}... (Rating: {row['rating']})")
        
except Exception as e:
    print(f"Error: {e}")
