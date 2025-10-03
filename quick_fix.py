# quick_fix.py
import sqlite3
import pandas as pd
import os
from config import DATABASE_PATH

def quick_fix_database():
    """Quick fix untuk database issues"""
    print("🔧 Applying quick fixes...")
    
    # Ensure database directory exists
    os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
    
    # Connect to database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    
    # Check if reviews table exists and has data
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='reviews'")
    table_exists = cursor.fetchone()
    
    if not table_exists:
        print("❌ Reviews table doesn't exist. Creating...")
        # Create tables
        from database import db
        db.create_tables()
    else:
        print("✅ Reviews table exists")
    
    # Check data
    cursor.execute("SELECT COUNT(*) FROM reviews")
    count = cursor.fetchone()[0]
    print(f"📊 Reviews in database: {count}")
    
    if count == 0:
        print("📝 Adding sample data...")
        # Add some sample data
        sample_reviews = [
            ("PRODUCT_001", "user_1", "I love this product! It's amazing!", 5, 10, 15, "2024-01-15"),
            ("PRODUCT_001", "user_2", "Terrible quality, very disappointed", 1, 2, 5, "2024-01-16"),
            ("PRODUCT_002", "user_3", "Good value for money", 4, 5, 8, "2024-01-17"),
            ("PRODUCT_002", "user_4", "Average product, nothing special", 3, 1, 3, "2024-01-18"),
            ("PRODUCT_003", "user_5", "Excellent quality and fast shipping", 5, 8, 10, "2024-01-19"),
        ]
        
        for review in sample_reviews:
            cursor.execute(
                "INSERT INTO reviews (product_id, user_id, review_text, rating, helpful_votes, total_votes, created_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                review
            )
        
        conn.commit()
        print(f"✅ Added {len(sample_reviews)} sample reviews")
    
    conn.close()
    print("🎉 Quick fix completed!")

if __name__ == "__main__":
    quick_fix_database()