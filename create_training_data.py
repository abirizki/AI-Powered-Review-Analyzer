# create_training_data.py
from database import db
import pandas as pd
from datetime import datetime, timedelta
import random

def create_training_dataset():
    """Create comprehensive training dataset"""
    print("🎯 Creating AI Training Dataset...")
    
    # Sample products
    products = [
        "SMARTPHONE_X1", "LAPTOP_PRO", "HEADPHONES_A2", 
        "TABLET_MAX", "CAMERA_4K", "SMARTWATCH_V2",
        "KEYBOARD_MECH", "MOUSE_GAMING", "MONITOR_32IN",
        "SPEAKER_BLUETOOTH", "ROUTER_WIFI6", "WEBCAM_4K"
    ]
    
    # Comprehensive training reviews dengan berbagai pola
    training_reviews = [
        # 🟢 POSITIVE REVIEWS (Rating 4-5)
        ("Absolutely love this product! The quality is exceptional and it works perfectly. Highly recommended!", 5),
        ("Outstanding performance and excellent build quality. Worth every penny!", 5),
        ("This exceeded my expectations. Fast delivery and amazing customer service.", 5),
        ("Perfect product for my needs. Easy to use and very reliable.", 5),
        ("Great value for money. Would definitely buy again!", 4),
        ("Very satisfied with this purchase. Good quality and fair price.", 4),
        ("Works exactly as described. No complaints at all.", 4),
        ("Impressive features and sleek design. Very happy!", 4),
        
        # 🟡 NEUTRAL REVIEWS (Rating 3)
        ("It's okay for the price. Does the job but nothing special.", 3),
        ("Average product. Met my basic expectations but not exceptional.", 3),
        ("The product is fine, but shipping took longer than expected.", 3),
        ("Decent quality but could be improved. Overall acceptable.", 3),
        ("Not bad, but there are better options available.", 3),
        ("Does what it's supposed to, but lacks advanced features.", 3),
        
        # 🔴 NEGATIVE REVIEWS (Rating 1-2)
        ("Terrible product. Stopped working after 2 days. Very disappointed.", 1),
        ("Poor quality and awful customer service. Would not recommend.", 1),
        ("Not worth the money. Many better alternatives available.", 2),
        ("Defective item received. Packaging was also damaged.", 1),
        ("Very disappointed with this purchase. Does not work as advertised.", 2),
        ("Waste of money. Broke within a week of use.", 1),
        ("Poor build quality and unreliable performance.", 2),
    ]
    
    conn = db.get_connection()
    cursor = conn.cursor()
    
    # Clear existing data untuk training yang clean
    cursor.execute("DELETE FROM reviews")
    conn.commit()
    
    inserted_count = 0
    for i in range(100):  # Create 100 diverse training samples
        review_text, rating = random.choice(training_reviews)
        product_id = random.choice(products)
        user_id = f"train_user_{i:03d}"
        
        # Variasikan dates untuk trends analysis
        days_ago = random.randint(0, 90)
        created_date = (datetime.now() - timedelta(days=days_ago)).strftime('%Y-%m-%d')
        
        cursor.execute('''
            INSERT INTO reviews 
            (product_id, user_id, review_text, rating, helpful_votes, total_votes, created_date)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            product_id,
            user_id,
            review_text,
            rating,
            random.randint(0, 50),  # helpful_votes
            random.randint(10, 100), # total_votes
            created_date
        ))
        inserted_count += 1
    
    conn.commit()
    conn.close()
    
    print(f"✅ Training dataset created successfully!")
    print(f"📊 {inserted_count} training reviews inserted")
    print("🤖 Ready for AI model training!")

if __name__ == "__main__":
    create_training_dataset()