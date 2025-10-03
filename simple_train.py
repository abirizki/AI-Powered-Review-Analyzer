# simple_train.py
from database import db
from nlp_processor import processor
import pandas as pd

def simple_training_test():
    print("🧪 SIMPLE TRAINING TEST")
    print("=" * 50)
    
    # 1. Get data
    reviews = db.get_reviews(100)
    print(f"📊 Reviews loaded: {len(reviews)}")
    
    if reviews.empty:
        print("❌ No reviews found!")
        return False
    
    # 2. Check data quality
    print(f"📋 Columns: {list(reviews.columns)}")
    if 'rating' in reviews.columns:
        print(f"⭐ Ratings: {reviews['rating'].value_counts().to_dict()}")
    else:
        print("❌ No rating column found!")
        return False
    
    # 3. Simple manual training preparation
    training_data = []
    for idx, row in reviews.iterrows():
        if pd.notna(row.get('review_text')) and pd.notna(row.get('rating')):
            training_data.append({
                'review_text': str(row['review_text']),
                'rating': int(row['rating'])
            })
    
    print(f"✅ Valid training samples: {len(training_data)}")
    
    if len(training_data) < 5:
        print("❌ Not enough valid data")
        return False
    
    # 4. Convert to DataFrame
    train_df = pd.DataFrame(training_data)
    
    # 5. Try training
    print("🚀 Starting training...")
    try:
        success = processor.train_advanced_model(train_df)
        if success:
            print("✅ TRAINING SUCCESS!")
            return True
        else:
            print("❌ Training returned False")
            return False
    except Exception as e:
        print(f"❌ Training crashed: {e}")
        import traceback
        print(f"🔍 Details: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    result = simple_training_test()
    print(f"\n🎯 FINAL RESULT: {'SUCCESS' if result else 'FAILED'}")