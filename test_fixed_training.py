# test_fixed_training.py
from database import db
from nlp_processor import processor
import pandas as pd
import numpy as np

def test_fixed_training():
    print("🧪 TESTING FIXED TRAINING")
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
        rating_counts = reviews['rating'].value_counts()
        print(f"⭐ Ratings: {rating_counts.to_dict()}")
        
        # Check if we have all rating types
        if len(rating_counts) < 3:
            print("⚠️ Warning: Limited rating diversity")
    else:
        print("❌ No rating column found!")
        return False
    
    # 3. Test feature extraction
    print("🔧 Testing feature extraction...")
    try:
        test_text = "This is an amazing product! I love it!"
        features = processor.extract_features(test_text)
        print(f"✅ Feature extraction works")
        print(f"📊 Sample features: {features}")
        
        # Check feature types
        for key, value in features.items():
            print(f"   {key}: {value} (type: {type(value).__name__})")
            
    except Exception as e:
        print(f"❌ Feature extraction failed: {e}")
        return False
    
    # 4. Test data preparation
    print("🔄 Testing data preparation...")
    try:
        # Use small sample for testing
        sample_reviews = reviews.head(10)
        prepared_data = processor.prepare_advanced_training_data(sample_reviews)
        print(f"✅ Data preparation works")
        print(f"📊 Prepared data shape: {prepared_data.shape}")
        
        # Check if we have the right columns
        feature_columns = ['polarity', 'subjectivity', 'text_length', 'word_count',
                          'exclamation_count', 'question_count', 'capital_ratio',
                          'has_positive_words', 'has_negative_words']
        
        missing_features = [f for f in feature_columns if f not in prepared_data.columns]
        if missing_features:
            print(f"❌ Missing features: {missing_features}")
            return False
        else:
            print("✅ All feature columns present")
            
        # Check data types
        for col in feature_columns:
            dtype = prepared_data[col].dtype
            print(f"   {col}: {dtype}")
            if not np.issubdtype(dtype, np.number):
                print(f"❌ Column {col} is not numeric: {dtype}")
                return False
                
    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        import traceback
        print(f"🔍 Details: {traceback.format_exc()}")
        return False
    
    # 5. Try training with small dataset
    print("🚀 Testing training with small dataset...")
    try:
        # Use small sample for quick test
        sample_reviews = reviews.head(20)
        success = processor.train_advanced_model(sample_reviews)
        
        if success:
            print("✅ TRAINING SUCCESSFUL!")
            
            # Test prediction with trained model
            test_text = "This product is absolutely fantastic!"
            sentiment, confidence = processor.predict_sentiment_advanced(test_text)
            print(f"🎯 Test prediction: '{test_text}' → {sentiment} ({confidence:.2f})")
            
            return True
        else:
            print("❌ Training failed")
            return False
            
    except Exception as e:
        print(f"❌ Training crashed: {e}")
        import traceback
        print(f"🔍 Details: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    result = test_fixed_training()
    print(f"\n🎯 FINAL RESULT: {'SUCCESS' if result else 'FAILED'}")
    
    if result:
        print("\n🎉 Training is now working! You can:")
        print("1. Run 'python simple_train.py' to train with full dataset")
        print("2. Run 'streamlit run advanced_dashboard.py' to use the app")
    else:
        print("\n❌ Training still has issues. Check the errors above.")