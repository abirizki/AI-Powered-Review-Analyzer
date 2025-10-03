# train_model.py
from database import db
from nlp_processor import processor
import pandas as pd

print("🚀 SENTIMENT ANALYSIS MODEL TRAINING")
print("=" * 50)

def main():
    # Load reviews for training
    reviews = db.get_reviews(1000)  # Use all available reviews
    print(f"Loaded {len(reviews)} reviews for training")
    
    if len(reviews) == 0:
        print("No reviews found. Please initialize database first.")
        return
    
    # Try to load existing model first
    if processor.load_model():
        print("Using pre-trained model")
    else:
        print("Training new model...")
        # Train model if no pre-trained model exists
        success = processor.train_model(reviews)
        if not success:
            print("Training failed. Using TextBlob for sentiment analysis.")
    
    # Analyze sentiments for all reviews
    db.analyze_sentiments()
    
    # Show results
    stats = db.get_sentiment_stats()
    print("\n📊 SENTIMENT ANALYSIS RESULTS:")
    print(stats)
    
    # Show sample predictions
    sample_reviews = db.get_reviews(3)
    print("\n🔍 SAMPLE PREDICTIONS:")
    for _, review in sample_reviews.iterrows():
        sentiment = review.get('sentiment_label', 'Not analyzed')
        confidence = review.get('confidence_score', 0)
        text_preview = review['review_text'][:60] + "..." if len(review['review_text']) > 60 else review['review_text']
        print(f"- {text_preview}")
        print(f"  Sentiment: {sentiment} (Confidence: {confidence:.2f})")
        print()

if __name__ == "__main__":
    main()