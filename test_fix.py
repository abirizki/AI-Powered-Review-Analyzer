# test_fix.py
print("🧪 Testing fixed components...")

try:
    from config import DATABASE_PATH, SENTIMENT_MODEL_PATH
    print("✅ Config: OK")
except Exception as e:
    print(f"❌ Config Error: {e}")

try:
    from database import db
    print("✅ Database Import: OK")
    print(f"Database path: {DATABASE_PATH}")
except Exception as e:
    print(f"❌ Database Import Error: {e}")

try:
    from nlp_processor import processor
    print("✅ NLP Processor: OK")
except Exception as e:
    print(f"❌ NLP Processor Error: {e}")

print("\n🚀 Testing sentiment analysis...")
try:
    sentiment, confidence = processor.predict_sentiment_advanced("I love this product!")
    print(f"✅ Sentiment Analysis: {sentiment} (confidence: {confidence:.2f})")
except Exception as e:
    print(f"❌ Sentiment Analysis Error: {e}")

print("\n🎉 All tests completed!")