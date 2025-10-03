from database import db
import threading
import time

def test_database_operation(thread_id):
    """Test database operations from different threads"""
    print(f"Thread {thread_id}: Starting database test...")
    
    try:
        # Test various operations
        count = db.get_review_count()
        print(f"Thread {thread_id}: Review count = {count}")
        
        reviews = db.get_reviews(3)
        print(f"Thread {thread_id}: Got {len(reviews)} reviews")
        
        stats = db.get_sentiment_stats()
        print(f"Thread {thread_id}: Stats collected")
        
        print(f"Thread {thread_id}: ✅ All operations successful")
        
    except Exception as e:
        print(f"Thread {thread_id}: ❌ Error: {e}")

def main():
    print("🧪 Testing Thread Safety")
    print("=" * 40)
    
    # Test in main thread
    test_database_operation("Main")
    
    # Test in multiple threads
    threads = []
    for i in range(3):
        thread = threading.Thread(target=test_database_operation, args=(f"Thread-{i+1}",))
        threads.append(thread)
        thread.start()
    
    # Wait for all threads to complete
    for thread in threads:
        thread.join()
    
    print("=" * 40)
    print("🎉 Thread safety test completed!")

if __name__ == "__main__":
    main()
