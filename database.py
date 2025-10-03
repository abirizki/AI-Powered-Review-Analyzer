# database.py
import sqlite3
import pandas as pd
import os
from config import DATABASE_PATH
from nlp_processor import processor
import threading

class ReviewDatabase:
    def __init__(self):
        # Use thread-local storage for connections
        self._local = threading.local()
        self.create_tables()
    
    def get_connection(self):
        """Get thread-specific database connection"""
        if not hasattr(self._local, 'conn'):
            # Ensure directory exists
            os.makedirs(os.path.dirname(DATABASE_PATH), exist_ok=True)
            self._local.conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
            self._local.conn.row_factory = sqlite3.Row
        return self._local.conn
    
    def create_tables(self):
        """Create necessary tables"""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS reviews (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                user_id TEXT,
                review_text TEXT,
                rating INTEGER,
                helpful_votes INTEGER DEFAULT 0,
                total_votes INTEGER DEFAULT 0,
                sentiment_label TEXT,
                confidence_score REAL,
                created_date DATE,
                processed_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS sentiment_summary (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                product_id TEXT,
                total_reviews INTEGER,
                positive_count INTEGER,
                negative_count INTEGER,
                neutral_count INTEGER,
                avg_rating REAL,
                summary_date DATE
            )
        ''')
        
        conn.commit()
        print("✅ Database tables created successfully!")
    
    def insert_review(self, review_data):
        """Insert single review into database"""
        conn = self.get_connection()
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO reviews (product_id, user_id, review_text, rating, helpful_votes, total_votes, created_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
            review_data
        )
        conn.commit()
        return cursor.lastrowid
    
    def analyze_sentiments(self):
        """Analyze sentiments for all reviews - Thread safe version"""
        print("Analyzing sentiments...")
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get all reviews without sentiment analysis
        cursor.execute("SELECT id, review_text FROM reviews WHERE sentiment_label IS NULL")
        reviews = cursor.fetchall()
        
        analyzed_count = 0
        for review in reviews:
            review_id, review_text = review
            try:
                sentiment, confidence = processor.predict_sentiment_advanced(review_text)
                
                # Update database with sentiment analysis
                cursor.execute(
                    "UPDATE reviews SET sentiment_label = ?, confidence_score = ? WHERE id = ?",
                    (sentiment, confidence, review_id)
                )
                analyzed_count += 1
            except Exception as e:
                print(f"Error analyzing review {review_id}: {e}")
                continue
        
        conn.commit()
        print(f"✅ Sentiment analysis completed for {analyzed_count} reviews")
        return analyzed_count
    
    def get_reviews(self, limit=1000):
        """Get reviews from database"""
        conn = self.get_connection()
        try:
            query = "SELECT * FROM reviews LIMIT ?"
            return pd.read_sql_query(query, conn, params=(limit,))
        except Exception as e:
            print(f"Error getting reviews: {e}")
            return pd.DataFrame()
    
    def get_sentiment_stats(self):
        """Get sentiment statistics"""
        conn = self.get_connection()
        query = """
        SELECT 
            sentiment_label,
            COUNT(*) as count,
            AVG(confidence_score) as avg_confidence
        FROM reviews 
        WHERE sentiment_label IS NOT NULL
        GROUP BY sentiment_label
        """
        try:
            return pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Error getting sentiment stats: {e}")
            return pd.DataFrame()
    
    def get_review_count(self):
        """Get total review count"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM reviews")
            return cursor.fetchone()[0]
        except Exception as e:
            print(f"Error getting review count: {e}")
            return 0
    
    def get_sentiment_trends(self, days=30):
        """Get sentiment trends for the last N days"""
        conn = self.get_connection()
        query = """
        SELECT 
            DATE(created_date) as date,
            COUNT(*) as total_reviews,
            SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
            SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
        FROM reviews 
        WHERE created_date >= date('now', '-' || ? || ' days')
            AND sentiment_label IS NOT NULL
        GROUP BY DATE(created_date)
        ORDER BY date
        """
        try:
            return pd.read_sql_query(query, conn, params=(days,))
        except Exception as e:
            print(f"Error getting trends: {e}")
            return pd.DataFrame()

    def get_product_stats(self):
        """Get statistics by product"""
        conn = self.get_connection()
        query = """
        SELECT 
            product_id,
            COUNT(*) as total_reviews,
            AVG(rating) as avg_rating,
            SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
            SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
            SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
        FROM reviews
        WHERE sentiment_label IS NOT NULL
        GROUP BY product_id
        HAVING COUNT(*) > 0
        """
        try:
            return pd.read_sql_query(query, conn)
        except Exception as e:
            print(f"Error getting product stats: {e}")
            return pd.DataFrame()
    
    def close_connection(self):
        """Close connection for current thread"""
        if hasattr(self._local, 'conn'):
            self._local.conn.close()
            del self._local.conn

def get_sentiment_trends(self, days=30):
    """Get sentiment trends for the last N days dengan data yang lebih robust"""
    conn = self.get_connection()
    query = """
    SELECT 
        DATE(created_date) as date,
        COUNT(*) as total_reviews,
        SUM(CASE WHEN sentiment_label = 'positive' THEN 1 ELSE 0 END) as positive_count,
        SUM(CASE WHEN sentiment_label = 'negative' THEN 1 ELSE 0 END) as negative_count,
        SUM(CASE WHEN sentiment_label = 'neutral' THEN 1 ELSE 0 END) as neutral_count
    FROM reviews 
    WHERE created_date >= date('now', '-' || ? || ' days')
        AND sentiment_label IS NOT NULL
    GROUP BY DATE(created_date)
    ORDER BY date
    """
    try:
        trends = pd.read_sql_query(query, conn, params=(days,))
        
        # Ensure we have all required columns and handle empty data
        if not trends.empty:
            required_columns = ['positive_count', 'negative_count', 'neutral_count', 'total_reviews']
            for col in required_columns:
                if col not in trends.columns:
                    trends[col] = 0
            
            # Calculate percentages safely
            trends['positive_percentage'] = (trends['positive_count'] / trends['total_reviews'].replace(0, 1) * 100).round(2)
            trends['negative_percentage'] = (trends['negative_count'] / trends['total_reviews'].replace(0, 1) * 100).round(2)
            trends['neutral_percentage'] = (trends['neutral_count'] / trends['total_reviews'].replace(0, 1) * 100).round(2)
        
        return trends
        
    except Exception as e:
        print(f"Error getting trends: {e}")
        return pd.DataFrame()

    def insert_batch_reviews(self, reviews_data):
        """Insert multiple reviews at once"""
        conn = self.get_connection()
        cursor = conn.cursor()
        try:
            for review in reviews_data:
                cursor.execute(
                    "INSERT INTO reviews (product_id, user_id, review_text, rating, helpful_votes, total_votes, created_date) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    review
                )
            conn.commit()
            return len(reviews_data)
        except Exception as e:
            conn.rollback()
            print(f"Error inserting batch: {e}")
            return 0

# Global instance - this is thread-safe now
db = ReviewDatabase()