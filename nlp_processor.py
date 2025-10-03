# nlp_processor.py - PERBAIKAN UNTUK SCIPY SPARSE ERROR
import pandas as pd
import re
import nltk
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import joblib
from config import SENTIMENT_MODEL_PATH, VECTORIZER_PATH
import os
import numpy as np
from datetime import datetime

# Download NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class AdvancedReviewProcessor:
    def __init__(self):
        self.vectorizer = None
        self.model = None
        self.advanced_model = None
        self.aspect_model = None
        self.stop_words = set(stopwords.words('english'))
        self.performance_metrics = {}
        
    def advanced_text_processing(self, text):
        """Advanced text preprocessing dengan NLP techniques"""
        if pd.isna(text):
            return ""
            
        text = str(text).lower()
        
        # Remove special characters, digits, and extra whitespace
        text = re.sub(r'[^a-zA-Z\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Tokenize and remove stopwords
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token not in self.stop_words and len(token) > 2]
        
        # Lemmatization (simple version)
        lemmatized_tokens = self.simple_lemmatize(tokens)
        
        return ' '.join(lemmatized_tokens)
    
    def simple_lemmatize(self, tokens):
        """Simple lemmatization rules"""
        lemmatized = []
        for token in tokens:
            if token.endswith('ing'):
                token = token[:-3]
            elif token.endswith('ed'):
                token = token[:-2]
            elif token.endswith('s'):
                token = token[:-1]
            lemmatized.append(token)
        return lemmatized
    
    def extract_features(self, text):
        """Extract advanced features dari text - FIXED DATA TYPES"""
        cleaned_text = self.advanced_text_processing(text)
        analysis = TextBlob(cleaned_text)
        
        # FIX: Convert boolean to integer untuk sparse matrix compatibility
        has_positive = int(any(word in text.lower() for word in ['excellent', 'amazing', 'great', 'love', 'perfect', 'awesome']))
        has_negative = int(any(word in text.lower() for word in ['terrible', 'awful', 'horrible', 'hate', 'disappointing', 'poor']))
        
        features = {
            'text_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text.replace(' ', '')),
            'polarity': float(analysis.sentiment.polarity),
            'subjectivity': float(analysis.sentiment.subjectivity),
            'exclamation_count': int(text.count('!')),
            'question_count': int(text.count('?')),
            'capital_ratio': float(sum(1 for c in text if c.isupper()) / max(1, len(text))),
            'has_positive_words': has_positive,  # Now integer
            'has_negative_words': has_negative,  # Now integer
        }
        
        return features
    
    def get_sentiment_advanced(self, text):
        """Advanced sentiment analysis dengan feature engineering"""
        features = self.extract_features(text)
        
        # Rule-based scoring dengan weights
        rule_score = 0
        rule_score += features['polarity'] * 0.4
        rule_score += (1 - features['subjectivity']) * 0.1  # Lebih objective -> lebih reliable
        
        if features['has_positive_words']:
            rule_score += 0.2
        if features['has_negative_words']:
            rule_score -= 0.2
        
        # Adjust based on text characteristics
        if features['exclamation_count'] > 2:
            rule_score *= 1.1  # Emphasize emotional texts
        if features['word_count'] < 5:
            rule_score *= 0.8  # Penalize very short reviews
        
        # Convert to sentiment label
        if rule_score > 0.1:
            sentiment = "positive"
            confidence = min(0.95, (rule_score + 1) / 2)
        elif rule_score < -0.1:
            sentiment = "negative" 
            confidence = min(0.95, (-rule_score + 1) / 2)
        else:
            sentiment = "neutral"
            confidence = 0.7
        
        return sentiment, confidence
    
    def prepare_advanced_training_data(self, reviews_df):
        """Prepare data untuk advanced model training dengan improved handling"""
        print("🔄 Preparing advanced training data...")
        
        # Use existing columns from database
        def rating_to_sentiment(rating):
            try:
                rating_val = int(rating)
                if rating_val >= 4:
                    return "positive"
                elif rating_val == 3:
                    return "neutral"
                else:
                    return "negative"
            except (ValueError, TypeError):
                return "neutral"
        
        # Create a copy to avoid modifying original
        df = reviews_df.copy()
        
        # Map ratings to sentiments
        df['sentiment'] = df['rating'].apply(rating_to_sentiment)
        df['cleaned_text'] = df['review_text'].apply(self.advanced_text_processing)
        
        # Extract advanced features
        feature_data = []
        for text in df['review_text']:
            feature_data.append(self.extract_features(text))
        
        features_df = pd.DataFrame(feature_data)
        
        # FIX: Ensure all feature columns have proper numeric types
        numeric_columns = ['text_length', 'word_count', 'char_count', 'polarity', 'subjectivity', 
                          'exclamation_count', 'question_count', 'capital_ratio', 
                          'has_positive_words', 'has_negative_words']
        
        for col in numeric_columns:
            if col in features_df.columns:
                features_df[col] = pd.to_numeric(features_df[col], errors='coerce').fillna(0)
        
        df = pd.concat([df, features_df], axis=1)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"✅ Prepared {len(df)} samples for training")
        return df
    
    def train_advanced_model(self, reviews_df):
        """Train advanced machine learning model dengan FIXED SPARSE MATRIX ISSUE"""
        print("🤖 Starting Advanced AI Model Training...")
        
        try:
            # Validasi input data
            if reviews_df is None or reviews_df.empty:
                print("❌ No data provided for training")
                return False
            
            print(f"📊 Initial data shape: {reviews_df.shape}")
            
            # Check required columns
            required_columns = ['review_text', 'rating']
            missing_columns = [col for col in required_columns if col not in reviews_df.columns]
            if missing_columns:
                print(f"❌ Missing required columns: {missing_columns}")
                print(f"📋 Available columns: {list(reviews_df.columns)}")
                return False
            
            # Clean data
            print("🧹 Cleaning training data...")
            clean_df = reviews_df.copy()
            
            # Remove rows with missing values
            initial_count = len(clean_df)
            clean_df = clean_df.dropna(subset=['review_text', 'rating'])
            clean_df = clean_df[clean_df['review_text'].str.strip().str.len() > 0]
            
            print(f"📈 Data after cleaning: {len(clean_df)}/{initial_count} samples")
            
            if len(clean_df) < 5:
                print(f"❌ Insufficient data after cleaning: {len(clean_df)} samples")
                return False
            
            # Prepare features
            print("🔧 Preparing features...")
            try:
                training_df = self.prepare_advanced_training_data(clean_df)
                
                if len(training_df) < 5:
                    print(f"❌ Insufficient data after text processing: {len(training_df)} samples")
                    return False
                    
            except Exception as e:
                print(f"❌ Feature preparation error: {e}")
                return False
            
            # Define feature columns - FIXED: ensure all are numeric
            feature_columns = [
                'polarity', 'subjectivity', 'text_length', 'word_count',
                'exclamation_count', 'question_count', 'capital_ratio',
                'has_positive_words', 'has_negative_words'
            ]
            
            # Check if all feature columns exist
            missing_features = [f for f in feature_columns if f not in training_df.columns]
            if missing_features:
                print(f"❌ Missing feature columns: {missing_features}")
                return False
            
            # FIX: Ensure all feature columns are numeric
            for col in feature_columns:
                training_df[col] = pd.to_numeric(training_df[col], errors='coerce').fillna(0)
            
            print("🎯 Starting model training...")
            
            try:
                X_advanced = training_df[feature_columns]
                y = training_df['sentiment']
                
                # FIX: Convert to numpy array dengan tipe data yang tepat
                X_advanced_array = X_advanced.astype(np.float64).values
                
                # Text features with smaller vocabulary for stability
                self.vectorizer = TfidfVectorizer(
                    max_features=200,  # Small for stability
                    stop_words='english',
                    ngram_range=(1, 1)
                )
                
                X_text = self.vectorizer.fit_transform(training_df['cleaned_text'])
                
                # FIX: Combine features dengan cara yang compatible
                from scipy.sparse import hstack, csr_matrix
                
                # Convert advanced features to sparse matrix
                X_advanced_sparse = csr_matrix(X_advanced_array)
                
                # Combine features - sekarang kedua matrix sparse
                X_combined = hstack([X_text, X_advanced_sparse])
                
                # Simple train-test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X_combined, y, test_size=0.2, random_state=42
                )
                
                # Use simpler model for stability
                self.advanced_model = RandomForestClassifier(
                    n_estimators=30,    # Small for speed
                    max_depth=6,        # Small for generalization
                    random_state=42,
                    min_samples_split=5,
                    min_samples_leaf=2
                )
                
                print("📚 Training model...")
                self.advanced_model.fit(X_train, y_train)
                
                # Basic evaluation
                y_pred = self.advanced_model.predict(X_test)
                accuracy = accuracy_score(y_test, y_pred)
                
                # Simple cross-validation
                cv_scores = cross_val_score(self.advanced_model, X_combined, y, cv=3, scoring='accuracy')
                
                # Store performance metrics
                self.performance_metrics = {
                    'accuracy': accuracy,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std(),
                    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'training_samples': len(training_df),
                    'feature_importance': dict(zip(feature_columns, self.advanced_model.feature_importances_))
                }
                
                print(f"✅ Training completed!")
                print(f"📈 Accuracy: {accuracy:.3f}")
                print(f"🎯 Cross-validation: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
                
                # Save model
                self.save_advanced_model()
                
                return True
                
            except Exception as e:
                print(f"❌ Model training error: {e}")
                import traceback
                print(f"🔍 Detailed error: {traceback.format_exc()}")
                return False
                
        except Exception as e:
            print(f"❌ Training process error: {e}")
            import traceback
            print(f"🔍 Detailed error: {traceback.format_exc()}")
            return False
    
    def predict_sentiment_advanced(self, text):
        """Predict sentiment menggunakan advanced model"""
        if self.advanced_model and self.vectorizer:
            try:
                # Extract features
                features = self.extract_features(text)
                
                # FIX: Prepare features dengan tipe data yang benar
                feature_values = [
                    features['polarity'], features['subjectivity'], features['text_length'],
                    features['word_count'], features['exclamation_count'], features['question_count'],
                    features['capital_ratio'], features['has_positive_words'], features['has_negative_words']
                ]
                
                # Convert to numpy array dengan tipe float
                feature_array = np.array([feature_values], dtype=np.float64)
                
                # Text features
                cleaned_text = self.advanced_text_processing(text)
                if cleaned_text:
                    X_text = self.vectorizer.transform([cleaned_text])
                    
                    # Combine features - FIXED: convert to sparse matrix
                    from scipy.sparse import hstack, csr_matrix
                    X_features = csr_matrix(feature_array)
                    X_combined = hstack([X_text, X_features])
                    
                    # Predict
                    prediction = self.advanced_model.predict(X_combined)[0]
                    probabilities = self.advanced_model.predict_proba(X_combined)[0]
                    confidence = max(probabilities)
                    
                    return prediction, confidence
                else:
                    return "neutral", 0.5
            except Exception as e:
                print(f"Advanced model prediction error: {e}")
                # Fallback to rule-based
                return self.get_sentiment_advanced(text)
        else:
            # Fallback to rule-based
            return self.get_sentiment_advanced(text)
    
    def detect_aspects(self, text):
        """Detect product aspects mentioned in review"""
        aspects = {
            'price': ['price', 'cost', 'expensive', 'cheap', 'value', 'affordable'],
            'quality': ['quality', 'durable', 'well-made', 'cheaply', 'sturdy'],
            'performance': ['performance', 'speed', 'fast', 'slow', 'efficient'],
            'design': ['design', 'look', 'appearance', 'style', 'beautiful', 'ugly'],
            'features': ['feature', 'function', 'capability', 'option', 'setting'],
            'battery': ['battery', 'charge', 'power', 'life', 'lasting'],
            'service': ['service', 'support', 'warranty', 'return', 'shipping']
        }
        
        detected_aspects = []
        text_lower = text.lower()
        
        for aspect, keywords in aspects.items():
            if any(keyword in text_lower for keyword in keywords):
                detected_aspects.append(aspect)
        
        return detected_aspects
    
    def analyze_review_comprehensive(self, text):
        """Comprehensive review analysis"""
        sentiment, confidence = self.predict_sentiment_advanced(text)
        aspects = self.detect_aspects(text)
        features = self.extract_features(text)
        
        analysis = {
            'sentiment': sentiment,
            'confidence': confidence,
            'aspects': aspects,
            'features': features,
            'text_length': features['text_length'],
            'word_count': features['word_count'],
            'polarity': features['polarity'],
            'subjectivity': features['subjectivity']
        }
        
        return analysis
    
    def save_advanced_model(self):
        """Save advanced model dan metrics"""
        if self.advanced_model and self.vectorizer:
            try:
                model_data = {
                    'model': self.advanced_model,
                    'vectorizer': self.vectorizer,
                    'performance_metrics': self.performance_metrics,
                    'timestamp': datetime.now()
                }
                os.makedirs(os.path.dirname(SENTIMENT_MODEL_PATH), exist_ok=True)
                joblib.dump(model_data, SENTIMENT_MODEL_PATH.replace('.pkl', '_advanced.pkl'))
                print("✅ Advanced model saved successfully!")
            except Exception as e:
                print(f"Error saving model: {e}")
    
    def load_advanced_model(self):
        """Load advanced model"""
        try:
            advanced_path = SENTIMENT_MODEL_PATH.replace('.pkl', '_advanced.pkl')
            if os.path.exists(advanced_path):
                model_data = joblib.load(advanced_path)
                self.advanced_model = model_data['model']
                self.vectorizer = model_data['vectorizer']
                self.performance_metrics = model_data.get('performance_metrics', {})
                print("✅ Advanced model loaded successfully!")
                if self.performance_metrics:
                    print(f"📊 Model performance: {self.performance_metrics.get('accuracy', 'N/A')}")
                return True
        except Exception as e:
            print(f"Error loading advanced model: {e}")
        return False

# Global instance dengan fallback
processor = AdvancedReviewProcessor()

# Fallback functions untuk compatibility
def get_sentiment_textblob(text):
    return processor.get_sentiment_advanced(text)

def predict_sentiment(text):
    return processor.predict_sentiment_advanced(text)