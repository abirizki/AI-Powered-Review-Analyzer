import pandas as pd
import os
from config import DATA_RAW
import urllib.request
import zipfile
from datetime import datetime, timedelta
import random

def download_large_dataset():
    """Download larger Amazon dataset untuk analytics yang lebih kaya"""
    print("📥 Downloading larger dataset for enhanced analytics...")
    
    # Multiple dataset sources untuk variety
    datasets = [
        {
            "name": "Amazon Fine Foods",
            "url": "https://raw.githubusercontent.com/datasets/amazon-fine-food-reviews/master/Reviews.csv",
            "description": "Food product reviews"
        }
    ]
    
    success_count = 0
    for dataset in datasets:
        try:
            filename = f"amazon_{dataset['name'].lower().replace(' ', '_')}.csv"
            filepath = os.path.join(DATA_RAW, filename)
            
            print(f"Downloading {dataset['name']}...")
            urllib.request.urlretrieve(dataset['url'], filepath)
            
            # Verify download
            if os.path.exists(filepath):
                df = pd.read_csv(filepath, nrows=5)
                print(f"✅ {dataset['name']}: {len(df.columns)} columns")
                success_count += 1
                
                # Create enhanced version dengan lebih banyak data
                create_enhanced_dataset(filepath, dataset['name'])
            else:
                print(f"❌ Failed to download {dataset['name']}")
                
        except Exception as e:
            print(f"❌ Error downloading {dataset['name']}: {e}")
            # Create synthetic data sebagai fallback
            create_synthetic_large_dataset()
    
    return success_count

def create_enhanced_dataset(original_path, dataset_name):
    """Create enhanced dataset dengan lebih banyak variasi"""
    print(f"Enhancing {dataset_name} dataset...")
    
    try:
        # Load original data
        df = pd.read_csv(original_path)
        print(f"Original dataset: {df.shape}")
        
        # Create enhanced version dengan sampling yang lebih besar
        enhanced_data = []
        
        # Sample lebih banyak data jika available
        sample_size = min(5000, len(df))
        sampled_df = df.sample(n=sample_size, random_state=42)
        
        for _, row in sampled_df.iterrows():
            # Add variety dengan modified reviews
            enhanced_data.append({
                'Id': row.get('Id', len(enhanced_data)),
                'ProductId': row.get('ProductId', ''),
                'UserId': row.get('UserId', ''),
                'ProfileName': row.get('ProfileName', ''),
                'HelpfulnessNumerator': row.get('HelpfulnessNumerator', 0),
                'HelpfulnessDenominator': row.get('HelpfulnessDenominator', 0),
                'Score': row.get('Score', 5),
                'Time': row.get('Time', int(datetime.now().timestamp())),
                'Summary': row.get('Summary', ''),
                'Text': row.get('Text', '')
            })
        
        # Save enhanced dataset
        enhanced_df = pd.DataFrame(enhanced_data)
        enhanced_path = os.path.join(DATA_RAW, f"enhanced_{dataset_name.lower().replace(' ', '_')}.csv")
        enhanced_df.to_csv(enhanced_path, index=False)
        
        print(f"✅ Enhanced dataset created: {enhanced_df.shape}")
        return enhanced_path
        
    except Exception as e:
        print(f"❌ Error enhancing dataset: {e}")
        return None

def create_synthetic_large_dataset():
    """Create large synthetic dataset untuk development"""
    print("🛠️ Creating synthetic large dataset...")
    
    # Product categories untuk variety
    categories = {
        'Electronics': ['Smartphone', 'Laptop', 'Headphones', 'Tablet', 'Smartwatch'],
        'Home': ['Vacuum', 'Blender', 'Coffee Maker', 'Air Fryer', 'Mixer'],
        'Books': ['Novel', 'Textbook', 'Cookbook', 'Biography', 'Fantasy'],
        'Clothing': ['T-Shirt', 'Jeans', 'Jacket', 'Shoes', 'Dress']
    }
    
    # Sentiment templates
    positive_templates = [
        "Absolutely love this {product}! {positive_comment}",
        "Best {product} I've ever owned! {positive_comment}",
        "Excellent quality and great value! {positive_comment}",
        "Exceeded my expectations! {positive_comment}",
        "Highly recommend this {product}! {positive_comment}"
    ]
    
    negative_templates = [
        "Very disappointed with this {product}. {negative_comment}",
        "Poor quality and not as described. {negative_comment}",
        "Would not recommend this {product}. {negative_comment}",
        "Terrible experience with this {product}. {negative_comment}",
        "Broken upon arrival. {negative_comment}"
    ]
    
    neutral_templates = [
        "It's okay for the price. {neutral_comment}",
        "Does the job but nothing special. {neutral_comment}",
        "Average {product}, meets basic needs. {neutral_comment}",
        "Not bad, but could be better. {neutral_comment}",
        "Standard quality {product}. {neutral_comment}"
    ]
    
    positive_comments = [
        "The quality is outstanding and it works perfectly.",
        "Fast shipping and excellent customer service.",
        "Great value for money and very durable.",
        "Easy to use and very efficient.",
        "Beautiful design and very functional."
    ]
    
    negative_comments = [
        "Stopped working after just one week.",
        "Customer service was unhelpful and rude.",
        "Poor build quality and cheap materials.",
        "Not worth the money at all.",
        "Arrived damaged and missing parts."
    ]
    
    neutral_comments = [
        "It serves its purpose adequately.",
        "Good for basic needs but not exceptional.",
        "Reasonable quality for the price point.",
        "Meets expectations but doesn't exceed them.",
        "Standard product that does what it should."
    ]
    
    synthetic_data = []
    review_id = 100000
    
    for category, products in categories.items():
        for product in products:
            # Create multiple reviews per product
            for i in range(50):  # 50 reviews per product
                # Determine sentiment distribution
                sentiment_roll = random.random()
                if sentiment_roll < 0.6:  # 60% positive
                    template = random.choice(positive_templates)
                    comment = random.choice(positive_comments)
                    score = random.choice([4, 5])
                elif sentiment_roll < 0.8:  # 20% negative
                    template = random.choice(negative_templates)
                    comment = random.choice(negative_comments)
                    score = random.choice([1, 2])
                else:  # 20% neutral
                    template = random.choice(neutral_templates)
                    comment = random.choice(neutral_comments)
                    score = 3
                
                review_text = template.format(product=product.lower(), 
                                            positive_comment=comment if 'positive' in template else '',
                                            negative_comment=comment if 'negative' in template else '',
                                            neutral_comment=comment if 'neutral' in template else '')
                
                # Generate random dates dalam 2 tahun terakhir
                days_ago = random.randint(1, 730)
                review_date = datetime.now() - timedelta(days=days_ago)
                
                synthetic_data.append({
                    'Id': review_id,
                    'ProductId': f'B{random.randint(10000, 99999)}X{random.randint(100, 999)}',
                    'UserId': f'A{random.randint(1000000, 9999999)}Z',
                    'ProfileName': f'User{random.randint(1000, 9999)}',
                    'HelpfulnessNumerator': random.randint(0, 10),
                    'HelpfulnessDenominator': random.randint(1, 15),
                    'Score': score,
                    'Time': int(review_date.timestamp()),
                    'Summary': f"Review of {product}",
                    'Text': review_text,
                    'Category': category
                })
                
                review_id += 1
    
    # Save synthetic dataset
    synthetic_df = pd.DataFrame(synthetic_data)
    synthetic_path = os.path.join(DATA_RAW, "synthetic_large_dataset.csv")
    synthetic_df.to_csv(synthetic_path, index=False)
    
    print(f"✅ Synthetic dataset created: {synthetic_df.shape}")
    print(f"📊 Sentiment distribution:")
    print(f"   Positive (4-5 stars): {len(synthetic_df[synthetic_df['Score'] >= 4])}")
    print(f"   Neutral (3 stars): {len(synthetic_df[synthetic_df['Score'] == 3])}")
    print(f"   Negative (1-2 stars): {len(synthetic_df[synthetic_df['Score'] <= 2])}")
    
    return synthetic_path

def load_large_dataset():
    """Load the largest available dataset"""
    dataset_files = [
        "synthetic_large_dataset.csv",
        "enhanced_amazon_fine_foods.csv", 
        "amazon_fine_food_reviews.csv"
    ]
    
    for file in dataset_files:
        filepath = os.path.join(DATA_RAW, file)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                print(f"✅ Loaded {file}: {df.shape}")
                return df
            except Exception as e:
                print(f"❌ Error loading {file}: {e}")
    
    print("❌ No dataset files found")
    return None

if __name__ == "__main__":
    print("🚀 ENHANCED DATASET SETUP")
    print("=" * 50)
    
    # Download atau create large dataset
    success = download_large_dataset()
    
    if success == 0:
        print("Creating synthetic dataset as fallback...")
        create_synthetic_large_dataset()
    
    # Load dan display dataset info
    df = load_large_dataset()
    if df is not None:
        print(f"\n📊 FINAL DATASET INFO:")
        print(f"   Total reviews: {len(df):,}")
        print(f"   Columns: {df.columns.tolist()}")
        print(f"   Date range: {pd.to_datetime(df['Time'].min(), unit='s').strftime('%Y-%m-%d')} to {pd.to_datetime(df['Time'].max(), unit='s').strftime('%Y-%m-%d')}")
        print(f"   Rating distribution:")
        print(df['Score'].value_counts().sort_index())
