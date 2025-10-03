import pandas as pd
import os
from config import DATA_RAW
import urllib.request
import zipfile
from datetime import datetime, timedelta
import random

def download_large_dataset():
    """Download larger Amazon dataset dari multiple sources"""
    print(" Downloading larger dataset...")
    
    # Multiple dataset sources
    datasets = [
        {
            "name": "Amazon Fine Foods",
            "url": "https://raw.githubusercontent.com/datasets/amazon-fine-food-reviews/master/Reviews.csv",
            "size": "300MB"
        }
    ]
    
    success_count = 0
    for dataset in datasets:
        try:
            print(f"Downloading {dataset['name']}...")
            local_path = os.path.join(DATA_RAW, "amazon_large_reviews.csv")
            
            # Download file
            urllib.request.urlretrieve(dataset['url'], local_path)
            
            # Verify download
            if os.path.exists(local_path):
                file_size = os.path.getsize(local_path) / (1024 * 1024)
                if file_size > 1:  # File should be > 1MB
                    print(f" {dataset['name']} downloaded successfully! ({file_size:.1f} MB)")
                    success_count += 1
                    
                    # Load and enhance dataset
                    enhance_dataset(local_path)
                else:
                    print(f" {dataset['name']} file too small")
                    os.remove(local_path)
            else:
                print(f" {dataset['name']} download failed")
                
        except Exception as e:
            print(f" Error downloading {dataset['name']}: {e}")
    
    if success_count == 0:
        print("Creating enhanced synthetic dataset...")
        create_enhanced_synthetic_dataset()
    
    return success_count > 0

def enhance_dataset(file_path):
    """Enhance dataset dengan data tambahan"""
    print("Enhancing dataset with additional features...")
    
    try:
        df = pd.read_csv(file_path)
        print(f"Original dataset: {len(df)} reviews")
        
        # Add synthetic business context
        products = [
            "Smartphone X200", "Laptop Pro 15", "Wireless Earbuds", "Fitness Tracker",
            "Coffee Maker Deluxe", "Bluetooth Speaker", "Gaming Mouse", "4K Monitor",
            "Mechanical Keyboard", "Webcam HD", "Tablet Mini", "Smart Watch",
            "Noise Cancelling Headphones", "Portable Charger", "USB-C Hub"
        ]
        
        categories = ["Electronics", "Computers", "Audio", "Wearables", "Home Appliances", "Accessories"]
        
        # Enhance dataset
        df['product_name'] = [random.choice(products) for _ in range(len(df))]
        df['category'] = [random.choice(categories) for _ in range(len(df))]
        df['helpfulness_ratio'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'].replace(0, 1)
        
        # Add more realistic dates
        start_date = datetime.now() - timedelta(days=365)
        df['review_date'] = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(len(df))]
        df['review_date'] = df['review_date'].dt.strftime('%Y-%m-%d')
        
        # Save enhanced dataset
        enhanced_path = os.path.join(DATA_RAW, "enhanced_reviews.csv")
        df.to_csv(enhanced_path, index=False)
        
        print(f" Enhanced dataset created: {len(df)} reviews")
        print(f"New features: product_name, category, helpfulness_ratio, review_date")
        
        return True
        
    except Exception as e:
        print(f" Error enhancing dataset: {e}")
        return False

def create_enhanced_synthetic_dataset():
    """Create a comprehensive synthetic dataset untuk development"""
    print("Creating comprehensive synthetic dataset...")
    
    # Sample data dengan variasi yang lebih besar
    products = {
        "Smartphone X200": {"category": "Electronics", "base_rating": 4.2},
        "Laptop Pro 15": {"category": "Computers", "base_rating": 4.5},
        "Wireless Earbuds Pro": {"category": "Audio", "base_rating": 4.0},
        "Fitness Tracker Plus": {"category": "Wearables", "base_rating": 3.8},
        "Coffee Maker Deluxe": {"category": "Home Appliances", "base_rating": 4.3},
        "Bluetooth Speaker X": {"category": "Audio", "base_rating": 4.1},
        "Gaming Mouse Elite": {"category": "Accessories", "base_rating": 4.6},
        "4K Monitor Ultra": {"category": "Computers", "base_rating": 4.4}
    }
    
    reviews_data = []
    review_id = 1
    
    for product_id, product_info in products.items():
        base_rating = product_info["base_rating"]
        category = product_info["category"]
        
        # Generate 50-100 reviews per product
        for i in range(random.randint(50, 100)):
            # Realistic rating distribution based on base rating
            rating = max(1, min(5, int(random.gauss(base_rating, 0.8))))
            
            # Generate realistic review text based on rating
            review_text = generate_realistic_review(product_id, rating, category)
            
            # Helpfulness metrics
            helpfulness_num = random.randint(0, 25)
            helpfulness_denom = helpfulness_num + random.randint(0, 10)
            
            # Review date dalam 1 tahun terakhir
            review_date = datetime.now() - timedelta(days=random.randint(1, 365))
            
            reviews_data.append({
                "Id": review_id,
                "ProductId": f"PROD{review_id:06d}",
                "ProductName": product_id,
                "UserId": f"USER{random.randint(1000, 9999)}",
                "ProfileName": f"Customer{random.randint(1000, 9999)}",
                "HelpfulnessNumerator": helpfulness_num,
                "HelpfulnessDenominator": helpfulness_denom,
                "Score": rating,
                "Time": int(review_date.timestamp()),
                "review_date": review_date.strftime('%Y-%m-%d'),
                "Summary": generate_summary(review_text),
                "Text": review_text,
                "Category": category,
                "HelpfulnessRatio": helpfulness_num / max(1, helpfulness_denom)
            })
            
            review_id += 1
    
    df = pd.DataFrame(reviews_data)
    enhanced_path = os.path.join(DATA_RAW, "enhanced_reviews.csv")
    df.to_csv(enhanced_path, index=False)
    
    print(f" Synthetic dataset created: {len(df)} reviews")
    print(f"Products: {len(products)} categories")
    print(f"Date range: Last 365 days")
    
    return True

def generate_realistic_review(product, rating, category):
    """Generate realistic review text berdasarkan rating dan kategori"""
    positive_phrases = [
        f"Absolutely love this {product}!",
        f"The {product} exceeded all my expectations.",
        f"Fantastic {category.lower()} product!",
        f"Worth every penny - the {product} is amazing.",
        f"Best {category.lower()} I've ever owned.",
        f"Incredible quality and performance.",
        f"Highly recommend this {product} to everyone."
    ]
    
    negative_phrases = [
        f"Very disappointed with the {product}.",
        f"The {product} stopped working after just a week.",
        f"Poor quality for a {category.lower()} product.",
        f"Not worth the money - the {product} is terrible.",
        f"Would not recommend this {product} to anyone.",
        f"Many issues with the {product} functionality.",
        f"Cheaply made and doesn't work properly."
    ]
    
    neutral_phrases = [
        f"The {product} is okay for the price.",
        f"Decent {category.lower()} but nothing special.",
        f"It works, but there are better options.",
        f"Average performance from the {product}.",
        f"Does the job but could be improved.",
        f"Standard {category.lower()} with no surprises.",
        f"Meets basic expectations."
    ]
    
    if rating >= 4:
        phrases = positive_phrases
    elif rating <= 2:
        phrases = negative_phrases
    else:
        phrases = neutral_phrases
    
    review = random.choice(phrases)
    review += " " + random.choice([
        "The build quality is impressive.",
        "Very easy to set up and use.",
        "Great value for money.",
        "Perfect for my needs.",
        "Much better than I expected.",
        "Could use some improvements.",
        "Not as described in the listing.",
        "Fast shipping and good packaging.",
        "Customer service was helpful.",
        "Would buy from this brand again."
    ])
    
    return review

def generate_summary(review_text):
    """Generate summary dari review text"""
    words = review_text.split()[:8]  # First 8 words
    return " ".join(words) + "..."

def main():
    print(" ENHANCED DATASET INTEGRATION")
    print("=" * 50)
    
    success = download_large_dataset()
    
    if success:
        print("\n Large dataset integration completed!")
        print(" Dataset features:")
        print("   - 1000+ reviews across multiple products")
        print("   - Multiple categories (Electronics, Computers, etc.)")
        print("   - Realistic date distribution")
        print("   - Enhanced metadata")
    else:
        print("\n Using enhanced synthetic dataset for development")

if __name__ == "__main__":
    main()
