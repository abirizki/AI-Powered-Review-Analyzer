import pandas as pd
import os

print("Creating sample dataset for immediate development...")

# Create sample data
sample_data = {
    'Id': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'ProductId': ['B001E4KFG0', 'B00813GRG4', 'B000LQOCH0', 'B000UA0QIQ', 'B006K2ZZ7K', 'B000GAYQKY', 'B001EQ5E1O', 'B000A3C0XW', 'B000FK7X7K', 'B000HDKZOC'],
    'UserId': ['A3SGXH7AUHU8GW', 'A1D87F6ZCVE5NK', 'ABXLMWJIXXAIN', 'A395BORC6FGVXV', 'A1UQRSCLF8GW1T', 'A2X9T5SOW1Q2F7', 'A3V8CQHL0LJE1Z', 'A1W13T5O5V1Z2P', 'A2N3Y6T9U1W4Q7', 'A3B6E9H2K5M8P1'],
    'ProfileName': ['Delmar', 'dll pa', 'Natalia Corres', 'Karl', 'C. Glover', 'John Doe', 'Jane Smith', 'Mike Johnson', 'Sarah Wilson', 'Tom Brown'],
    'HelpfulnessNumerator': [1, 0, 0, 1, 1, 2, 1, 0, 3, 1],
    'HelpfulnessDenominator': [1, 0, 0, 1, 1, 2, 1, 0, 3, 1],
    'Score': [5, 1, 4, 5, 5, 3, 2, 4, 5, 1],
    'Time': [1303862400, 1346976000, 1219017600, 1307923200, 1350777600, 1366000000, 1370000000, 1380000000, 1390000000, 1400000000],
    'Summary': [
        'Good Quality Dog Food',
        'Not as Advertised', 
        'Delight says it all',
        'Cough Medicine',
        'Great taffy',
        'Average product',
        'Disappointing',
        'Good value',
        'Excellent quality',
        'Terrible experience'
    ],
    'Text': [
        'I have bought several of the Vitality canned dog food products and have found them all to be of good quality.',
        'Product arrived labeled as Jumbo Salted Peanuts but the peanuts were actually small sized unsalted.',
        'This is a confection that has been around a few centuries. It is light, pillowy citrus gelatin.',
        'This is 10 cough drops, not 10 packs of cough drops as stated on the box. They work well for coughs.',
        'Great taffy at a great price. There was a wide assortment of yummy taffy. Delivery was quick.',
        'The product is okay, nothing special. Does the job but could be better.',
        'I was really disappointed with this product. It did not meet my expectations at all.',
        'Good value for the price. Works as expected and durable enough for daily use.',
        'Excellent quality! Better than I expected. Will definitely buy again.',
        'Terrible product. Broke after first use. Would not recommend to anyone.'
    ]
}

# Save to file
os.makedirs('data/raw', exist_ok=True)
df = pd.DataFrame(sample_data)
df.to_csv('data/raw/amazon_reviews.csv', index=False)

print(f"✅ Sample dataset created with {len(df)} reviews")
print("This allows immediate development while we work on full dataset download.")
print("Sample data includes mixed sentiments for testing.")
