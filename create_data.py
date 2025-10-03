import pandas as pd
import os
from config import DATA_RAW

print("Creating sample dataset...")

sample_data = {
    "Id": [1, 2, 3, 4, 5],
    "ProductId": ["B001E4KFG0", "B00813GRG4", "B000LQOCH0", "B000UA0QIQ", "B006K2ZZ7K"],
    "UserId": ["A3SGXH7AUHU8GW", "A1D87F6ZCVE5NK", "ABXLMWJIXXAIN", "A395BORC6FGVXV", "A1UQRSCLF8GW1T"],
    "ProfileName": ["Delmar", "dll pa", "Natalia Corres", "Karl", "C. Glover"],
    "HelpfulnessNumerator": [1, 0, 0, 1, 1],
    "HelpfulnessDenominator": [1, 0, 0, 1, 1],
    "Score": [5, 1, 4, 5, 5],
    "Time": [1303862400, 1346976000, 1219017600, 1307923200, 1350777600],
    "Summary": [
        "Good Quality Dog Food",
        "Not as Advertised", 
        "Delight says it all",
        "Cough Medicine",
        "Great taffy"
    ],
    "Text": [
        "I have bought several of the Vitality canned dog food products and have found them all to be of good quality.",
        "Product arrived labeled as Jumbo Salted Peanuts but the peanuts were actually small sized unsalted.",
        "This is a confection that has been around a few centuries. It is light and delicious.",
        "This is 10 cough drops, not 10 packs of cough drops as stated on the box.",
        "Great taffy at a great price. There was a wide assortment of yummy taffy."
    ]
}

df = pd.DataFrame(sample_data)
df.to_csv(os.path.join(DATA_RAW, "amazon_reviews.csv"), index=False)
print(f"Sample dataset created with {len(df)} reviews")
