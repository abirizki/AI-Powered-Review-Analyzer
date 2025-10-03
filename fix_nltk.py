# fix_nltk.py
import nltk
import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

print("📥 Downloading NLTK resources...")

# Download required NLTK data
nltk.download('punkt', quiet=False)
nltk.download('stopwords', quiet=False)
nltk.download('punkt_tab', quiet=False)  # New required resource
nltk.download('averaged_perceptron_tagger', quiet=False)
nltk.download('wordnet', quiet=False)

print("✅ NLTK resources downloaded successfully!")