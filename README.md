# Project1-MachineLearningForNLP

## Data preparation
### Data formatting
Dobbiamo capire come rimouvere (se rimuovere) le recensioni in altre lingue.
Nella pagina del dataset dice che ce ne sono alcune in francese, ma in realtà ce ne sono un po' in tutte le lingue.

# pre-trained models
For this type of work, which involves text similarity and recommendation based on reviews, you can consider the following pre-trained models:

1. **TF-IDF (Term Frequency-Inverse Document Frequency)**:
    - A statistical measure used to evaluate the importance of a word in a document relative to a collection of documents (corpus).
    - Can be used with cosine similarity to find similar documents.

2. **Word2Vec**:
    - A group of related models that are used to produce word embeddings.
    - Can be used to find similarities between words and documents.

3. **GloVe (Global Vectors for Word Representation)**:
    - An unsupervised learning algorithm for obtaining vector representations for words.
    - Can be used to find similarities between words and documents.

4. **BERT (Bidirectional Encoder Representations from Transformers)**:
    - A transformer-based model designed to understand the context of a word in search queries.
    - Can be fine-tuned for specific tasks like text classification, similarity, and more.

5. **Sentence-BERT (SBERT)**:
    - A modification of the BERT network that uses siamese and triplet network structures to derive semantically meaningful sentence embeddings.
    - Suitable for tasks like semantic textual similarity and clustering.

6. **Doc2Vec**:
    - An extension of Word2Vec that generates vectors for entire documents.
    - Useful for finding similarities between documents.

Here is an example of how you might use TF-IDF with cosine similarity in Python:

```python
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Sample data
data = {
    'offering_id': [1, 2, 3],
    'text': [
        "Great service and clean rooms",
        "Poor service and dirty rooms",
        "Average service and decent rooms"
    ]
}

df = pd.DataFrame(data)

# Initialize the TF-IDF Vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit and transform the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text'])

# Example query
query = "clean rooms and great service"
query_tfidf = tfidf_vectorizer.transform([query])

# Calculate cosine similarity
cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()

# Find the most similar document
most_similar_index = cosine_similarities.argmax()
most_similar_offering = df.iloc[most_similar_index]

print("Most similar offering based on the query:")
print(most_similar_offering)
```

This code uses TF-IDF to vectorize the text data and then calculates the cosine similarity between the query and the documents to find the most similar one.