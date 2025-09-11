from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pymongo import MongoClient
from nltk.tokenize import word_tokenize

client = MongoClient("mongodb://localhost:27017/")
db = client["Rekom_ERD"]
collection = db["erd"]

stop_words = set(stopwords.words('indonesian'))

class ERDRecommendationSystem:
    def __init__(self):
        self.tfidf_vectorizer = TfidfVectorizer(
            stop_words=list(stop_words),
            lowercase=True,
            ngram_range=(1, 2),  # Unigram dan bigram
            max_features=1000
        )
        self.erd_documents = []
        self.erd_data = []
        self.tfidf_matrix = None
        self.load_and_process_erds()
    
    def create_erd_document(self, erd):
        """Membuat dokumen teks dari data ERD untuk TF-IDF"""
        doc_parts = []
        
        # Tambahkan nama ERD
        doc_parts.append(erd['name'].replace('_', ' '))
        
        # Tambahkan nama entitas
        for entity in erd['entities']:
            doc_parts.append(entity['name'])
            # Tambahkan atribut entitas
            doc_parts.extend(entity['attributes'])
        
        # Tambahkan informasi relationship
        for rel in erd['relationships']:
            doc_parts.append(f"{rel['entity1']} {rel['entity2']} {rel['type']}")
        
        return ' '.join(doc_parts).lower()
    
    def load_and_process_erds(self):
        """Memuat data ERD dari MongoDB dan memproses untuk TF-IDF"""
        erds = list(collection.find({}))
        self.erd_data = erds
        self.erd_documents = [self.create_erd_document(erd) for erd in erds]
        
        if self.erd_documents:
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.erd_documents)

    # Menghilangkan stopword

    def recommend_erds(self, query, top_k=5, min_similarity=0.1):
        """
        Merekomendasikan ERD berdasarkan query menggunakan TF-IDF
        
        Args:
            query (str): Query dari user
            top_k (int): Jumlah maksimal rekomendasi yang dikembalikan
            min_similarity (float): Threshold similarity minimum
            
        Returns:
            list: List ERD yang direkomendasikan dengan skor similarity
        """
        if not self.erd_documents or self.tfidf_matrix is None:
            return []

        def remove_stopwords(text):
            words = word_tokenize(text)
            filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
            return " ".join(filtered_words)

        # Preprocess query
        processed_query = remove_stopwords(query.lower())
        
        # Transform query menggunakan TF-IDF vectorizer yang sama
        query_vector = self.tfidf_vectorizer.transform([processed_query])
        
        # Hitung cosine similarity
        similarities = cosine_similarity(query_vector, self.tfidf_matrix).flatten()
        
        # Buat list hasil dengan skor similarity
        results = []
        for i, similarity in enumerate(similarities):
            if similarity >= min_similarity:
                results.append({
                    'erd': self.erd_data[i],
                    'similarity': float(similarity),
                    'rank': i
                })
        
        # Urutkan berdasarkan similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        
        return results[:top_k]
