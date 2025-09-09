from flask import Flask, render_template, request, url_for, jsonify, send_from_directory
import graphviz
import os
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pymongo import MongoClient
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Download NLTK data jika belum ada
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')

# Koneksi MongoDB
client = MongoClient("mongodb://localhost:27017/")
db = client["Rekom_ERD"]
collection = db["erd"]

stop_words = set(stopwords.words('indonesian'))

app = Flask(__name__)

UPLOAD_FOLDER = "static"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Variabel global untuk menyimpan nama file ERD terbaru
erd_filename = "default_erd.png"

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

# Inisialisasi sistem rekomendasi
recommendation_system = ERDRecommendationSystem()

# Fungsi normalisasi nama ERD
def normalize_erd_name(erd_name):
    return erd_name.lower().replace(" ", "_")

# Fungsi untuk mengambil ERD berdasarkan nama dari MongoDB
def find_erd_by_name(name):
    return collection.find_one({"name": normalize_erd_name(name)})

# Fungsi untuk menyimpan ERD ke MongoDB
def save_erd_to_mongo(erd):
    erd["name"] = normalize_erd_name(erd["name"])
    if not find_erd_by_name(erd["name"]):
        collection.insert_one(erd)
        # Reload sistem rekomendasi setelah menambah data baru
        recommendation_system.load_and_process_erds()

# Fungsi untuk generate gambar ERD dengan standar visualisasi yang benar
def generate_erd_image(erd_name, erd):
    dot = graphviz.Digraph(format='png')
    dot.attr(nodesep="0.8", ranksep="1.2", bgcolor="white")
    dot.attr(dpi='300')  # High resolution

    # Tambahkan entitas (bentuk persegi)
    for entity in erd['entities']:
        entity_label = entity['name']
        dot.node(entity['name'], entity_label, 
                shape='rectangle', 
                style='filled', 
                fillcolor='lightblue',
                fontname='Arial Bold',
                fontsize='12')
        
        # Tambahkan atribut (bentuk elips)
        for attr in entity['attributes']:
            attr_node_id = f"{entity['name']}_{attr}"
            
            # Tandai primary key dengan underline
            if 'primary_key' in entity and attr == entity['primary_key']:
                attr_label = f"<u>{attr}</u>"
                dot.node(attr_node_id, f"<{attr_label}>", 
                        shape='ellipse',
                        style='filled',
                        fillcolor='yellow',
                        fontname='Arial',
                        fontsize='10')
            else:
                dot.node(attr_node_id, attr, 
                        shape='ellipse',
                        style='filled',
                        fillcolor='lightgreen',
                        fontname='Arial',
                        fontsize='10')
            
            # Hubungkan atribut ke entitas
            dot.edge(entity['name'], attr_node_id, arrowhead='none', len='0.5')

    # Tambahkan relationships (bentuk belah ketupat)
    relationship_counter = 0
    for relation in erd['relationships']:
        if relation['entity1'] in [e['name'] for e in erd['entities']] and relation['entity2'] in [e['name'] for e in erd['entities']]:
            relationship_counter += 1
            rel_node_id = f"rel_{relationship_counter}"
            
            # Node relationship (diamond shape)
            rel_label = relation.get('name', relation['relation'])
            dot.node(rel_node_id, rel_label,
                    shape='diamond',
                    style='filled',
                    fillcolor='orange',
                    fontname='Arial',
                    fontsize='10')
            
            # Konversi tipe relationship ke kardinalitas
            cardinality = convert_relationship_to_cardinality(relation['type'])
            
            # Edge dari entity1 ke relationship
            dot.edge(relation['entity1'], rel_node_id, 
                    label=cardinality['entity1'],
                    arrowhead='none',
                    fontsize='9',
                    fontname='Arial Bold')
            
            # Edge dari relationship ke entity2
            dot.edge(rel_node_id, relation['entity2'],
                    label=cardinality['entity2'],
                    arrowhead='none',
                    fontsize='9',
                    fontname='Arial Bold')

    filename = normalize_erd_name(erd_name) + "_erd"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    dot.render(filepath, format='png', cleanup=True)

    return f"{filename}.png"

# Fungsi untuk mengkonversi tipe relationship ke kardinalitas
def convert_relationship_to_cardinality(relationship_type):
    """Konversi tipe relationship ke format kardinalitas standar"""
    cardinality_map = {
        'one-to-one': {'entity1': '1', 'entity2': '1'},
        'one-to-many': {'entity1': '1', 'entity2': 'M'},
        'many-to-one': {'entity1': 'M', 'entity2': '1'},
        'many-to-many': {'entity1': 'M', 'entity2': 'M'}
    }
    
    return cardinality_map.get(relationship_type.lower(), {'entity1': '1', 'entity2': 'M'})

# Menghilangkan stopword
def remove_stopwords(text):
    words = word_tokenize(text)
    filtered_words = [word for word in words if word.lower() not in stop_words and len(word) > 2]
    return " ".join(filtered_words)

# Extract ERD name dari query
def extract_erd_name(text):
    text = text.lower()
    match = re.search(r"\berd\b\s*(.*)", text)
    if match:
        return normalize_erd_name(match.group(1).strip())
    return None

@app.route('/generate-erd-image/<erd_name>', methods=['GET'])
def generate_erd_image_endpoint(erd_name):
    """Endpoint untuk generate gambar ERD berdasarkan nama"""
    try:
        # Cari ERD berdasarkan nama
        erd = find_erd_by_name(erd_name)
        if not erd:
            return jsonify({"error": "ERD tidak ditemukan"}), 404
        
        # Generate gambar ERD
        filename = generate_erd_image(erd_name, erd)
        
        return jsonify({
            "erd_image": url_for('static', filename=filename),
            "erd_name": erd['name'].replace('_', ' ').title(),
            "download_url": url_for('download_erd', filename=filename)
        })
        
    except Exception as e:
        return jsonify({"error": f"Gagal generate ERD: {str(e)}"}), 500

@app.route('/download-erd/<filename>')
def download_erd(filename):
    """Endpoint untuk download file ERD"""
    try:
        return send_from_directory(app.config["UPLOAD_FOLDER"], filename, as_attachment=True)
    except Exception as e:
        return jsonify({"error": "File tidak ditemukan"}), 404

@app.route('/search-erd', methods=['POST'])
def search_erd():
    """Endpoint khusus untuk pencarian ERD dengan detail similarity"""
    data = request.json
    query = data.get("text", "").strip()
    top_k = data.get("top_k", 10)
    min_similarity = data.get("min_similarity", 0.05)
    
    if not query:
        return jsonify({"error": "Query tidak boleh kosong"}), 400
    
    recommendations = recommendation_system.recommend_erds(query, top_k, min_similarity)
    
    results = []
    for rec in recommendations:
        erd = rec['erd']
        results.append({
            'name': erd['name'].replace('_', ' ').title(),
            'similarity': round(rec['similarity'], 4),
            'entities': [entity['name'] for entity in erd['entities']],
            'entity_count': len(erd['entities']),
            'relationship_count': len(erd['relationships'])
        })
    
    return jsonify({
        "query": query,
        "results": results,
        "total_found": len(results)
    })

@app.route("/add-erd", methods=['POST'])
def add_erd():
    data = request.json
    if "name" not in data or "entities" not in data or "relationships" not in data:
        return jsonify({"error": "Data tidak lengkap"}), 400

    save_erd_to_mongo(data)
    return jsonify({"message": "ERD berhasil disimpan"}), 201

@app.route("/list-erds", methods=['GET'])
def list_erds():
    erds = list(collection.find({}, {"_id": 0, "name": 1}))
    # Format nama untuk display
    for erd in erds:
        erd['display_name'] = erd['name'].replace('_', ' ').title()
    return jsonify({"erds": erds})

@app.route("/reload-system", methods=['POST'])
def reload_system():
    """Endpoint untuk reload sistem rekomendasi (berguna saat ada perubahan data)"""
    try:
        recommendation_system.load_and_process_erds()
        return jsonify({"message": "Sistem rekomendasi berhasil di-reload"}), 200
    except Exception as e:
        return jsonify({"error": f"Gagal reload sistem: {str(e)}"}), 500

@app.route("/")
def home():
    return render_template("indeks.html", erd_image=url_for('static', filename=erd_filename))

if __name__ == '__main__':
    # Pastikan folder static ada
    if not os.path.exists(UPLOAD_FOLDER):
        os.makedirs(UPLOAD_FOLDER)
    
    app.run(debug=True)