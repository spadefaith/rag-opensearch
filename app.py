from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
from opensearchpy import OpenSearch
from dotenv import load_dotenv
import os

app = Flask(__name__)
load_dotenv()

model = SentenceTransformer('all-MiniLM-L6-v2')

client = OpenSearch(
    hosts=[{'host': os.environ.get('OPENSEARCH_HOST'), 'port': os.environ.get('OPENSEARCH_PORT')}],
    http_auth=(os.environ.get('OPENSEARCH_USERNAME'), os.environ.get('OPENSEARCH_PASSWORD')),  # Update if needed
    use_ssl=os.environ.get('OPENSEARCH_SSL',False),
    verify_certs=os.environ.get('OPENSEARCH_VERIFY_CERTS',False),
    timeout=30,  # increase timeout seconds
    max_retries=3,
    retry_on_timeout=True,
)

index_body = {
        "settings": {
        "index": {
            "knn": True  # Important: enables k-NN on this index
        }
    },
    "mappings": {
        "properties": {
             "content": {
                "type": "text",
                "fields": {
                    "keyword": { "type": "keyword" }
                }
            },
            "category": {
                "type": "text",
                "fields": {
                    "keyword": { "type": "keyword" }
                }
            },
            "keywords": {
                "type": "text"
                # add .keyword if you want exact match on keywords string
            },
            "questions": {
                "type": "text"
            },
            "embedding": {
                "type": "knn_vector",
                "dimension": 384,  # Depends on your embedding model
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

@app.route('/')
def add():
    return "Hello, World!"

@app.route('/create/<index_name>', methods=['POST'])
def create(index_name):
    data = request.json
    documents = data.get('documents', [])

    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name, body=index_body)

    if not documents:
        return jsonify({"error": "No documents provided"}), 400
    
    has_invalid_docs = [doc for doc in documents if not (doc.get('content') and doc.get('category'))]

    if has_invalid_docs:
        return jsonify({"error": "Content and category are required"}), 400
    
    for doc in documents:
        content = doc.get('content', '')
        keywords = doc.get('keywords', [])
        questions = doc.get('questions', [])
        category = doc.get('category', '')

        if( not content or not category):
            return jsonify({"error": "Content and category are required"}), 400
        
        search_resp = client.search(index=index_name, body={
            "_source": ["content", "category"],
            "query": {
                "bool": {
                    "must": [
                        {"term": {"content.keyword": content}},
                        {"term": {"category.keyword": category}}
                    ]
                }
            }
        })

        if not search_resp['hits']['hits']:
            embedding = model.encode(content).tolist()
            client.index(index=index_name, body={
                'category': category,
                'keywords': ', '.join(keywords),
                'questions': ', '.join(questions),
                'content': content,
                'embedding': embedding
            })
            
    return jsonify({"status": "success"}), 201

@app.route('/index/<index_name>',methods=['DELETE'])
def delete_index(index_name):
    if client.indices.exists(index=index_name):
        client.indices.delete(index=index_name)
    return jsonify({"status": "success", "message": f"Index {index_name} deleted"}), 200

@app.route('/index/record/<index_name>', methods=['DELETE'])
def delete_index_record(index_name):
    record_id = request.json.get('id')
    if not record_id:
        return jsonify({"error": "ID is required"}), 400

    if client.exists(index=index_name, id=record_id):
        client.delete(index=index_name, id=record_id)
        return jsonify({"status": "success", "message": f"Record {record_id} deleted from index {index_name}"}), 200
    else:
        return jsonify({"error": "Record not found"}), 404

@app.route('/search/exact/<index_name>', methods=['POST'])
def search_exact(index_name):
    content = request.json.get('content', '')
    category = request.json.get('category', '')
    if not content or not category:
        return {"error": "Query and category is required"}, 400

    response = client.search(index=index_name, body={
        "_source": ["content", "category"],
        "query": {
            "bool": {
                "must": [
                    {"term": {"content.keyword": content}},
                    {"term": {"category.keyword": category}}
                ]
            }
        }
    })

    results = []
    for hit in response['hits']['hits']:
        score = hit['_score']
        id = hit['_id']
        sourse = hit['_source']    
        results.append({
            "content": sourse.get('content', ''),
            "category": sourse.get('category', ''),
            "keywords": sourse.get('keywords', ''),
            "questions": sourse.get('questions', ''),
            "score": score,
            "id": id
        })

    return {"results": results}

@app.route('/search/<index_name>', methods=['POST'])       
def search(index_name):
    query = request.json.get('query', '')
    nearest = request.json.get('nearest', 5)
    print(f"Searching in index: {index_name} for query: {query} with nearest: {nearest}")
    if not query:
        return {"error": "Query is required"}, 400

    embedding = model.encode(query).tolist()
    response = client.search(
        index=index_name,
        body={
            "size": nearest,
            "_source": ["content", "category","keywords","questions"],
            "query": {
                "knn": {
                    "embedding": {
                        "vector": embedding,
                        "k": nearest  # Number of nearest neighbors to return
                    }
                }
            }
        }
    )

    results = []
    for hit in response['hits']['hits']:
        score = hit['_score']
        sourse = hit['_source']    
        results.append({
            "content": sourse.get('content', ''),
            "category": sourse.get('category', ''),
            "keywords": sourse.get('keywords', ''),
            "questions": sourse.get('questions', ''),
            "score": score
        })

    return {"results": results}

    

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)  # important for Docker