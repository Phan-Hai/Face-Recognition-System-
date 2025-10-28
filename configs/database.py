# Cấu hình cho Milvus
milvus_config = {
    "host": "localhost",
    "port": "19530",
    "collection_name": "face_embeddings",
    "vector_dim": 512,  # Điều chỉnh theo mô hình embedding của bạn
    "index_params": {
        "metric_type": "COSINE",
        "index_type": "IVF_FLAT",
        "params": {
            "nlist": 1024
        }
    },
    "search_params": {
        "metric_type": "COSINE",
        "params": {
            "nprobe": 10
        }
    }
}