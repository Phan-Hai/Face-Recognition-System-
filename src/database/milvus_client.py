from pymilvus import connections, Collection, utility, FieldSchema, CollectionSchema, DataType
from typing import List
import numpy as np

class MilvusClient:
    def __init__(self, host: str = "localhost", port: str = "19530"):
        """
        Khởi tạo kết nối với Milvus server
        """
        self.host = host
        self.port = port
        self._connect()

    def _connect(self):
        """
        Thiết lập kết nối với Milvus server
        """
        try:
            connections.connect(
                alias="default", 
                host=self.host, 
                port=self.port
            )
        except Exception as e:
            raise Exception(f"Failed to connect to Milvus: {str(e)}")

    def create_face_collection(self, collection_name: str = "face_embeddings", dim: int = 512):
        """
        Tạo collection cho face embeddings
        Args:
            collection_name: Tên của collection
            dim: Số chiều của vector embedding
        """
        if utility.has_collection(collection_name):
            return Collection(collection_name)

        # Build FieldSchema objects and CollectionSchema for pymilvus
        fields = [
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="employee_id", dtype=DataType.VARCHAR, max_length=100),
            FieldSchema(name="name", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="department", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="position", dtype=DataType.VARCHAR, max_length=200),
            FieldSchema(name="face_embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
            FieldSchema(name="timestamp", dtype=DataType.INT64),
        ]

        schema = CollectionSchema(fields=fields, description="Collection for face embeddings")

        collection = Collection(name=collection_name, schema=schema)
        
        # Tạo index cho vector search
        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 100}
        }
        collection.create_index(
            field_name="face_embedding", 
            index_params=index_params
        )
        return collection

    def insert_face(self, collection_name: str, employee_id: str, 
                   face_embedding: np.ndarray, timestamp: int,
                   name: str = None, department: str = None, position: str = None):
        """
        Thêm face embedding mới vào collection
        """
        collection = Collection(collection_name)
        # Prepare entities as column lists matching schema order (except id which is auto)
        # Each element is a list of values (we insert one entity at a time)
        employee_col = [employee_id]
        name_col = [name if name is not None else ""]
        department_col = [department if department is not None else ""]
        position_col = [position if position is not None else ""]
        embedding_col = [face_embedding.tolist()]
        timestamp_col = [timestamp]

        entities = [employee_col, name_col, department_col, position_col, embedding_col, timestamp_col]
        collection.insert(entities)
        collection.flush()

    def search_face(self, collection_name: str, query_embedding: np.ndarray, 
                   top_k: int = 1) -> List[dict]:
        """
        Tìm kiếm khuôn mặt gần nhất
        Args:
            collection_name: Tên collection
            query_embedding: Vector embedding của khuôn mặt cần tìm
            top_k: Số kết quả trả về
        Returns:
            List các kết quả phù hợp nhất
        """
        collection = Collection(collection_name)
        collection.load()

        search_params = {
            "metric_type": "COSINE",
            "params": {"nprobe": 10}
        }

        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="face_embedding",
            param=search_params,
            limit=top_k,
            output_fields=["employee_id", "name", "department", "position", "timestamp"]
        )

        matches = []
        for hits in results:
            for hit in hits:
                entity = hit.entity
                matches.append({
                    "employee_id": entity.get('employee_id'),
                    "name": entity.get('name'),
                    "department": entity.get('department'),
                    "position": entity.get('position'),
                    "distance": hit.distance,
                    "timestamp": entity.get('timestamp')
                })

        collection.release()
        return matches

    def disconnect(self):
        """
        Đóng kết nối với Milvus server
        """
        connections.disconnect("default")