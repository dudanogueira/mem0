import logging
import os
import shutil
import uuid

from weaviate import WeaviateClient, classes as wvc
from weaviate.collections.classes.filters import _Filters

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    Range,
    VectorParams,
)

from mem0.vector_stores.base import VectorStoreBase
from mem0.configs.base import MemoryItem

from typing import List, Optional, Union

from pydantic import BaseModel

logger = logging.getLogger(__name__)

class OutputData(BaseModel):
    id: Optional[str]
    score: Optional[float]
    payload: Optional[dict]
    hash: Optional[str]

class Weaviate(VectorStoreBase):
    def __init__(
        self,
        collection_name: str,
        embedding_model_dims: int,
        client: WeaviateClient = None
    ):
        """
        Initialize Weaviate vector store.

        Args:
            collection_name (str): Name of the collection.
            client (WeaviateClient, optional): Existing Weaviate client instance. Defaults to None.
        """
        self.collection_name = collection_name
        if client:
            self.client = client
        self.create_col()

    def _parse_results(self, objects) -> List[OutputData]:
        memories = []
        for o in objects:
            o.properties["hash"] = str(o.properties["hash"])
            o.properties["created_at"] = str(o.properties["created_at"])
            if o.properties.get("updated_at"):
                o.properties["updated_at"] = str(o.properties["updated_at"])
            # object may have metadata (will not for fetch_object_by_id)
            try:
                distance = float(o.metadata.distance)
            except AttributeError:
                distance = None
            memories.append(
                OutputData(id=str(o.uuid), score=distance, payload=o.properties, hash=str(o.properties["hash"]))
            )
        return memories

    def create_col(self):
        """
        Create a new collection.

        Args:
            vector_size (int): Size of the vectors to be stored.
            on_disk (bool): Enables persistent storage.
            distance (Distance, optional): Distance metric for vector similarity. Defaults to Distance.COSINE.
        """
        # Skip creating collection if already exists
        if self.client.collections.exists(self.collection_name):
            logging.debug(f"Collection {self.collection_name} already exists. Skipping creation.")
            return
        
        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none()
        )

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        """
        Insert vectors into a collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to None.
            ids (list, optional): List of IDs corresponding to vectors. Defaults to None.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name}")
        with self.client.batch.dynamic() as batch:
            for vector, payload in zip(vectors, payloads):
                batch.add_object(
                    properties=payload,
                    vector=vector,
                    collection=self.collection_name,
                )
        if self.client.batch.failed_objects:
            logger.error(f"{len(self.client.batch.failed_objects)} Errors while inserting facts to {self.collection_name}: {self.client.batch.failed_objects}")

    def _create_filter(self, filters: dict) -> _Filters:
        """
        Create a Filter object from the provided filters.

        Args:
            filters (dict): Filters to apply.

        Returns:
            Filter: The created Filter object.
        """
        conditions = []
        for key, value in filters.items():
            if isinstance(value, dict) and "gte" in value and "lte" in value:
                conditions.append(FieldCondition(key=key, range=Range(gte=value["gte"], lte=value["lte"])))
            else:
                conditions.append(FieldCondition(key=key, match=MatchValue(value=value)))
        return Filter(must=conditions) if conditions else None

    def search(self, query: list, limit: int = 5, filters: _Filters = None) -> list[MemoryItem]:
        """
        Search for similar vectors.

        Args:
            query (list): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        #query_filter = self._create_filter(filters) if filters else None
        print("filters", filters)
        collection = self.client.collections.get(self.collection_name)
        results = collection.query.near_vector(
            near_vector=query,
            #filters=filters,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        return self._parse_results(results.objects)


    def delete(self, vector_id: str):
        """
        Delete a vector by UUID.

        Args:
            vector_id (str): UUID of the vector to delete.
        """
        return self.client.collections.get(self.collection_name).data.delete_by_id(vector_id)
        

    def update(self, vector_id: int, vector: list = None, payload: dict = None):
        """
        Update a vector and its payload.

        Args:
            vector_id (int): ID of the vector to update.
            vector (list, optional): Updated vector. Defaults to None.
            payload (dict, optional): Updated payload. Defaults to None.
        """
        point = PointStruct(id=vector_id, vector=vector, payload=payload)
        self.client.upsert(collection_name=self.collection_name, points=[point])

    def get(self, vector_id: str) -> dict:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str or List[str]): ID of the vector to retrieve.

        Returns:
            dict: Retrieved vector.
        """
        return self._parse_results(
            [self.client.collections.get(self.collection_name).query.fetch_object_by_id(vector_id)]
        )

    def list_cols(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        return self.client.get_collections()

    def delete_col(self):
        """Delete a collection."""
        self.client.delete_collection(collection_name=self.collection_name)

    def col_info(self) -> dict:
        """
        Get information about a collection.

        Returns:
            dict: Collection information.
        """
        return self.client.get_collection(collection_name=self.collection_name)

    def list(self, filters: dict = None, limit: int = 100) -> list:
        """
        List all vectors in a collection.

        Args:
            filters (dict, optional): Filters to apply to the list. Defaults to None.
            limit (int, optional): Number of vectors to return. Defaults to 100.

        Returns:
            list: List of vectors.
        """
        query_filter = self._create_filter(filters) if filters else None
        result = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
            with_vectors=False,
        )
        return result
