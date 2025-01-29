import logging
import datetime

import weaviate
from weaviate import WeaviateClient, classes as wvc
from weaviate.collections.classes.filters import _Filters
from weaviate.classes.query import Filter


from qdrant_client.models import (
    FieldCondition,
    Filter,
    MatchValue,
    Range,
)

from mem0.vector_stores.base import VectorStoreBase
from mem0.configs.base import MemoryItem

from typing import List, Optional
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
        embedding_model_dims: int, # not used
        client: WeaviateClient = None,
        tenant_name: str = None,
        weaviate_cloud_url: str = None,
        weaviate_cloud_api_key: str = None,
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
        else:
            self.client = weaviate.connect_to_weaviate_cloud(
                cluster_url=weaviate_cloud_url,
                auth_credentials=weaviate.auth.AuthApiKey(weaviate_cloud_api_key)
            )
        # the option to use the user_id as tenant is on.
        self.tenant_name = tenant_name
        collection = self.client.collections.get(self.collection_name)
        if self.tenant_name:
            collection = collection.with_tenant(self.tenant_name)   
            self.multi_tenant = True
        self.collection = collection

        self.create_col(tenant_name=tenant_name)

    def _parse_results(self, objects) -> List[OutputData]:
        memories = []
        for o in objects:
            o.properties["hash"] = str(o.properties["hash"])
            o.properties["created_at"] = str(o.properties["created_at"])
            if o.properties.get("updated_at"):
                o.properties["updated_at"] = str(o.properties["updated_at"])
            # object may have metadata (will not for fetch_object_by_id)
            if not o.metadata.distance:
                distance=None
            else:
                try:
                    distance = float(o.metadata.distance)
                except AttributeError:
                    distance = None
            memories.append(
                OutputData(id=str(o.uuid), score=distance,
                           payload=o.properties, hash=str(o.properties["hash"]))
            )
        return memories

    def create_col(self, tenant_name: str = None):
        """
        Create a new collection.

        Args:
            vector_size (int): Size of the vectors to be stored.
            on_disk (bool): Enables persistent storage.
            distance (Distance, optional): Distance metric for vector similarity. Defaults to Distance.COSINE.
        """
        # Skip creating collection if already exists
        if self.client.collections.exists(self.collection_name):
            logging.debug(
                f"Collection {self.collection_name} already exists. Skipping creation.")
            # we need to make sure also that our tenant exists
            self.create_tenant()
            return

        multi_tenant_config = None
        if tenant_name:
            multi_tenant_config = wvc.config.Configure.multi_tenancy(
                auto_tenant_activation=True,
                auto_tenant_creation=True
            )
        self.client.collections.create(
            name=self.collection_name,
            vectorizer_config=wvc.config.Configure.Vectorizer.none(),
            multi_tenancy_config=multi_tenant_config,
            properties=[
                wvc.config.Property(name="user_id", data_type=wvc.config.DataType.TEXT),
            ]
        )
        # make sure our tenant exists, as mem0 can look it before adding new memories
        if self.tenant_name:
            self.create_tenant()

    def create_tenant(self):
        self.client.collections.get(self.collection_name).tenants.create(self.tenant_name)

    def insert(self, vectors: list, payloads: list = None, ids: list = None):
        """
        Insert vectors into a collection.

        Args:
            vectors (list): List of vectors to insert.
            payloads (list, optional): List of payloads corresponding to vectors. Defaults to None.
            ids (list, optional): List of IDs corresponding to vectors. Defaults to None.
        """
        logger.info(f"Inserting {len(vectors)} vectors into collection {self.collection_name} and tenant {self.tenant_name}")
        with self.client.batch.dynamic() as batch:
            for vector, payload in zip(vectors, payloads):
                batch.add_object(
                    properties=payload,
                    vector=vector,
                    collection=self.collection_name,
                    tenant=self.tenant_name
                )
        if self.client.batch.failed_objects:
            logger.error(f"{len(self.client.batch.failed_objects)} Errors while inserting facts to {
                         self.collection_name}: {self.client.batch.failed_objects} and tenant {self.tenant_name}")

    def _create_filter(self, json_filter: dict) -> _Filters:
        """
        Create a Filter object from the provided filters.

        Args:
            filters (dict): Filters to apply.

        Returns:
            Filter: The created Filter object.
        """
        if not json_filter:
            return None

        if json_filter:
            filters = []
            for k, v in json_filter.items():
                filters.append(
                    wvc.query.Filter.by_property(k).equal(v)
                )
        final_filters = wvc.query.Filter.all_of(filters)
        logger.info(f"Processed json filters: {json_filter} into {final_filters}")
        return final_filters

        def process_condition(field, value):
            if isinstance(value, dict):
                if "in" in value:
                    return wvc.query.Filter.by_property(field).contains_any(value["in"])
                elif "gte" in value:
                    return wvc.query.Filter.by_property(field).greater_or_equal(value["gte"])
                elif "lte" in value:
                    return wvc.query.Filter.by_property(field).less_or_equal(value["lte"])
                elif "gt" in value:
                    return wvc.query.Filter.by_property(field).greater_than(value["gt"])
                elif "lt" in value:
                    return wvc.query.Filter.by_property(field).less_than(value["lt"])
            return wvc.query.Filter.by_property(field).equal(value)

        def process_and_conditions(conditions):
            combined_filter = None
            for condition in conditions:
                for field, value in condition.items():
                    current_filter = process_condition(field, value)
                    if combined_filter is None:
                        combined_filter = current_filter
                    else:
                        combined_filter = combined_filter & current_filter
            return combined_filter

        def process_or_conditions(conditions):
            combined_filter = None
            for condition in conditions:
                for field, value in condition.items():
                    current_filter = process_condition(field, value)
                    if combined_filter is None:
                        combined_filter = current_filter
                    else:
                        combined_filter = combined_filter | current_filter
            return combined_filter

        final_filter = None

        # if "AND" in json_filter:
        #     and_filter = process_and_conditions(json_filter["AND"])
        #     if final_filter is not None:
        #         final_filter = final_filter & and_filter
        #     else:
        #         final_filter = and_filter            

        # if "OR" in json_filter:
        #     or_filter = process_or_conditions(json_filter["OR"])
        #     if final_filter is not None:
        #         final_filter = final_filter & or_filter
        #     else:
        #         final_filter = or_filter

        # # handle simple conditions
        # if "AND" not in json_filter and "OR" not in json_filter:
        #     and_filter = process_or_conditions(json_filter["OR"])
        #     if final_filter is not None:
        #         final_filter = final_filter & or_filter
        #     else:
        #         final_filter = or_filter
        
    def search(self, query: list, limit: int = 5, filters=None) -> list[MemoryItem]:
        """
        Search for similar vectors.

        Args:
            query (list): Query vector.
            limit (int, optional): Number of results to return. Defaults to 5.
            filters (dict, optional): Filters to apply to the search. Defaults to None.

        Returns:
            list: Search results.
        """
        filters_parsed = self._create_filter(filters) if filters else None
        results = self.collection.query.near_vector(
            near_vector=query,
            filters=filters_parsed,
            limit=limit,
            return_metadata=wvc.query.MetadataQuery(distance=True)
        )
        return self._parse_results(results.objects)

    def delete(self, vector_id: str) -> bool:
        """
        Delete a vector by UUID.

        Args:
            vector_id (str): UUID of the vector to delete.
        """
     
        return self.collection.get(self.collection_name).data.delete_by_id(vector_id)

    def update(self, vector_id: int, vector: list = None, payload: dict = None):
        """
        Update a vector and its payload.

        Args:
            vector_id (int): ID of the vector to update.
            vector (list, optional): Updated vector. Defaults to None.
            payload (dict, optional): Updated payload. Defaults to None.
        """
        # for some reason the dates are coming differently.
        # TODO: investigate
        payload["created_at"] = datetime.datetime.strptime(
            payload["created_at"], "%Y-%m-%d %H:%M:%S.%f%z"
        )
        payload["updated_at"] = datetime.datetime.strptime(
            payload["updated_at"], "%Y-%m-%dT%H:%M:%S.%f%z"
        )

        self.collection.data.update(
            uuid=vector_id,
            vector=vector,
            properties=payload
        )

    def get(self, vector_id: str) -> dict:
        """
        Retrieve a vector by ID.

        Args:
            vector_id (str or List[str]): ID of the vector to retrieve.

        Returns:
            dict: Retrieved vector.
        """
        object = self.collection.query.fetch_object_by_id(vector_id)
        return self._parse_results([object])[0] if object else None

    def list_cols(self) -> list:
        """
        List all collections.

        Returns:
            list: List of collection names.
        """
        return self.client.collections.list_all()

    def delete_col(self):
        """Delete a collection."""
        self.client.collections.delete(self.collection_name)

    def col_info(self) -> dict:
        """
        Get information about a collection.

        Returns:
            dict: Collection information.
        """
        return self.client.collections.get(self.collection_name).config.get().to_dict()

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
        result = self.collection.query.fetch_objects(
            filters=query_filter,
            limit=limit,
            include_vector=True
        )
        parsed_results = self._parse_results(result.objects)
        return [parsed_results]

    def __del__(self):
        """
        Close the database connection when the object is deleted.
        """
        self.client.close()