from typing import Any, ClassVar, Dict, Optional

from pydantic import BaseModel, Field, model_validator


class WeaviateConfig(BaseModel):
    from weaviate import WeaviateClient

    WeaviateClient: ClassVar[type] = WeaviateClient

    collection_name: str = Field("mem0", description="Name of the collection")
    embedding_model_dims: Optional[int] = Field(1536, description="Dimensions of the embedding model")
    client: Optional[WeaviateClient] = Field(None, description="Existing Weaviate client instance")
    tenant_name: Optional[str] = Field(
        None, description="Name to use as tenant. This is used for multi-tenancy."
    )
    weaviate_cloud_url: Optional[str] = Field(None, description="Weaviate Cloud URL")
    weaviate_cloud_api_key: Optional[str] = Field(None, description="Weaviate Cloud API key")

    @model_validator(mode="before")
    @classmethod
    def validate_extra_fields(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        allowed_fields = set(cls.model_fields.keys())
        input_fields = set(values.keys())
        extra_fields = input_fields - allowed_fields
        if extra_fields:
            raise ValueError(
                f"Extra fields not allowed: {', '.join(extra_fields)}. Please input only the following fields: {', '.join(allowed_fields)}"
            )
        return values

    # TODO:
    # validate: it must have client or weaviate_cloud_url and weaviate_cloud_api_key

    model_config = {
        "arbitrary_types_allowed": True,
    }
