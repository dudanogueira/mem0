import unittest
from unittest.mock import MagicMock
import weaviate
from weaviate import Client as WeaviateClient

from mem0 import Memory

class TestWeaviate(unittest.TestCase):
    def setUp(self):
        self.client_mock = MagicMock(spec=WeaviateClient)
        self.client = weaviate.connect_to_embedded()

    def test_server_up(self):
        self.assertTrue(self.client.is_ready())

    # def test_create_collection(self):
    #     self.client.collections.delete("Test")
    #     config = {
    #         "version": "v1.1",
    #         "vector_store": {
    #             "provider": "weaviate",
    #             "config": {
    #                 # Provider-specific settings go here
    #                 "client": self.client,
    #                 "collection_name": "Test"
    #             }
    #         }
    #     }

    #     m = Memory.from_config(config)
    #     m.add("Your text here", user_id="user", metadata={"category": "example"})
    #     assert self.client.collections.exists("Test")


    def tearDown(self):
        self.weaviate.close()
