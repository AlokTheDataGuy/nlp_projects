"""
Document Store Module

This module provides an interface to the MongoDB document store.
"""

import os
import logging
import yaml
import asyncio
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from motor.motor_asyncio import AsyncIOMotorClient
from pymongo import MongoClient
from pymongo.collection import Collection
from pymongo.database import Database

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentStore:
    """
    Class for interacting with the MongoDB document store.
    """

    def __init__(self, config_path: str = "config/app_config.yaml", async_mode: bool = False):
        """
        Initialize the DocumentStore.

        Args:
            config_path: Path to the configuration file.
            async_mode: Whether to use async MongoDB client.
        """
        self.config = self._load_config(config_path)
        self.async_mode = async_mode

        # MongoDB configuration
        self.uri = os.environ.get("MONGODB_URI", self.config["database"]["mongodb"]["uri"])
        self.db_name = os.environ.get("MONGODB_DB", self.config["database"]["mongodb"]["db_name"])
        self.collections_config = self.config["database"]["mongodb"]["collections"]

        # Initialize MongoDB client
        if self.async_mode:
            self.client = AsyncIOMotorClient(self.uri)
            self.db = self.client[self.db_name]
        else:
            self.client = MongoClient(self.uri)
            self.db = self.client[self.db_name]

        # Initialize collections
        self.collections = {}
        self._init_collections()

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file.

        Args:
            config_path: Path to the configuration file.

        Returns:
            Dict containing configuration.
        """
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config

    def _init_collections(self) -> None:
        """
        Initialize MongoDB collections.
        """
        for collection_name, collection_key in self.collections_config.items():
            self.collections[collection_key] = self.db[collection_name]

    def get_collection(self, collection_key: str) -> Optional[Union[Collection, Any]]:
        """
        Get a MongoDB collection.

        Args:
            collection_key: Key of the collection in the configuration.

        Returns:
            MongoDB collection or None if not found.
        """
        if collection_key not in self.collections:
            logger.error(f"Collection key {collection_key} not found")
            return None

        return self.collections[collection_key]

    # Synchronous methods

    def insert_document(self, collection_key: str, document: Dict[str, Any]) -> str:
        """
        Insert a document into a collection.

        Args:
            collection_key: Key of the collection in the configuration.
            document: Document to insert.

        Returns:
            ID of the inserted document.
        """
        if self.async_mode:
            logger.error("Cannot use synchronous method in async mode")
            return None

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return None

        try:
            result = collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            return None

    def insert_documents(self, collection_key: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents into a collection.

        Args:
            collection_key: Key of the collection in the configuration.
            documents: List of documents to insert.

        Returns:
            List of IDs of the inserted documents.
        """
        if self.async_mode:
            logger.error("Cannot use synchronous method in async mode")
            return None

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return None

        try:
            result = collection.insert_many(documents)
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            logger.error(f"Error inserting documents: {str(e)}")
            return None

    def find_document(self, collection_key: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find a document in a collection.

        Args:
            collection_key: Key of the collection in the configuration.
            query: Query to find the document.

        Returns:
            Found document or None.
        """
        if self.async_mode:
            logger.error("Cannot use synchronous method in async mode")
            return None

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return None

        try:
            return collection.find_one(query)
        except Exception as e:
            logger.error(f"Error finding document: {str(e)}")
            return None

    def find_documents(self, collection_key: str, query: Dict[str, Any], limit: int = 0) -> List[Dict[str, Any]]:
        """
        Find documents in a collection.

        Args:
            collection_key: Key of the collection in the configuration.
            query: Query to find the documents.
            limit: Maximum number of documents to return (0 for no limit).

        Returns:
            List of found documents.
        """
        if self.async_mode:
            logger.error("Cannot use synchronous method in async mode")
            return None

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return None

        try:
            cursor = collection.find(query)
            if limit > 0:
                cursor = cursor.limit(limit)
            return list(cursor)
        except Exception as e:
            logger.error(f"Error finding documents: {str(e)}")
            return None

    def update_document(self, collection_key: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """
        Update a document in a collection.

        Args:
            collection_key: Key of the collection in the configuration.
            query: Query to find the document to update.
            update: Update to apply to the document.

        Returns:
            True if the update was successful, False otherwise.
        """
        if self.async_mode:
            logger.error("Cannot use synchronous method in async mode")
            return False

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return False

        try:
            result = collection.update_one(query, {"$set": update})
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False

    def delete_document(self, collection_key: str, query: Dict[str, Any]) -> bool:
        """
        Delete a document from a collection.

        Args:
            collection_key: Key of the collection in the configuration.
            query: Query to find the document to delete.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        if self.async_mode:
            logger.error("Cannot use synchronous method in async mode")
            return False

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return False

        try:
            result = collection.delete_one(query)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

    # Asynchronous methods

    async def insert_document_async(self, collection_key: str, document: Dict[str, Any]) -> str:
        """
        Insert a document into a collection asynchronously.

        Args:
            collection_key: Key of the collection in the configuration.
            document: Document to insert.

        Returns:
            ID of the inserted document.
        """
        if not self.async_mode:
            logger.error("Cannot use asynchronous method in sync mode")
            return None

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return None

        try:
            result = await collection.insert_one(document)
            return str(result.inserted_id)
        except Exception as e:
            logger.error(f"Error inserting document: {str(e)}")
            return None

    async def insert_documents_async(self, collection_key: str, documents: List[Dict[str, Any]]) -> List[str]:
        """
        Insert multiple documents into a collection asynchronously.

        Args:
            collection_key: Key of the collection in the configuration.
            documents: List of documents to insert.

        Returns:
            List of IDs of the inserted documents.
        """
        if not self.async_mode:
            logger.error("Cannot use asynchronous method in sync mode")
            return None

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return None

        try:
            result = await collection.insert_many(documents)
            return [str(id) for id in result.inserted_ids]
        except Exception as e:
            logger.error(f"Error inserting documents: {str(e)}")
            return None

    async def find_document_async(self, collection_key: str, query: Dict[str, Any]) -> Dict[str, Any]:
        """
        Find a document in a collection asynchronously.

        Args:
            collection_key: Key of the collection in the configuration.
            query: Query to find the document.

        Returns:
            Found document or None.
        """
        if not self.async_mode:
            logger.error("Cannot use asynchronous method in sync mode")
            return None

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return None

        try:
            return await collection.find_one(query)
        except Exception as e:
            logger.error(f"Error finding document: {str(e)}")
            return None

    async def find_documents_async(self, collection_key: str, query: Dict[str, Any], limit: int = 0) -> List[Dict[str, Any]]:
        """
        Find documents in a collection asynchronously.

        Args:
            collection_key: Key of the collection in the configuration.
            query: Query to find the documents.
            limit: Maximum number of documents to return (0 for no limit).

        Returns:
            List of found documents.
        """
        if not self.async_mode:
            logger.error("Cannot use asynchronous method in sync mode")
            return None

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return None

        try:
            cursor = collection.find(query)
            if limit > 0:
                cursor = cursor.limit(limit)
            return await cursor.to_list(length=limit if limit > 0 else None)
        except Exception as e:
            logger.error(f"Error finding documents: {str(e)}")
            return None

    async def update_document_async(self, collection_key: str, query: Dict[str, Any], update: Dict[str, Any]) -> bool:
        """
        Update a document in a collection asynchronously.

        Args:
            collection_key: Key of the collection in the configuration.
            query: Query to find the document to update.
            update: Update to apply to the document.

        Returns:
            True if the update was successful, False otherwise.
        """
        if not self.async_mode:
            logger.error("Cannot use asynchronous method in sync mode")
            return False

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return False

        try:
            result = await collection.update_one(query, {"$set": update})
            return result.modified_count > 0
        except Exception as e:
            logger.error(f"Error updating document: {str(e)}")
            return False

    async def delete_document_async(self, collection_key: str, query: Dict[str, Any]) -> bool:
        """
        Delete a document from a collection asynchronously.

        Args:
            collection_key: Key of the collection in the configuration.
            query: Query to find the document to delete.

        Returns:
            True if the deletion was successful, False otherwise.
        """
        if not self.async_mode:
            logger.error("Cannot use asynchronous method in sync mode")
            return False

        collection = self.get_collection(collection_key)
        if collection is None:  # Compare with None instead of truth test
            return False

        try:
            result = await collection.delete_one(query)
            return result.deleted_count > 0
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage (synchronous)
    doc_store = DocumentStore(async_mode=False)

    # Insert a test document
    test_doc = {
        "title": "Test Document",
        "content": "This is a test document",
        "tags": ["test", "example"]
    }

    doc_id = doc_store.insert_document("papers", test_doc)
    print(f"Inserted document with ID: {doc_id}")

    # Find the document
    found_doc = doc_store.find_document("papers", {"title": "Test Document"})
    print(f"Found document: {found_doc}")

    # Update the document
    updated = doc_store.update_document("papers", {"title": "Test Document"}, {"content": "Updated content"})
    print(f"Updated document: {updated}")

    # Find the updated document
    found_doc = doc_store.find_document("papers", {"title": "Test Document"})
    print(f"Found updated document: {found_doc}")

    # Delete the document
    deleted = doc_store.delete_document("papers", {"title": "Test Document"})
    print(f"Deleted document: {deleted}")

    # Example usage (asynchronous)
    async def async_example():
        doc_store = DocumentStore(async_mode=True)

        # Insert a test document
        test_doc = {
            "title": "Async Test Document",
            "content": "This is an async test document",
            "tags": ["test", "async", "example"]
        }

        doc_id = await doc_store.insert_document_async("papers", test_doc)
        print(f"Inserted async document with ID: {doc_id}")

        # Find the document
        found_doc = await doc_store.find_document_async("papers", {"title": "Async Test Document"})
        print(f"Found async document: {found_doc}")

        # Update the document
        updated = await doc_store.update_document_async("papers", {"title": "Async Test Document"}, {"content": "Updated async content"})
        print(f"Updated async document: {updated}")

        # Find the updated document
        found_doc = await doc_store.find_document_async("papers", {"title": "Async Test Document"})
        print(f"Found updated async document: {found_doc}")

        # Delete the document
        deleted = await doc_store.delete_document_async("papers", {"title": "Async Test Document"})
        print(f"Deleted async document: {deleted}")

    # Uncomment to run async example
    # asyncio.run(async_example())
