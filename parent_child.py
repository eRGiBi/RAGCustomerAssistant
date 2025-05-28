import os
import time
import logging
import uuid
import json
from typing import List
from pathlib import Path

import asyncio
from tqdm.asyncio import tqdm

import pandas as pd
from pinecone import Pinecone, ServerlessSpec
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.Logger(__name__)


class ParentChildRetriever():
    """Implementation of a Parent - Child style RAG retriever, 
    with Pinecone."""

    def __init__(
            self,
            embedding_model,
            index_name: str = "parentchild-langchain-document-index",
            chunk_parents: bool = False,
            parent_chunk_size: int = 2000,
            parent_overlap: int = 500,
            child_chunk_size: int = 500,
            child_overlap: int = 100,
            embedding_dimension: int = 512,
            namespace: str = "",
            build_persistent: bool = False,
            build_from_json: bool = False
        ):
        self.index_name = index_name
        self.namespace = namespace

        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        self.embedding_model = embedding_model
        self.embedding_dimension = embedding_dimension

        # The storage layer for the parent documents
        self.chunk_parents = chunk_parents
        self.parent_chunk_size = parent_chunk_size
        self.parent_overlap = parent_overlap
        self.child_chunk_size = child_chunk_size
        self.child_overlap = child_overlap
        self.build_persistent = build_persistent

        logger.info("Parent-Child Initialization ran.")

        # Constructing the vectorstore
        self._build_vector_store(build_from_json)
     
    def _generate_parent_id(self) -> str:
        """Generate a unique ID for a parent document."""
        return f"parent-{str(uuid.uuid4())}"
    
    def _generate_parent_chunk_id(self, parent_id: str, chunk_index: int) -> str:
        """Generate a unique ID for a parent chunk."""
        return f"{parent_id}-pchunk-{chunk_index}"
    
    def _generate_child_id(self, parent_id: str, chunk_index: int) -> str:
        """Generate a unique ID for a child chunk."""
        return f"{parent_id}-child-{chunk_index}"


    def _build_vector_store(self, build_from_json: bool):
        """Initializes the Pinecone Index 
        (connects to or constructs based on the index name)."""

        existing_indexes = [index_info["name"] for index_info in self.pc.list_indexes()]

        if self.index_name not in existing_indexes:
            self.pc.create_index(
                name=self.index_name,
                dimension=self.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws", 
                    region="us-east-1"),
                deletion_protection="enabled",
            )
            while not self.pc.describe_index(self.index_name).status["ready"]:
                time.sleep(1)

        self.child_index = self.pc.Index(self.index_name)

        self.parent_docs = {}
        if build_from_json:
            self._ingest_parents()
            
        logger.info("VS built.")


    def _ingest_parents(self):
        """Load parent documents from disk."""
        if self.chunk_parents:
            raise NotImplementedError
        else:
            with open('parent_store/parents.json', 'r') as j:
                parents_dict = json.load(j)
            for id, data in parents_dict.items():
                self.parent_docs[id] = Document(
                        page_content=data["page_content"],
                        metadata=data["metadata"])
    

    def add_documents(
            self, documents: List[Document], save_parents: bool = False
    ) -> List[str]:
        """
        Pre-process and add documents to the vectorstore, 
        maintaining parent-child relationships.
        """
        logger.info("Starting document addition.")

        parent_ids = []
        all_ids = []
        all_embeddings = []
        all_metadata = []
        all_texts_to_embed = []
        all_metadata_map = {}
        current_child_i = 0
        embedding_batch_size = 100

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size, chunk_overlap=self.child_overlap
        )
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size, chunk_overlap=self.parent_overlap
        )
        
        logger.info("Processing documents and extracting chunks...")
        for doc in documents:
            parent_id = self._generate_parent_id()
            parent_ids.append(parent_id)
            
            self.parent_docs[parent_id] = doc

            if not self.chunk_parents:
                parent_units = [doc.page_content]
                parent_unit_ids = [parent_id]
            else:
                # Chunk the parent document
                parent_chunks = parent_splitter.split_text(doc.page_content)
                parent_units = parent_chunks
                parent_unit_ids = [self._generate_parent_chunk_id(parent_id, i) for i in range(len(parent_chunks))]
                
                # Store the mapping of parent chunks to parent ID
                for i, chunk in enumerate(parent_chunks):
                    chunk_id = parent_unit_ids[i]
                    self.parent_docs[chunk_id] = Document(
                        page_content=chunk,
                        metadata={**doc.metadata, 
                                  "parent_id": parent_id, 
                                  "is_parent_chunk": True, 
                                  "chunk_index": i}
                        )
            
            # Process each parent unit (whole doc or parent chunk)
            for parent_unit, parent_unit_id in zip(parent_units, parent_unit_ids):
                ch_chunks = child_splitter.split_text(parent_unit)
                
                # Create child documents with embeddings
                for i, ch_chunk_text in enumerate(ch_chunks):
                    child_id = self._generate_child_id(parent_unit_id, i)
                    
                    child_metadata = {
                        "text": ch_chunk_text,  
                        "parent_unit_id": parent_unit_id, # ID of immediate parent (whole doc or chunk)
                        "original_parent_id": parent_id, # ID of original parent document
                        "is_chunked_parent": self.chunk_parents,
                        "chunk_index": i,
                        **doc.metadata
                    }

                    all_ids.append(child_id)
                    all_texts_to_embed.append(ch_chunk_text)
                    all_metadata_map[current_child_i] = child_metadata
                    current_child_i += 1
        
        logger.info(f"Creating embeddings for {len(all_texts_to_embed)} chunks...")

        if not all_ids:
            return parent_ids
        
        # Embed documents
        for i in tqdm(
            range(0, len(all_texts_to_embed), embedding_batch_size), 
            desc="Creating embeddings"):
            batch_texts = all_texts_to_embed[i:i+embedding_batch_size]
            batch_embeddings = self.embedding_model.embed_documents(batch_texts)
            all_embeddings.extend(batch_embeddings)
    
        all_metadata = [all_metadata_map[i] for i in range(len(all_metadata_map))]

        df = pd.DataFrame({
            'id': all_ids,
            'values': all_embeddings,
            'metadata': all_metadata
        })
        
        self.child_index.upsert_from_dataframe(
            df=df,
            namespace=self.namespace,
            batch_size=512,
            show_progress=True
        )
        logger.info("Done upserting DataFrame.")

        if save_parents:
            with open(f'parent_store/parents_{self.namespace}.json', 'w', 
                      encoding ='utf8') as f:
                json.dump(
                    {k: {"metadata": v.metadata,
                         "page_content": v.page_content} 
                         for k, v in self.parent_docs.items()},
                    fp=f, 
                    indent = 4
                )
            logger.info("Saved parent documents.")

        return parent_ids
    
    async def aadd_documents(
        self, 
        documents: List[Document], 
        save_parents: bool = False,
        parent_store_path: str = "parent_store"
    ) -> List[str]:
        """
        Process and add documents to the vector store, 
        maintaining parent-child relationships, using asynchronous embedding calls.
        """
        logger.info("Starting async ingestion.")
        parent_ids = []
        all_ids = []
        all_embeddings = []
        all_metadata = []
        all_texts_to_embed = []

        child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.child_chunk_size, chunk_overlap=self.child_overlap
        )
        parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.parent_chunk_size, chunk_overlap=self.parent_overlap
        )
        
        logger.info("Processing documents and extracting chunks...")
        for doc in documents:
            parent_id = self._generate_parent_id()
            parent_ids.append(parent_id)
            
            self.parent_docs[parent_id] = doc

            if not self.chunk_parents:
                parent_units = [doc.page_content]
                parent_unit_ids = [parent_id]
            else:
                # Chunk the parent document
                parent_chunks = parent_splitter.split_text(doc.page_content)
                parent_units = parent_chunks
                parent_unit_ids = [self._generate_parent_chunk_id(parent_id, i) 
                                   for i in range(len(parent_chunks))]
                
                # Store the mapping of parent chunks to parent ID
                for i, chunk in enumerate(parent_chunks):
                    chunk_id = parent_unit_ids[i]
                    self.parent_docs[chunk_id] = Document(
                        page_content=chunk,
                        metadata={**doc.metadata, 
                                  "parent_id": parent_id, 
                                  "is_parent_chunk": True, 
                                  "chunk_index": i}
                        )
            
            # Process each parent unit (whole doc or parent chunk)
            for parent_unit, parent_unit_id in zip(parent_units, parent_unit_ids):
                ch_chunks = child_splitter.split_text(parent_unit)
                
                # Create child documents with embeddings
                for i, ch_chunk_text in enumerate(ch_chunks):
                    child_id = self._generate_child_id(parent_unit_id, i)
                    
                    child_metadata = {
                        "text": ch_chunk_text,  
                        "parent_unit_id": parent_unit_id, # ID of immediate parent (whole doc or chunk)
                        "original_parent_id": parent_id, # ID of original parent document
                        "is_chunked_parent": self.chunk_parents,
                        "chunk_index": i,
                        **doc.metadata
                    }

                    all_ids.append(child_id)
                    all_texts_to_embed.append(ch_chunk_text)
                    all_metadata.append(child_metadata)
        
        if not all_ids:
            logger.error("No document could be processed.")
            return parent_ids

        logger.info(f"Creating embeddings for {len(all_texts_to_embed)} chunks...")

        embedding_batch_size = 100
        concurrency_limit = 5
        semaphore = asyncio.Semaphore(concurrency_limit)
        all_embeddings = [None] * len(all_texts_to_embed)

        # @retry(stop=stop_after_attempt(3), wait=wait_fixed(1))
        async def embed_batch(batch_idx, start_idx):
            # Calculate batch boundaries
            end_idx = min(start_idx + embedding_batch_size, len(all_texts_to_embed))
            batch_texts = all_texts_to_embed[start_idx:end_idx]
            
            # Use semaphore to limit concurrent API calls
            async with semaphore:
                try:
                    batch_embeddings = await self.embedding_model.aembed_documents(
                        batch_texts
                    )
                    for i, embedding in enumerate(batch_embeddings):
                        all_embeddings[start_idx + i] = embedding
                    
                    return batch_idx, True
                except Exception as e:
                    print(f"Error embedding batch {batch_idx}: {e}")
                    return batch_idx, False

        # Create embedding tasks for all batches
        tasks = []
        for batch_idx, start_idx in enumerate(range(
            0, len(all_texts_to_embed), embedding_batch_size)
        ):
            tasks.append(embed_batch(batch_idx, start_idx))
        
        # Execute all embedding tasks concurrently with progress tracking
        results = await tqdm.gather(*tasks, desc="Embedding batches")
        
        # Check for failures
        failures = [idx for idx, success in results if not success]
        if failures:
            print(f"Warning: {len(failures)} batch(es) failed to embed properly.")

        # Check if we have any valid embeddings
        if not any(emb is not None for emb in all_embeddings):
            print("No valid embeddings were created.")
            return parent_ids
        
        logger.info("Creating dataframe and upserting to vector database...")
        valid_entries = [
            (id, embedding, metadata)
            for id, embedding, metadata in zip(all_ids, all_embeddings, all_metadata)
            if embedding is not None
        ]

        valid_ids, valid_embeddings, valid_metadata = zip(*valid_entries)

        df = pd.DataFrame(
            {'id': valid_ids,'values': valid_embeddings, 'metadata': valid_metadata}
        )
        
        self.child_index.upsert_from_dataframe(
            df=df, namespace=self.namespace, batch_size=512, show_progress=True
        )

        logger.info(
            f"Successfully processed {len(valid_entries)} chunks from {len(documents)} documents."
        )

        if save_parents:
            try:
                Path(parent_store_path).mkdir(parents=True, exist_ok=True)
                with open(f'{parent_store_path}/parents_{self.namespace}.json', 'w', encoding ='utf8') as f:
                    json.dump(
                        {k:{"metadata": v.metadata,
                           "page_content": v.page_content}
                           for k, v in self.parent_docs.items()},
                        fp=f, 
                        indent=4
                    )
            except ValueError as e:
                logger.error(f"Value error: {e}")
            except FileNotFoundError as e:
                logger.error(f"File not found.")

            logger.info("Saved parents.")
                
        return parent_ids
    

    def invoke(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve parent documents or chunks based on child chunk retrieval.
        
        Args:
            query: The search query
            top_k: Number of parent documents to return
            
        Returns:
            If chunk_parents=False: List of parent documents
            If chunk_parents=True: List of lists, where each inner list contains chunks of a parent
        """
        results = self.child_index.query(
            namespace=self.namespace,
            vector=[self.embedding_model.embed_query(query)], 
            top_k=top_k, # * 3,
            include_metadata=True,
            include_values=False
        )
                
        # Extract only unique parent IDs
        seen_parent_ids = set()
        retrieved_docs = []

        for result in results["matches"]:
            parent_id = result["metadata"]["original_parent_id"]

            if parent_id and parent_id not in seen_parent_ids:
                if parent_id in self.parent_docs.keys():
                    seen_parent_ids.add(parent_id)
                    retrieved_docs.append(self.parent_docs[parent_id])
                    
                    # Limit to only k parents (if we query more)
                    # if len(retrieved_docs) == top_k:
                        # break
        
        return retrieved_docs
   
    def describe(self):
        return self.child_index.describe_index_stats()

    def delete_namespace(self):
        self.child_index.delete(delete_all=True, namespace=self.namespace)


def estimate_batch_size(batch: List[str]) -> int:
    """Estimate the size of the batch in bytes."""
    return sum(len(item.encode('utf-8')) for item in batch)
