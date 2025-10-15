#!/usr/bin/env python3
"""
LocalCorpus: In-memory BM25 corpus using Hugging Face datasets.
"""

from typing import List, Optional, Any, cast
import logging
import json
import faiss
import numpy as np

import bm25s
import Stemmer
from datasets import load_dataset, Dataset

from embedding_providers.azure_oai_embedding import get_azure_openai_client

logger = logging.getLogger(__name__)


def normalize(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


class LocalCorpus:
    """
    In-memory BM25 corpus that loads data from Hugging Face dataset
    withpi/nlweb_allsites and provides search functionality.
    """

    def __init__(
        self,
        dataset_name: str = "withpi/nlweb_allsites",
        split: str = "train",
        stemmer_language: str = "english",
        stopwords: str = "en",
    ):
        """
        Initialize the LocalCorpus by loading and indexing the dataset.

        Args:
            dataset_name: Hugging Face dataset identifier
            split: Dataset split to load (default: "train")
            stemmer_language: Language for stemming (default: "english")
            stopwords: Stopwords language code (default: "en")
        """
        self.dataset_name = dataset_name
        self.split = split
        self.stemmer_language = stemmer_language
        self.stopwords = stopwords

        # Initialize components
        self.stemmer = Stemmer.Stemmer(stemmer_language)
        self.retriever: Optional[bm25s.BM25] = None
        self.corpus: List[str] = []
        self.dataset: Optional[Dataset] = None

        # Load and index the dataset
        self._load_dataset()
        self._build_index()

    def _load_dataset(self) -> None:
        """Load the dataset from Hugging Face."""
        logger.info(f"Loading dataset {self.dataset_name} split={self.split}")

        dataset_raw = load_dataset(self.dataset_name, split=self.split)
        self.dataset = cast(Dataset, dataset_raw)

        # Extract text corpus (adjust field name based on actual dataset structure)
        # Assuming the dataset has a 'text' or similar field
        # Build corpus from dataset - may need adjustment based on actual schema
        self.corpus_text = []
        self.corpus_structured = []
        self.embeddings = []

        for item in self.dataset:
            self.corpus_text.append(str(item["schema_object"]))
            json_obj = json.loads(item["schema_object"])
            self.corpus_structured.append(
                (item["url"], item["schema_object"], json_obj["name"], "nlweb_sites")
            )
        self.embeddings = normalize(np.array(self.dataset["schema_object_embedding"]))

        self.non_shopify_dataset = self.dataset.filter(
            lambda ex: "myshopify.com" not in ex["url"]
        )
        self.non_shopify_corpus_text = []
        self.non_shopify_corpus_structured = []
        self.non_shopify_embeddings = []
        for item in self.non_shopify_dataset:
            self.non_shopify_corpus_text.append(str(item["schema_object"]))
            json_obj = json.loads(item["schema_object"])
            self.non_shopify_corpus_structured.append(
                (item["url"], item["schema_object"], json_obj["name"], "nlweb_sites")
            )
        self.non_shopify_embeddings = normalize(
            np.array(self.non_shopify_dataset["schema_object_embedding"])
        )

        logger.info(f"Loaded {len(self.corpus_text)} documents")

    def _build_index(self) -> None:
        """Tokenize corpus and build BM25 index."""
        logger.info("Building BM25 index...")

        # Tokenize the corpus
        corpus_tokens = bm25s.tokenize(
            self.corpus_text, stopwords=self.stopwords, stemmer=self.stemmer
        )

        # Create and index the BM25 model
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        self.retriever = retriever

        # Tokenize the corpus
        non_shopify_corpus_tokens = bm25s.tokenize(
            self.non_shopify_corpus_text, stopwords=self.stopwords, stemmer=self.stemmer
        )

        # Create and index the BM25 model
        non_shopify_retriever = bm25s.BM25()
        non_shopify_retriever.index(non_shopify_corpus_tokens)
        self.non_shopify_retriever = non_shopify_retriever

        logger.info("BM25 index built successfully")

        faiss.omp_set_num_threads(1)

        hnsw_M = 16
        hnsw_efConstruction = 128
        hnsw_efSearch = 100

        self.index = faiss.IndexHNSWFlat(1024, hnsw_M, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efConstruction = hnsw_efConstruction
        self.index.hnsw.efSearch = hnsw_efSearch

        self.index.add(self.embeddings)

        self.non_shopify_index = faiss.IndexHNSWFlat(
            1024, hnsw_M, faiss.METRIC_INNER_PRODUCT
        )
        self.non_shopify_index.hnsw.efConstruction = hnsw_efConstruction
        self.non_shopify_index.hnsw.efSearch = hnsw_efSearch

        self.non_shopify_index.add(self.non_shopify_embeddings)

    async def internal_search(
        self, query: str, retriever, embedding_index, corpus_structured, k: int = 10
    ) -> List[dict[str, Any]]:
        """
        Search the corpus for documents matching the query.

        Args:
            query: Search query string
            k: Number of top results to return (default: 10)

        Returns:
            List of dicts with keys: rank, score, docid, doc
        """
        if retriever is None:
            raise RuntimeError("Index not built. Cannot search.")

        # Tokenize query
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)

        client = get_azure_openai_client()

        response = await client.embeddings.create(
            # input="helpful websites for my search: " + query,
            input=query,
            model="text-embedding-3-large",
            dimensions=1024,
        )
        embedding = [response.data[0].embedding]
        query_embeddings = normalize(np.array(embedding))
        _, embedding_results = embedding_index.search(np.array(query_embeddings), k=k)

        # Retrieve top-k results
        bm25_results, scores = retriever.retrieve(query_tokens, k=k)

        results = list(set(bm25_results[0]).union(set(embedding_results[0])))

        # Format results
        rows = []
        for docid in results:
            # score = float(scores[0, i])
            rows.append(corpus_structured[docid])

        return rows

    async def search(self, query: str, k: int = 10) -> List[dict[str, Any]]:
        non_shopify_items = await self.internal_search(
            query=query,
            retriever=self.non_shopify_retriever,
            embedding_index=self.non_shopify_index,
            corpus_structured=self.non_shopify_corpus_structured,
            k=k,
        )

        all_items = await self.internal_search(
            query=query,
            retriever=self.retriever,
            embedding_index=self.index,
            corpus_structured=self.corpus_structured,
            k=k,
        )

        merged = non_shopify_items + all_items
        return list({x[0]: x for x in merged}.values())

    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.corpus)

    def __repr__(self) -> str:
        return f"LocalCorpus(dataset={self.dataset_name}, docs={len(self.corpus)})"


# Initialize a global instance of LocalCorpus
LOCAL_CORPUS = LocalCorpus()
