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
from collections import defaultdict
from tqdm import tqdm

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

        # TODO: clean this up
        auxiliary_metadata_ds = load_dataset(self.dataset_name, "metadata", split="train")
        metadata = {}
        for row in auxiliary_metadata_ds:
            url = row["url"]
            is_shopify = "myshopify.com" in url
            if is_shopify and row["shopping_cat_parsed"]:
                tlc = list(json.loads(row["shopping_cat_parsed"]).keys())
            elif row["pi_cat_1"]:
                tlc = row["pi_cat_1"]
            else:
                tlc = []
            assert(url not in metadata)
            metadata[url] = {"is_shopify": is_shopify, "top_level_categories": tlc}

        self.corpus_text_map = defaultdict(list)
        self.corpus_structured_map = defaultdict(list)
        self.embeddings_map = defaultdict(list)

        for item in tqdm(self.dataset):
            json_obj = json.loads(item["schema_object"])
            self.corpus_text_map["ALL"].append(str(item["schema_object"]))
            self.corpus_structured_map["ALL"].append(
                (item["url"], item["schema_object"], json_obj["name"], "nlweb_sites")
            )
            self.embeddings_map["ALL"].append(item["schema_object_embedding"])

            if "myshopify.com" not in item["url"]:
                self.corpus_text_map["NON_SHOPIFY"].append(str(item["schema_object"]))
                self.corpus_structured_map["NON_SHOPIFY"].append(
                    (
                        item["url"],
                        item["schema_object"],
                        json_obj["name"],
                        "nlweb_sites",
                    )
                )
                self.embeddings_map["NON_SHOPIFY"].append(
                    item["schema_object_embedding"]
                )
                top_level_categories = metadata[item["url"]]["top_level_categories"]
            else:
                top_level_categories = metadata[item["url"]]["top_level_categories"]

            for category in top_level_categories:
                json_obj = json.loads(item["schema_object"])
                self.corpus_text_map[category].append(str(item["schema_object"]))
                self.corpus_structured_map[category].append(
                    (
                        item["url"],
                        item["schema_object"],
                        json_obj["name"],
                        "nlweb_sites",
                    )
                )
                self.embeddings_map[category].append(item["schema_object_embedding"])

        for category in self.embeddings_map.keys():
            print(f"{category}")
            self.embeddings_map[category] = normalize(
                np.array(self.embeddings_map[category])
            )

        logger.info(f"Loaded {len(self.dataset)} documents")
        logger.info(f"Loaded {len(self.corpus_text_map)} sub-corpora")

    def _build_index(self) -> None:
        """Tokenize corpus and build BM25 index."""
        logger.info("Building BM25 index...")

        self.retriever_map = {}

        for category in self.corpus_text_map.keys():
            # Tokenize the corpus
            corpus_tokens = bm25s.tokenize(
                self.corpus_text_map[category],
                stopwords=self.stopwords,
                stemmer=self.stemmer,
            )

            # Create and index the BM25 model
            retriever = bm25s.BM25()
            retriever.index(corpus_tokens)
            self.retriever_map[category] = retriever

        logger.info("BM25 index built successfully")

        faiss.omp_set_num_threads(1)

        hnsw_M = 16
        hnsw_efConstruction = 128
        hnsw_efSearch = 100

        self.index_map = {}
        for category in self.corpus_text_map.keys():
            self.index_map[category] = faiss.IndexHNSWFlat(
                1024, hnsw_M, faiss.METRIC_INNER_PRODUCT
            )
            self.index_map[category].hnsw.efConstruction = hnsw_efConstruction
            self.index_map[category].hnsw.efSearch = hnsw_efSearch

            self.index_map[category].add(self.embeddings_map[category])

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

        # --- Get corpus size and enforce bounds
        corpus_size = retriever.scores["num_docs"]
        if corpus_size is None:
            raise RuntimeError("Cannot determine corpus size from retriever.")

        # Print corpus size every time
        print(f"[internal_search] Corpus size: {corpus_size}")

        bm25_k = k
        if bm25_k > corpus_size:
            print(f"[internal_search] Reducing bm25_k from {k} to {corpus_size}")
            bm25_k = corpus_size


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
        bm25_results, scores = retriever.retrieve(query_tokens, k=bm25_k)

        results = list(set(bm25_results[0]).union(set(embedding_results[0])))

        # Format results
        rows = []
        for docid in results:
            # score = float(scores[0, i])
            rows.append(corpus_structured[docid])

        return rows

    async def search(
        self, query: str, categories: list[str], k: int = 10
    ) -> List[dict[str, Any]]:
        items = []
        for category in categories:
            items.extend(
                await self.internal_search(
                    query=query,
                    retriever=self.retriever_map[category],
                    embedding_index=self.index_map[category],
                    corpus_structured=self.corpus_structured_map[category],
                    k=k,
                )
            )
        return list({x[0]: x for x in items}.values())

    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.corpus)

    def __repr__(self) -> str:
        return f"LocalCorpus(dataset={self.dataset_name}, docs={len(self.corpus)})"


# Initialize a global instance of LocalCorpus
LOCAL_CORPUS = LocalCorpus()
