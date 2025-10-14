#!/usr/bin/env python3
"""
LocalCorpus: In-memory BM25 corpus using Hugging Face datasets.
"""

import sys
from typing import List, Optional, Any, cast
import logging
import json

import bm25s
import Stemmer
from datasets import load_dataset, Dataset

logger = logging.getLogger(__name__)

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
        stopwords: str = "en"
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
        for item in self.dataset:
            self.corpus_text.append(str(item["schema_object"]))
            json_obj = json.loads(item["schema_object"])
            self.corpus_structured.append((item["url"], item["schema_object"], json_obj["name"], "nlweb_sites"))
        logger.info(f"Loaded {len(self.corpus_text)} documents")

    def _build_index(self) -> None:
        """Tokenize corpus and build BM25 index."""
        logger.info("Building BM25 index...")

        # Tokenize the corpus
        corpus_tokens = bm25s.tokenize(
            self.corpus_text,
            stopwords=self.stopwords,
            stemmer=self.stemmer
        )

        # Create and index the BM25 model
        retriever = bm25s.BM25()
        retriever.index(corpus_tokens)
        self.retriever = retriever

        logger.info("BM25 index built successfully")

    def search(
        self,
        query: str,
        k: int = 10
    ) -> List[dict[str, Any]]:
        """
        Search the corpus for documents matching the query.

        Args:
            query: Search query string
            k: Number of top results to return (default: 10)

        Returns:
            List of dicts with keys: rank, score, docid, doc
        """
        if self.retriever is None:
            raise RuntimeError("Index not built. Cannot search.")

        # Tokenize query
        query_tokens = bm25s.tokenize(query, stemmer=self.stemmer)

        # Retrieve top-k results
        results, scores = self.retriever.retrieve(query_tokens, k=k)

        # Format results
        rows = []
        for i in range(results.shape[1]):
            docid = int(results[0, i])
            #score = float(scores[0, i])
            rows.append(self.corpus_structured[docid])

        return rows

    def __len__(self) -> int:
        """Return the number of documents in the corpus."""
        return len(self.corpus)

    def __repr__(self) -> str:
        return f"LocalCorpus(dataset={self.dataset_name}, docs={len(self.corpus)})"

# Initialize a global instance of LocalCorpus
LOCAL_CORPUS = LocalCorpus()
