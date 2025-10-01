# Copyright (c) 2025 Microsoft Corporation.
# Licensed under the MIT License

"""
Pi Labs embedding implementation.

WARNING: This code is under development and may undergo changes in future releases.
Backwards compatibility is not guaranteed at this time.
"""

import os
import threading
from typing import List

from withpi import AsyncPiClient
from core.config import CONFIG

from misc.logger.logging_config_helper import get_configured_logger, LogLevel
logger = get_configured_logger("openai_embedding")

# Add lock for thread-safe client access
_client_lock = threading.Lock()
pi_labs_client = None

def get_pi_labs_api_key() -> str:
    """
    Retrieve the Pi Labs API key from configuration.
    """
    # Get the API key from the embedding provider config
    provider_config = CONFIG.get_embedding_provider("pi_labs")
    if provider_config and provider_config.api_key:
        api_key = provider_config.api_key
        if api_key:
            return api_key
    
    # Fallback to environment variable
    api_key = os.getenv("WITHPI_API_KEY")
    if not api_key:
        error_msg = "Pi Labs API key not found in configuration or environment"
        logger.error(error_msg)
        raise ValueError(error_msg)
    
    return api_key

def get_async_client() -> AsyncPiClient:
    """
    Configure and return an asynchronous Pi Labs client.
    """
    global pi_labs_client
    with _client_lock:  # Thread-safe client initialization
        if pi_labs_client is None:
            try:
                api_key = get_pi_labs_api_key()
                pi_labs_client = AsyncPiClient(api_key=api_key)
                logger.debug("Pi Labs client initialized successfully")
            except Exception:
                logger.exception("Failed to initialize Pi Labs client")
                raise

    return pi_labs_client

async def get_pi_labs_embeddings(
    text: str,
) -> List[float]:
    """
    Generate an embedding for a single text using Pi Labs API.
    
    Args:
        text: The text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    logger.debug("Generating Pi Labs embeddings")
    logger.debug(f"Text length: {len(text)} chars")
    
    client = get_async_client()

    try:
        # Clean input text (replace newlines with spaces)
        text = text.replace("\n", " ")
        
        response = await client.search.embed(
            query=[text],
        )
        
        embedding = response[0]
        logger.debug(f"Pi Labs embedding generated, dimension: {len(embedding)}")
        return embedding
    except Exception as e:
        logger.exception("Error generating Pi Labs embedding")
        logger.log_with_context(
            LogLevel.ERROR,
            "Pi Labs embedding generation failed",
            {
                "text_length": len(text),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        raise

async def get_pi_labs_batch_embeddings(
    texts: List[str]
) -> List[List[float]]:
    """
    Generate embeddings for multiple texts using OpenAI API.
    
    Args:
        texts: List of texts to embed
        
    Returns:
        List of embedding vectors, each a list of floats
    """
    # If model not provided, get it from config
    logger.debug("Generating Pi Labs batch embeddings")
    logger.debug(f"Batch size: {len(texts)} texts")
    
    client = get_async_client()

    try:
        # Clean input texts (replace newlines with spaces)
        cleaned_texts = [text.replace("\n", " ") for text in texts]
        
        response = await client.search.embed(
            query=cleaned_texts,
        )
        
        logger.debug(f"Pi Labs batch embeddings generated, count: {len(response)}")
        return response
    except Exception as e:
        logger.exception("Error generating Pi Labs batch embeddings")
        logger.log_with_context(
            LogLevel.ERROR,
            "Pi Labs batch embedding generation failed",
            {
                "batch_size": len(texts),
                "error_type": type(e).__name__,
                "error_message": str(e)
            }
        )
        raise
