"""
web_search.py – Secure web search adapter
Uses security manager for credential retrieval
"""

import time
from typing import List
import googlesearch
from core.telemetry import get_logger
from security.security_manager import retrieve_credential  # Security integration

LOGGER = get_logger("web_search")

def perform_search(query: str, num_results: int = 5) -> List[str]:
    """Perform a web search and return URLs"""
    try:
        # Add a short delay to prevent rate limiting
        time.sleep(1)

        # Get credentials if needed (future-proofing)
        credentials = retrieve_credential("web_search", "default_user")

        # Perform the search and return URLs
        return list(googlesearch.search(
            query,
            num_results=num_results,
            advanced=False,
            lang="en",
            # Additional parameters would go here
        ))
    except Exception as e:
        LOGGER.error("Search failed", query=query, error=str(e))
        raise RuntimeError(f"Search failed: {str(e)}") from e

async def execute(parameters: dict) -> List[str]:
    """
    Adapter execute function for web_search tool

    Parameters:
        query: Search query string
        num_results: Number of results to return (default: 5)

    Returns:
        List of URLs
    """
    # Validate required parameters
    if "query" not in parameters:
        raise ValueError("Missing required parameter: query")

    # Get parameters with defaults
    query = parameters["query"]
    num_results = parameters.get("num_results", 5)

    # Log the search request
    LOGGER.info("Performing web search", query=query, num_results=num_results)

    # Perform the search
    results = perform_search(query, num_results)

    # Log the results
    LOGGER.debug("Search completed", result_count=len(results))
    return results