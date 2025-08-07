"""
Example tool adapter for web scraping.
"""

import asyncio
from typing import Any, Dict
import httpx


async def execute(parameters: Dict[str, Any], http_client: httpx.AsyncClient = None) -> Dict[str, Any]:
    """
    Execute web scraping tool.
    
    Args:
        parameters: Tool parameters including 'url' and optional 'selector'
        http_client: Optional shared HTTP client
        
    Returns:
        Dictionary with scraped content and metadata
    """
    url = parameters.get("url")
    selector = parameters.get("selector")
    timeout = parameters.get("timeout", 30)
    
    if not url:
        raise ValueError("URL parameter is required")
    
    # Use provided client or create new one
    client = http_client or httpx.AsyncClient(timeout=timeout)
    
    try:
        response = await client.get(url)
        response.raise_for_status()
        
        content = response.text
        
        # Basic content extraction (in real implementation, use BeautifulSoup)
        if selector:
            # Placeholder for CSS selector implementation
            extracted_content = f"Content from {selector}: {content[:200]}..."
        else:
            extracted_content = content[:500] + "..." if len(content) > 500 else content
        
        return {
            "success": True,
            "url": url,
            "content": extracted_content,
            "content_length": len(content),
            "status_code": response.status_code,
            "headers": dict(response.headers)
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": url
        }
    finally:
        # Only close if we created the client
        if not http_client:
            await client.aclose() 