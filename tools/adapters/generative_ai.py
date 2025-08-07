"""
generative_ai.py – Secure AI text generation adapter
Uses the security manager for credential retrieval
"""

import httpx
import logging
from typing import Dict, Any
from core.telemetry import get_logger
from security.security_manager import retrieve_credential

LOGGER = get_logger("generative_ai")

# Configuration for different AI providers
PROVIDER_ENDPOINTS = {
    "openai": "https://api.openai.com/v1/chat/completions",
    "gemini": "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent",
    "deepseek": "https://api.deepseek.com/v1/chat/completions",
    "mistral": "https://api.mistral.ai/v1/chat/completions",
    "anthropic": "https://api.anthropic.com/v1/messages",
    "cohere": "https://api.cohere.ai/v1/generate",
}

# Default models for each provider
PROVIDER_MODELS = {
    "openai": "gpt-4-turbo",
    "gemini": "gemini-pro",
    "deepseek": "deepseek-chat",
    "mistral": "mistral-large-latest",
    "anthropic": "claude-3-opus-20240229",
    "cohere": "command-r-plus",
}

async def generate_text(provider: str, prompt: str) -> str:
    """Generate text using the specified provider"""
    endpoint = PROVIDER_ENDPOINTS.get(provider)
    if not endpoint:
        raise ValueError(f"Unsupported provider: {provider}")

    # Retrieve API key from security manager
    api_key = retrieve_credential(provider, "default_user")
    if not api_key:
        raise RuntimeError(f"API key for {provider} not found in credential store")

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    # Special header for Gemini
    if provider == "gemini":
        headers = {"Content-Type": "application/json"}

    # Format request based on provider
    payload = {}
    if provider in ["openai", "deepseek", "mistral"]:
        payload = {
            "model": PROVIDER_MODELS[provider],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 1000
        }
    elif provider == "gemini":
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000
            }
        }
    elif provider == "anthropic":
        payload = {
            "model": PROVIDER_MODELS[provider],
            "max_tokens": 1000,
            "messages": [{"role": "user", "content": prompt}]
        }
    elif provider == "cohere":
        payload = {
            "model": PROVIDER_MODELS[provider],
            "prompt": prompt,
            "max_tokens": 1000,
            "temperature": 0.7
        }

    try:
        async with httpx.AsyncClient() as client:
            # Special handling for Gemini
            if provider == "gemini":
                endpoint = f"{endpoint}?key={api_key}"

            response = await client.post(
                endpoint,
                json=payload,
                headers=headers,
                timeout=60.0
            )
            response.raise_for_status()
            response_data = response.json()

            # Extract response based on provider
            if provider in ["openai", "deepseek", "mistral"]:
                return response_data["choices"][0]["message"]["content"]
            elif provider == "gemini":
                return response_data["candidates"][0]["content"]["parts"][0]["text"]
            elif provider == "anthropic":
                return response_data["content"][0]["text"]
            elif provider == "cohere":
                return response_data["generations"][0]["text"]

    except Exception as e:
        LOGGER.error("Text generation failed",
                    provider=provider,
                    error=str(e),
                    status_code=response.status_code if 'response' in locals() else None)
        raise RuntimeError(f"Text generation failed: {str(e)}") from e

async def execute(parameters: dict) -> str:
    """
    Adapter execute function for generative_ai tool

    Parameters:
        prompt: Text prompt for generation
        provider: AI provider (openai, gemini, etc.)

    Returns:
        Generated text
    """
    # Validate required parameters
    if "prompt" not in parameters:
        raise ValueError("Missing required parameter: prompt")

    # Get parameters with defaults
    prompt = parameters["prompt"]
    provider = parameters.get("provider", "openai").lower()

    # Validate provider
    if provider not in PROVIDER_ENDPOINTS:
        raise ValueError(f"Unsupported provider: {provider}. Valid options: {', '.join(PROVIDER_ENDPOINTS.keys())}")

    # Log the generation request
    LOGGER.info("Generating text",
                provider=provider,
                prompt_length=len(prompt),
                model=PROVIDER_MODELS.get(provider, "unknown"))

    # Generate text
    result = await generate_text(provider, prompt)

    # Log the result
    LOGGER.debug("Text generation completed",
                 result_length=len(result),
                 provider=provider)
    return result