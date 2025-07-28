from openai import OpenAI, AsyncOpenAI
import os
import logging
import asyncio
from typing import List, Dict, Any, Optional
from tenacity import (
    retry,
    stop_after_attempt,
    wait_fixed,
)


def before_retry_fn(retry_state):
    """Log retry attempts for debugging."""
    if retry_state.attempt_number > 1:
        logging.info(f"Retrying API call. Attempt #{retry_state.attempt_number}")


class LLMClient:
    """Basic OpenAI client with retry logic."""
    
    def __init__(self, model: str = "gpt-4"):
        """Initialize OpenAI client."""
        self.model = model
        api_key = os.environ.get("OPENAI_API_KEY")
        
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment variables")
        
        self.client = OpenAI(api_key=api_key)
        self.async_client = AsyncOpenAI(api_key=api_key)
    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(10), before=before_retry_fn)
    def generate_response(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Generate synchronous response with retry logic."""
        try:
            response = self.client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4000),
                timeout=kwargs.get("timeout", 180)
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM response failed: {e}")
            return None
    
    @retry(wait=wait_fixed(10), stop=stop_after_attempt(10), before=before_retry_fn)
    async def generate_response_async(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        """Generate async response with retry logic."""
        try:
            response = await self.async_client.chat.completions.create(
                model=kwargs.get("model", self.model),
                messages=messages,
                temperature=kwargs.get("temperature", 0.7),
                max_tokens=kwargs.get("max_tokens", 4000),
                timeout=kwargs.get("timeout", 180)
            )
            return response.choices[0].message.content
        except Exception as e:
            logging.error(f"LLM async response failed: {e}")
            await asyncio.sleep(1)
            return None


def create_llm_client(model: str = "gpt-4") -> LLMClient:
    """
    Factory function to create LLM client with default configuration.
    
    Args:
        model (str): Model name
    
    Returns:
        LLMClient: Configured LLM client
    """
    return LLMClient(model=model)


def load_llm_config_from_env() -> Dict[str, Any]:
    """
    Load LLM configuration from environment variables.
    
    Returns:
        dict: Configuration dictionary
    """
    return {
        "main_model": os.environ.get("MAIN_LLM_MODEL", "gpt-4"),
        "cheap_model": os.environ.get("CHEAP_LLM_MODEL", "gpt-3.5-turbo"),
        "temperature": float(os.environ.get("LLM_TEMPERATURE", "0.7")),
        "max_tokens": int(os.environ.get("LLM_MAX_TOKENS", "4000")),
    }


def validate_llm_config() -> bool:
    """
    Validate that required LLM configuration is present.
    
    Returns:
        bool: True if configuration is valid
    """
    required_vars = ["OPENAI_API_KEY"]
    
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logging.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    return True 