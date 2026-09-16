"""Groq LPU Client Wrapper for Ultra-Fast LLM Inference.
Equipped with Circuit Breaker, configurable timeouts, and graceful offline fallback.
"""
import time
import os
from typing import List, Dict, Any, Optional
from src.config import settings

class GroqClientManager:
    """Singleton Manager for Groq API interactions."""

    def __init__(self):
        self.api_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY")
        self.client = None
        self.timeout = settings.GROQ_TIMEOUT_SECONDS
        self._failure_count = 0
        self._circuit_open_until = 0.0
        self._init_client()

    def _init_client(self):
        """Initialize the Groq SDK client."""
        if not self.api_key:
            return
        try:
            from groq import Groq
            self.client = Groq(api_key=self.api_key, timeout=self.timeout)
            self._record_success()
        except Exception as e:
            self.client = None
            self._record_failure()

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is currently preventing API calls."""
        return time.time() < self._circuit_open_until

    def _record_success(self):
        """Reset failure tracking on successful call."""
        self._failure_count = 0
        self._circuit_open_until = 0.0

    def _record_failure(self):
        """Trip circuit breaker after 3 consecutive failures."""
        self._failure_count += 1
        if self._failure_count >= 3:
            self._circuit_open_until = time.time() + 30.0

    def is_available(self) -> bool:
        """Check if Groq service is configured and ready."""
        return bool(self.client is not None and not self._is_circuit_open())

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
        response_format: Optional[Dict[str, str]] = None
    ) -> Optional[str]:
        """Perform chat completion with automatic circuit breaker and fallback."""
        if not self.is_available():
            return None

        target_model = model or settings.GROQ_CHAT_MODEL
        kwargs: Dict[str, Any] = {
            "model": target_model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if response_format:
            kwargs["response_format"] = response_format

        try:
            completion = self.client.chat.completions.create(**kwargs)
            self._record_success()
            if completion.choices and len(completion.choices) > 0:
                return completion.choices[0].message.content
            return None
        except Exception:
            self._record_failure()
            return None

groq_client = GroqClientManager()
