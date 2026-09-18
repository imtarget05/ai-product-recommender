"""Local Ollama Client for RecSys-AI (M1 Pro 16GB plan, opt-in).

Uses httpx (already in requirements) — POST /api/chat with num_ctx 4096.
No `ollama` package needed. When OLLAMA_URL is unset or unreachable the
caller falls back to Groq, then to heuristic templates.
"""
import time
from typing import List, Dict, Any, Optional
import httpx
from src.config import settings


class OllamaClientManager:
    """Minimal Ollama chat client with health-probe cache + failure backoff."""

    def __init__(self):
        self.base_url = (settings.OLLAMA_URL or "").rstrip("/")
        self.model = settings.OLLAMA_CHAT_MODEL
        self.num_ctx = settings.OLLAMA_NUM_CTX
        self.timeout = settings.OLLAMA_TIMEOUT_SECONDS
        self._healthy_until = 0.0
        self._fail_until = 0.0

    @property
    def enabled(self) -> bool:
        return bool(self.base_url)

    def is_available(self) -> bool:
        """Cached health probe: True only if enabled and /api/tags answers."""
        if not self.enabled or time.time() < self._fail_until:
            return False
        if time.time() < self._healthy_until:
            return True
        try:
            resp = httpx.get(f"{self.base_url}/api/tags", timeout=2.0)
            healthy = resp.status_code == 200
        except Exception:
            healthy = False
        if healthy:
            self._healthy_until = time.time() + 10.0
        else:
            self._fail_until = time.time() + 10.0
        return healthy

    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 1024,
    ) -> Optional[str]:
        """POST /api/chat (non-streaming). Returns content or None on any error."""
        if not self.is_available():
            return None
        body: Dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
            "stream": False,
            "think": False,
            "keep_alive": "5m",
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,
                "num_predict": max_tokens,
            },
        }
        try:
            resp = httpx.post(
                f"{self.base_url}/api/chat", json=body, timeout=self.timeout
            )
            resp.raise_for_status()
            payload = resp.json()
            content = ((payload.get("message") or {}).get("content") or "").strip()
            if not content:
                self._fail_until = time.time() + 10.0
                return None
            return content
        except Exception:
            self._fail_until = time.time() + 10.0
            return None


ollama_client = OllamaClientManager()
