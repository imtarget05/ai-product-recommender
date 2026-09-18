"""Boundary for acknowledged cart mutations.

The recommendation service may propose cart work, but only a commerce
adapter acknowledgement permits it to report a completed add-to-cart action.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Protocol
from uuid import uuid4


class CartActionStatus(str, Enum):
    PROPOSED = "PROPOSED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class CartActionResult:
    status: CartActionStatus
    action_id: str | None
    product_id: int
    detail: str


class CartGateway(Protocol):
    """A commerce adapter that durably accepts idempotent cart writes."""

    requires_idempotency_key: bool

    def add_item(self, user_id: int, product_id: int, idempotency_key: str) -> CartActionResult:
        """Submit a cart mutation and return the commerce system's acknowledgement."""


class DisabledCartGateway:
    """Safe default when this service has no configured commerce adapter."""

    requires_idempotency_key = False

    def add_item(self, user_id: int, product_id: int, idempotency_key: str) -> CartActionResult:
        return CartActionResult(
            status=CartActionStatus.PROPOSED,
            action_id=None,
            product_id=product_id,
            detail="Cart integration is not configured; confirm this proposal in the commerce system.",
        )


class InMemoryCartGateway:
    """Acknowledging gateway for local tests; never configure it for production."""

    requires_idempotency_key = True

    def __init__(self) -> None:
        self.actions: dict[str, CartActionResult] = {}

    def add_item(self, user_id: int, product_id: int, idempotency_key: str) -> CartActionResult:
        existing = self.actions.get(idempotency_key)
        if existing:
            return existing

        result = CartActionResult(
            status=CartActionStatus.COMPLETED,
            action_id=str(uuid4()),
            product_id=product_id,
            detail="Cart item acknowledged by the configured commerce gateway.",
        )
        self.actions[idempotency_key] = result
        return result
