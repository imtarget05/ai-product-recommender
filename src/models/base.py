"""Abstract Base Class for all Recommender Systems."""
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from src.database.models import Product, Interaction

class BaseRecommender(ABC):
    """Abstract interface for all recommendation algorithms."""

    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False

    @abstractmethod
    def fit(self, products: List[Product], interactions: List[Interaction]) -> None:
        """Fit or train recommendation model using products and interactions."""
        pass

    @abstractmethod
    def recommend(
        self,
        user_id: int,
        top_k: int = 10,
        exclude_interacted: bool = True
    ) -> List[Dict[str, Any]]:
        """Generate top-K recommended items for a user.

        Returns a list of dicts with keys:
            - product_id: int
            - score: float (0.0 to 1.0)
            - model: str
            - reason: str (human readable explanation)
        """
        pass

    def similar_items(
        self,
        product_id: int,
        top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Return similar products given an item id."""
        return []
