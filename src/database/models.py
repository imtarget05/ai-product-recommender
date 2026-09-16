"""SQLAlchemy Database Models for RecSys-AI."""
from datetime import datetime
from typing import Optional
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    Text,
    DateTime,
    ForeignKey,
    Index,
    func
)
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


class Product(Base):
    """Product Catalog Item."""
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, autoincrement=True)
    title = Column(String(255), nullable=False, index=True)
    category = Column(String(100), nullable=False, index=True)
    price = Column(Float, nullable=False, default=0.0)
    description = Column(Text, nullable=True)
    image_url = Column(String(500), nullable=True)
    tags = Column(String(255), nullable=True)
    rating_avg = Column(Float, default=0.0)
    rating_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

    interactions = relationship("Interaction", back_populates="product", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "category": self.category,
            "price": self.price,
            "description": self.description,
            "image_url": self.image_url,
            "tags": self.tags,
            "rating_avg": self.rating_avg,
            "rating_count": self.rating_count,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class User(Base):
    """User profile."""
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String(100), unique=True, nullable=False, index=True)
    email = Column(String(150), nullable=True)
    segment = Column(String(50), default="general")
    created_at = Column(DateTime, default=datetime.utcnow)

    interactions = relationship("Interaction", back_populates="user", cascade="all, delete-orphan")
    search_logs = relationship("SearchLog", back_populates="user", cascade="all, delete-orphan")

    def to_dict(self):
        return {
            "id": self.id,
            "username": self.username,
            "email": self.email,
            "segment": self.segment,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Interaction(Base):
    """User behavior interactions: view, click, add_to_cart, purchase, rating."""
    __tablename__ = "interactions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True)
    product_id = Column(Integer, ForeignKey("products.id", ondelete="CASCADE"), nullable=False, index=True)
    event_type = Column(String(50), nullable=False)  # view, click, add_to_cart, purchase, rating
    rating_value = Column(Float, nullable=True)     # For rating events: 1.0 - 5.0
    weight = Column(Float, nullable=False, default=1.0) # Calculated implicit strength
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)

    user = relationship("User", back_populates="interactions")
    product = relationship("Product", back_populates="interactions")

    __table_args__ = (
        Index("ix_user_product_interaction", "user_id", "product_id"),
        Index("ix_event_timestamp", "event_type", "timestamp"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "product_id": self.product_id,
            "event_type": self.event_type,
            "rating_value": self.rating_value,
            "weight": self.weight,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }


class SearchLog(Base):
    """Search query history and search clicks."""
    __tablename__ = "search_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    query = Column(String(255), nullable=False, index=True)
    clicked_product_id = Column(Integer, ForeignKey("products.id", ondelete="SET NULL"), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="search_logs")

    def to_dict(self):
        return {
            "id": self.id,
            "user_id": self.user_id,
            "query": self.query,
            "clicked_product_id": self.clicked_product_id,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
        }
