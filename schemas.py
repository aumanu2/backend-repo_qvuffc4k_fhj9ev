"""
Database Schemas

Define your MongoDB collection schemas here using Pydantic models.
These schemas are used for data validation in your application.

Each Pydantic model represents a collection in your database.
Model name is converted to lowercase for the collection name:
- User -> "user" collection
- Product -> "product" collection
- BlogPost -> "blogpost" collection
"""

from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

# Example schemas (kept for reference; you can remove if not needed)
class User(BaseModel):
    name: str = Field(..., description="Full name")
    email: str = Field(..., description="Email address")
    address: str = Field(..., description="Address")
    age: Optional[int] = Field(None, ge=0, le=120, description="Age in years")
    is_active: bool = Field(True, description="Whether user is active")

class Product(BaseModel):
    title: str = Field(..., description="Product title")
    description: Optional[str] = Field(None, description="Product description")
    price: float = Field(..., ge=0, description="Price in dollars")
    category: str = Field(..., description="Product category")
    in_stock: bool = Field(True, description="Whether product is in stock")

# Facemaxxing Analysis schema
class FacemaxAnalysis(BaseModel):
    """
    Stores the result of a single face analysis.
    Collection name: "facemaxanalysis" (lowercase of class name)
    """
    filename: Optional[str] = Field(None, description="Original uploaded filename")
    score: float = Field(..., ge=1, le=10, description="Attractiveness score from 1 to 10")
    review: Dict[str, str] = Field(..., description="Short review per feature: jawline, cheekbones, eyes, skin, symmetry")
    tips: List[str] = Field(..., description="Basic improvement tips")
    metrics: Dict[str, Any] = Field(..., description="Raw computed metrics and ratios used for scoring")
    image_width: Optional[int] = Field(None, description="Image width in pixels")
    image_height: Optional[int] = Field(None, description="Image height in pixels")
