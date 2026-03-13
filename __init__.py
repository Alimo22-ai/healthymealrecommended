"""
NutriMatch AI - Personalized Meal Recommendation System
ML-based meal recommendation engine with FastAPI and Streamlit interfaces
"""

__version__ = "1.0.0"
__author__ = "MiniMax Agent"

from .engine import (
    MealRecommendationEngine,
    UserProfile,
    load_recipes,
    load_user_data,
    calculate_bmi,
    calculate_bmr,
    calculate_tdee,
    get_user_target_macros,
    apply_hard_filters
)

__all__ = [
    "MealRecommendationEngine",
    "UserProfile", 
    "load_recipes",
    "load_user_data",
    "calculate_bmi",
    "calculate_bmr",
    "calculate_tdee",
    "get_user_target_macros",
    "apply_hard_filters"
]
