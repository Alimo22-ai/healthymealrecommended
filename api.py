"""
FastAPI Application for Meal Recommendations
REST API endpoint for personalized meal recommendations
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import uvicorn
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from engine import MealRecommendationEngine, UserProfile

# Initialize FastAPI app
app = FastAPI(
    title="Personalized Meal Recommendation API",
    description="ML-based meal recommendation system with dietary constraints",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize recommendation engine
engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize the recommendation engine on startup"""
    global engine
    engine = MealRecommendationEngine()
    engine.initialize()

# Request models
class UserProfileRequest(BaseModel):
    age: int = Field(..., ge=15, le=100, description="User age in years")
    gender: str = Field(..., description="User gender: male or female")
    weight: float = Field(..., gt=30, lt=300, description="User weight in kg")
    height: float = Field(..., gt=100, lt=250, description="User height in cm")
    activity_level: str = Field(..., description="Activity level: sedentary, light, moderate, or high")
    goal: str = Field(..., description="Fitness goal: lose weight, gain weight, maintain weight, build muscle, or improve health")
    diet_type: str = Field(..., description="Diet type: balanced, vegan, vegetarian, keto, low carb, or high protein")
    allergies: List[str] = Field(default=["none"], description="List of allergies: nuts, lactose, gluten, eggs, or none")
    health_conditions: List[str] = Field(default=[], description="Health conditions: diabetes, hypertension, heart disease")
    meals_per_day: int = Field(default=3, ge=2, le=5, description="Number of meals per day")
    max_calories: Optional[float] = Field(default=None, description="Maximum calories per meal")
    max_protein: Optional[float] = Field(default=None, description="Maximum protein per meal")
    max_carbs: Optional[float] = Field(default=None, description="Maximum carbs per meal")
    max_fats: Optional[float] = Field(default=None, description="Maximum fats per meal")
    notes: str = Field(default="", description="Additional notes or preferences")

class RecommendationRequest(BaseModel):
    user: UserProfileRequest
    top_k: int = Field(default=10, ge=1, le=20, description="Number of recommendations to return")

# Response models
class MealRecommendation(BaseModel):
    name: str
    calories: int
    protein: int
    carbs: int
    fats: int
    image: str
    recipe: List[str]
    reason: str

class UserInfo(BaseModel):
    bmi: float
    bmr: float
    tdee: float
    target_calories: float
    target_protein: float
    target_carbs: float
    target_fats: float

class RecommendationResponse(BaseModel):
    recommendations: List[MealRecommendation]
    user_info: UserInfo
    message: Optional[str] = None

# API endpoints
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Personalized Meal Recommendation API",
        "version": "1.0.0",
        "endpoints": {
            "POST /recommend": "Get meal recommendations",
            "GET /health": "Health check",
            "GET /recipes/count": "Get total number of recipes"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "engine_loaded": engine is not None,
        "recipes_count": len(engine.recipes_df) if engine and engine.recipes_df is not None else 0
    }

@app.get("/recipes/count")
async def get_recipes_count():
    """Get total number of available recipes"""
    if engine is None or engine.recipes_df is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    return {"count": len(engine.recipes_df)}

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized meal recommendations
    
    The system applies:
    1. Allergy filtering - removes recipes with user allergens
    2. Health condition filtering - removes unsuitable recipes
    3. Diet compliance - ensures diet type match
    4. Nutrition limits - respects max calorie/macro limits
    5. ML ranking - scores and ranks by suitability
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        # Convert request to dict
        user_data = request.user.dict()
        
        # Get recommendations
        result = engine.get_recommendations(user_data, top_k=request.top_k)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

@app.post("/recommend/simple")
async def get_recommendations_simple(user_data: Dict[str, Any]):
    """
    Simple endpoint with flexible input
    
    Accepts any user data fields and returns recommendations
    """
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not initialized")
    
    try:
        result = engine.get_recommendations(user_data, top_k=10)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating recommendations: {str(e)}")

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
