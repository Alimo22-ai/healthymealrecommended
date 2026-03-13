"""
Test script for Meal Recommendation System
Demonstrates various use cases and edge cases
"""

import json
import sys
sys.path.insert(0, '/workspace/meal_recommender')

from engine import MealRecommendationEngine

def test_basic_recommendation():
    """Test basic recommendation without constraints"""
    print("=" * 60)
    print("TEST 1: Basic Recommendation (Male, Weight Loss, No Allergies)")
    print("=" * 60)
    
    engine = MealRecommendationEngine()
    engine.initialize()
    
    user_data = {
        "age": 30,
        "gender": "male",
        "weight": 80,
        "height": 180,
        "activity_level": "moderate",
        "goal": "weight loss",
        "diet_type": "balanced",
        "allergies": ["none"],
        "health_conditions": [],
        "meals_per_day": 3
    }
    
    result = engine.get_recommendations(user_data, top_k=3)
    
    print(f"\nUser BMI: {result['user_info']['bmi']}")
    print(f"User TDEE: {result['user_info']['tdee']}")
    print(f"Target Calories: {result['user_info']['target_calories']}")
    print(f"\nTop 3 Recommendations:")
    for i, meal in enumerate(result['recommendations'][:3], 1):
        print(f"  {i}. {meal['name']} - {meal['calories']} cal, {meal['protein']}g protein")
        print(f"     Reason: {meal['reason']}")
    print()

def test_allergy_filtering():
    """Test allergy filtering"""
    print("=" * 60)
    print("TEST 2: Allergy Filtering (Nut Allergy)")
    print("=" * 60)
    
    engine = MealRecommendationEngine()
    engine.initialize()
    
    user_data = {
        "age": 28,
        "gender": "female",
        "weight": 60,
        "height": 165,
        "activity_level": "light",
        "goal": "maintain weight",
        "diet_type": "vegan",
        "allergies": ["nuts"],
        "health_conditions": [],
        "meals_per_day": 3
    }
    
    result = engine.get_recommendations(user_data, top_k=5)
    
    print(f"Found {len(result['recommendations'])} nut-free recommendations")
    for i, meal in enumerate(result['recommendations'][:3], 1):
        print(f"  {i}. {meal['name']} - {meal['calories']} cal")
        print(f"     Reason: {meal['reason']}")
    print()

def test_health_conditions():
    """Test health condition filtering"""
    print("=" * 60)
    print("TEST 3: Health Conditions (Diabetes)")
    print("=" * 60)
    
    engine = MealRecommendationEngine()
    engine.initialize()
    
    user_data = {
        "age": 45,
        "gender": "male",
        "weight": 85,
        "height": 175,
        "activity_level": "moderate",
        "goal": "improve health",
        "diet_type": "low carb",
        "allergies": ["none"],
        "health_conditions": ["diabetes"],
        "meals_per_day": 3
    }
    
    result = engine.get_recommendations(user_data, top_k=5)
    
    print(f"Found {len(result['recommendations'])} diabetes-friendly recommendations")
    print("(Low carbs, suitable for diabetic diet)")
    for i, meal in enumerate(result['recommendations'][:3], 1):
        print(f"  {i}. {meal['name']} - {meal['carbs']}g carbs")
        print(f"     Reason: {meal['reason']}")
    print()

def test_diet_compliance():
    """Test diet type compliance"""
    print("=" * 60)
    print("TEST 4: Diet Compliance (Keto)")
    print("=" * 60)
    
    engine = MealRecommendationEngine()
    engine.initialize()
    
    user_data = {
        "age": 35,
        "gender": "female",
        "weight": 70,
        "height": 170,
        "activity_level": "high",
        "goal": "maintain weight",
        "diet_type": "keto",
        "allergies": ["none"],
        "health_conditions": [],
        "meals_per_day": 3
    }
    
    result = engine.get_recommendations(user_data, top_k=5)
    
    print(f"Found {len(result['recommendations'])} keto-friendly recommendations")
    for i, meal in enumerate(result['recommendations'][:3], 1):
        print(f"  {i}. {meal['name']} - {meal['fats']}g fats, {meal['carbs']}g carbs")
        print(f"     Reason: {meal['reason']}")
    print()

def test_nutrition_limits():
    """Test nutrition limits"""
    print("=" * 60)
    print("TEST 5: Nutrition Limits (Low Calories)")
    print("=" * 60)
    
    engine = MealRecommendationEngine()
    engine.initialize()
    
    user_data = {
        "age": 25,
        "gender": "female",
        "weight": 55,
        "height": 160,
        "activity_level": "sedentary",
        "goal": "weight loss",
        "diet_type": "balanced",
        "allergies": ["none"],
        "health_conditions": [],
        "meals_per_day": 3,
        "max_calories": 300,
        "max_protein": 30,
        "max_carbs": 40,
        "max_fats": 15
    }
    
    result = engine.get_recommendations(user_data, top_k=5)
    
    print(f"Found {len(result['recommendations'])} recommendations within limits")
    for i, meal in enumerate(result['recommendations'][:3], 1):
        print(f"  {i}. {meal['name']} - {meal['calories']} cal, {meal['protein']}g protein, {meal['carbs']}g carbs, {meal['fats']}g fats")
    print()

if __name__ == "__main__":
    test_basic_recommendation()
    test_allergy_filtering()
    test_health_conditions()
    test_diet_compliance()
    test_nutrition_limits()
    print("All tests completed!")
