"""
Streamlit Web Application for Meal Recommendations
User-friendly interface for getting personalized meal recommendations
"""

import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time

# Page configuration
st.set_page_config(
    page_title="NutriMatch AI - Personalized Meal Recommendations",
    page_icon="🍽️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #10B981;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1F2937;
        margin-top: 1rem;
    }
    .meal-card {
        background-color: #FAFAFA;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .meal-name {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1F2937;
    }
    .meal-macros {
        display: flex;
        gap: 1rem;
        margin-top: 0.5rem;
    }
    .macro-badge {
        background-color: #10B981;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 15px;
        font-size: 0.85rem;
    }
    .reason-text {
        color: #6B7280;
        font-size: 0.9rem;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .user-stats {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
    }
    .stat-item {
        text-align: center;
    }
    .stat-value {
        font-size: 1.5rem;
        font-weight: bold;
    }
    .stat-label {
        font-size: 0.85rem;
        opacity: 0.9;
    }
    .info-box {
        background-color: #FEF3C7;
        border-left: 4px solid #F59E0B;
        padding: 1rem;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #D1FAE5;
        border-left: 4px solid #10B981;
        padding: 1rem;
        margin: 1rem 0;
    }
    .warning-box {
        background-color: #FEE2E2;
        border-left: 4px solid #EF4444;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_BASE_URL = "http://localhost:8000"

def check_api_health() -> bool:
    """Check if API is running"""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=2)
        return response.status_code == 200
    except:
        return False

def get_recommendations(user_data: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
    """Get recommendations from API"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/recommend",
            json={"user": user_data, "top_k": top_k},
            timeout=30
        )
        response.raise_for_status()
        return response.json()
    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Please ensure the API server is running.")
        return None
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<div class="main-header">🍽️ NutriMatch AI</div>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #6B7280;">Your Personalized Meal Recommendation Engine</p>', unsafe_allow_html=True)
    
    # Check API health
    api_available = check_api_health()
    
    if not api_available:
        st.markdown("""
        <div class="warning-box">
            <strong>⚠️ API Not Available</strong><br>
            The recommendation API is not running. Please start the API server first:
            <code>python -m meal_recommender.api</code>
        </div>
        """, unsafe_allow_html=True)
    
    # Create two columns
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown('<div class="sub-header">👤 Your Profile</div>', unsafe_allow_html=True)
        
        # Personal Information
        st.markdown("### Personal Information")
        age = st.number_input("Age", min_value=15, max_value=100, value=30, step=1)
        gender = st.radio("Gender", ["male", "female"], horizontal=True)
        weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.5)
        height = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=1.0)
        
        # Activity Level
        st.markdown("### Activity Level")
        activity_level = st.select_slider(
            "Daily Activity Level",
            options=["sedentary", "light", "moderate", "high"],
            value="moderate"
        )
        activity_descriptions = {
            "sedentary": "Little or no exercise",
            "light": "Light exercise 1-3 days/week",
            "moderate": "Moderate exercise 3-5 days/week",
            "high": "Hard exercise 6-7 days/week"
        }
        st.caption(activity_descriptions[activity_level])
        
        # Fitness Goal
        st.markdown("### Fitness Goal")
        goal = st.selectbox(
            "Your Goal",
            ["lose weight", "gain weight", "maintain weight", "build muscle", "improve health"]
        )
        
        # Diet Type
        st.markdown("### Diet Preferences")
        diet_type = st.selectbox(
            "Diet Type",
            ["balanced", "vegan", "vegetarian", "keto", "low carb", "high protein"]
        )
        
        # Allergies
        st.markdown("### Allergies")
        allergies = st.multiselect(
            "Select Allergies",
            ["nuts", "lactose", "gluten", "eggs"],
            default=[]
        )
        if not allergies:
            allergies = ["none"]
        
        # Health Conditions
        st.markdown("### Health Conditions")
        health_conditions = st.multiselect(
            "Select Health Conditions",
            ["diabetes", "hypertension", "heart disease"],
            default=[]
        )
        
        # Meals per day
        st.markdown("### Meal Settings")
        meals_per_day = st.slider("Meals per day", 2, 5, 3)
        
        # Nutrition Limits (Expandable)
        with st.expander("⚙️ Advanced Nutrition Limits"):
            st.markdown("Set maximum values per meal:")
            col_limit1, col_limit2 = st.columns(2)
            with col_limit1:
                max_calories = st.number_input("Max Calories", min_value=100, max_value=2000, value=500, step=50)
                max_protein = st.number_input("Max Protein (g)", min_value=10, max_value=100, value=50, step=5)
            with col_limit2:
                max_carbs = st.number_input("Max Carbs (g)", min_value=10, max_value=200, value=60, step=10)
                max_fats = st.number_input("Max Fats (g)", min_value=5, max_value=100, value=25, step=5)
        
        # Additional Notes
        notes = st.text_area("Additional Notes", placeholder="Any other preferences or dietary restrictions...")
        
        # Get Recommendations Button
        st.markdown("---")
        get_recommendations_btn = st.button("🍽️ Get Recommendations", type="primary", use_container_width=True)
    
    with col2:
        if get_recommendations_btn:
            # Prepare user data
            user_data = {
                "age": age,
                "gender": gender,
                "weight": weight,
                "height": height,
                "activity_level": activity_level,
                "goal": goal,
                "diet_type": diet_type,
                "allergies": allergies,
                "health_conditions": health_conditions,
                "meals_per_day": meals_per_day,
                "max_calories": max_calories,
                "max_protein": max_protein,
                "max_carbs": max_carbs,
                "max_fats": max_fats,
                "notes": notes
            }
            
            # Show loading
            with st.spinner("Analyzing your profile and finding the best meals..."):
                result = get_recommendations(user_data, top_k=10)
            
            if result:
                st.markdown("---")
                
                # Display user stats
                if "user_info" in result:
                    user_info = result["user_info"]
                    st.markdown(f"""
                    <div class="user-stats">
                        <div style="display: flex; justify-content: space-around;">
                            <div class="stat-item">
                                <div class="stat-value">{user_info['bmi']}</div>
                                <div class="stat-label">BMI</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{int(user_info['tdee'])}</div>
                                <div class="stat-label">Daily Calories (TDEE)</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{int(user_info['target_calories'])}</div>
                                <div class="stat-label">Target Calories</div>
                            </div>
                            <div class="stat-item">
                                <div class="stat-value">{int(user_info['target_protein'])}g</div>
                                <div class="stat-label">Target Protein</div>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Display recommendations
                recommendations = result.get("recommendations", [])
                
                if len(recommendations) == 0:
                    st.markdown("""
                    <div class="warning-box">
                        <strong>😕 No meals found</strong><br>
                        No meals match your strict criteria. Try relaxing some constraints like:
                        <ul>
                            <li>Widening your allergy selections</li>
                            <li>Adjusting nutrition limits</li>
                            <li>Selecting a different diet type</li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f'<div class="sub-header">🍽️ Your Top {len(recommendations)} Meal Recommendations</div>', unsafe_allow_html=True)
                    
                    for i, meal in enumerate(recommendations, 1):
                        with st.container():
                            st.markdown(f"""
                            <div class="meal-card">
                                <div style="display: flex; gap: 1rem;">
                                    <div style="flex: 0 0 150px;">
                                        <img src="{meal['image']}" style="width: 150px; height: 100px; object-fit: cover; border-radius: 8px;" onerror="this.src='https://via.placeholder.com/150x100?text=No+Image'">
                                    </div>
                                    <div style="flex: 1;">
                                        <div class="meal-name">{i}. {meal['name']}</div>
                                        <div class="meal-macros">
                                            <span class="macro-badge">🔥 {meal['calories']} cal</span>
                                            <span class="macro-badge" style="background-color: #3B82F6;">🥩 {meal['protein']}g protein</span>
                                            <span class="macro-badge" style="background-color: #F59E0B;">🍞 {meal['carbs']}g carbs</span>
                                            <span class="macro-badge" style="background-color: #EF4444;">🥑 {meal['fats']}g fats</span>
                                        </div>
                                        <div class="reason-text">💡 {meal['reason']}</div>
                                    </div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Show recipe details in expander
                            with st.expander(f"📝 View Recipe for {meal['name']}"):
                                st.markdown("**Ingredients:**")
                                for ingredient in meal.get("recipe", []):
                                    st.markdown(f"- {ingredient}")
                    
                    # Summary
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>✅ Found {len(recommendations)} personalized recommendations!</strong><br>
                        These meals are tailored to your {goal} goal, {diet_type} diet, and respect your allergies and health conditions.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            # Welcome message when no recommendations yet
            st.markdown("""
            <div style="text-align: center; padding: 3rem; color: #6B7280;">
                <div style="font-size: 4rem; margin-bottom: 1rem;">🍽️</div>
                <h3>Welcome to NutriMatch AI!</h3>
                <p>Fill in your profile on the left and click "Get Recommendations" to receive personalized meal suggestions.</p>
                <p>Our ML-powered engine will:</p>
                <ul style="text-align: left; display: inline-block;">
                    <li>✅ Respect your dietary restrictions and allergies</li>
                    <li>✅ Consider your health conditions</li>
                    <li>✅ Match your fitness goals</li>
                    <li>✅ Optimize for your nutritional needs</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
