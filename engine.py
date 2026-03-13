"""
Personalized Meal Recommendation System
ML-based recommendation engine with FastAPI and Streamlit interfaces
"""

import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# DATA LOADING AND CLEANING
# ============================================================================

def load_recipes(filepath: str = 'recips.json') -> pd.DataFrame:
    """Load and clean recipes dataset"""
    with open(filepath, 'r') as f:
        recipes_data = json.load(f)
    
    df = pd.DataFrame(recipes_data)
    
    # Clean and normalize data
    df['calories'] = pd.to_numeric(df['calories'], errors='coerce')
    df['protein'] = pd.to_numeric(df['protein'], errors='coerce')
    df['carbs'] = pd.to_numeric(df['carbs'], errors='coerce')
    df['fats'] = pd.to_numeric(df['fats'], errors='coerce')
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['calories', 'protein', 'carbs', 'fats', 'name'])
    
    # Remove outliers (extreme calorie values)
    df = df[(df['calories'] >= 50) & (df['calories'] <= 2000)]
    
    # Normalize diet and allergy tags
    df['diet'] = df['diet'].apply(lambda x: [d.lower().strip() for d in x] if isinstance(x, list) else ['diet'])
    df['allergy'] = df['allergy'].apply(lambda x: [a.lower().strip() for a in x] if isinstance(x, list) else ['none'])
    df['diseases'] = df['diseases'].apply(lambda x: [d.lower().strip() for d in x] if isinstance(x, list) else [])
    
    # Flatten ingredients into single string for text matching
    df['ingredients_text'] = df['ingredients'].apply(lambda x: ' '.join(x).lower() if isinstance(x, list) else '')
    
    return df.reset_index(drop=True)

def load_user_data(filepath: str = 'nutrition_dataset.csv') -> pd.DataFrame:
    """Load and clean user nutrition dataset"""
    df = pd.read_csv(filepath)
    
    # Clean column names
    df.columns = df.columns.str.strip()
    
    # Standardize column names
    column_mapping = {
        'Height': 'Height',
        'Weight': 'Weight',
        'Activity Level': 'Activity Level',
        'Fitness Goal': 'Fitness Goal',
        'Dietary Preference': 'Dietary Preference',
        'Daily Calorie Target': 'Daily Calorie Target',
        'Protein (g)': 'Protein',
        'Carbohydrates (g)': 'Carbohydrates',
        'Fat (g)': 'Fat'
    }
    
    # Rename columns that exist
    for old, new in column_mapping.items():
        if old in df.columns:
            df = df.rename(columns={old: new})
    
    # Convert numeric columns
    numeric_cols = ['Age', 'Height', 'Weight', 'Daily Calorie Target', 'Protein', 'Carbohydrates', 'Fat']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Remove rows with missing critical values
    df = df.dropna(subset=['Age', 'Height', 'Weight', 'Activity Level', 'Fitness Goal'])
    
    return df.reset_index(drop=True)

# ============================================================================
# USER PROFILE PROCESSING
# ============================================================================

@dataclass
class UserProfile:
    """User profile for meal recommendations"""
    age: int
    gender: str
    weight: float  # kg
    height: float  # cm
    activity_level: str
    goal: str
    diet_type: str
    allergies: List[str]
    health_conditions: List[str]
    meals_per_day: int = 3
    max_calories: Optional[float] = None
    max_protein: Optional[float] = None
    max_carbs: Optional[float] = None
    max_fats: Optional[float] = None
    notes: str = ""
    
    def __post_init__(self):
        # Normalize values
        self.gender = self.gender.lower().strip()
        self.activity_level = self.activity_level.lower().strip()
        self.goal = self.goal.lower().strip()
        self.diet_type = self.diet_type.lower().strip()
        self.allergies = [a.lower().strip() for a in self.allergies]
        self.health_conditions = [h.lower().strip() for h in self.health_conditions]

def calculate_bmi(weight: float, height: float) -> float:
    """Calculate BMI"""
    height_m = height / 100
    return weight / (height_m ** 2)

def get_bmi_category(bmi: float) -> str:
    """Get BMI category"""
    if bmi < 18.5:
        return 'underweight'
    elif bmi < 25:
        return 'normal'
    elif bmi < 30:
        return 'overweight'
    else:
        return 'obese'

def calculate_bmr(weight: float, height: float, age: int, gender: str) -> float:
    """Calculate BMR using Mifflin-St Jeor equation"""
    if gender.lower() == 'male':
        return (10 * weight) + (6.25 * height) - (5 * age) + 5
    else:
        return (10 * weight) + (6.25 * height) - (5 * age) - 161

def calculate_tdee(bmr: float, activity_level: str) -> float:
    """Calculate TDEE based on activity level"""
    activity_factors = {
        'sedentary': 1.2,
        'light': 1.375,
        'lightly active': 1.375,
        'moderate': 1.55,
        'moderately active': 1.55,
        'high': 1.725,
        'very active': 1.725,
        'active': 1.725
    }
    factor = activity_factors.get(activity_level.lower(), 1.2)
    return bmr * factor

def adjust_macros_for_goal(tdee: float, goal: str, weight: float) -> Dict[str, float]:
    """Adjust macro targets based on fitness goal"""
    goal = goal.lower()
    
    if 'loss' in goal or 'lose' in goal:
        # Weight loss: create calorie deficit
        target_calories = tdee - 500
        protein_ratio = 0.30  # Higher protein for satiety
        carb_ratio = 0.35
        fat_ratio = 0.35
    elif 'gain' in goal or 'muscle' in goal:
        # Muscle gain: calorie surplus
        target_calories = tdee + 500
        protein_ratio = 0.30  # Higher protein for muscle building
        carb_ratio = 0.40
        fat_ratio = 0.30
    elif 'maintain' in goal or 'maintenance' in goal:
        target_calories = tdee
        protein_ratio = 0.25
        carb_ratio = 0.45
        fat_ratio = 0.30
    else:
        # Improve health - balanced
        target_calories = tdee
        protein_ratio = 0.25
        carb_ratio = 0.45
        fat_ratio = 0.30
    
    # Ensure minimum calories
    target_calories = max(target_calories, 1200)
    
    return {
        'calories': target_calories,
        'protein': (target_calories * protein_ratio) / 4,  # 4 cal per gram
        'carbs': (target_calories * carb_ratio) / 4,      # 4 cal per gram
        'fats': (target_calories * fat_ratio) / 9          # 9 cal per gram
    }

def process_user_profile(user_data: Dict[str, Any]) -> UserProfile:
    """Process raw user data into UserProfile"""
    return UserProfile(
        age=int(user_data.get('age', 30)),
        gender=user_data.get('gender', 'male'),
        weight=float(user_data.get('weight', 70)),
        height=float(user_data.get('height', 170)),
        activity_level=user_data.get('activity_level', 'moderate'),
        goal=user_data.get('goal', 'maintain'),
        diet_type=user_data.get('diet_type', 'balanced'),
        allergies=user_data.get('allergies', ['none']),
        health_conditions=user_data.get('health_conditions', []),
        meals_per_day=int(user_data.get('meals_per_day', 3)),
        max_calories=float(user_data.get('max_calories', 2000)) if user_data.get('max_calories') else None,
        max_protein=float(user_data.get('max_protein', 150)) if user_data.get('max_protein') else None,
        max_carbs=float(user_data.get('max_carbs', 250)) if user_data.get('max_carbs') else None,
        max_fats=float(user_data.get('max_fats', 80)) if user_data.get('max_fats') else None,
        notes=user_data.get('notes', '')
    )

def get_user_target_macros(user: UserProfile) -> Dict[str, float]:
    """Calculate target macros for user"""
    bmi = calculate_bmi(user.weight, user.height)
    bmr = calculate_bmr(user.weight, user.height, user.age, user.gender)
    tdee = calculate_tdee(bmr, user.activity_level)
    
    base_macros = adjust_macros_for_goal(tdee, user.goal, user.weight)
    
    # Adjust for meals per day
    meal_multiplier = user.meals_per_day / 3
    
    return {
        'calories': base_macros['calories'] * meal_multiplier,
        'protein': base_macros['protein'] * meal_multiplier,
        'carbs': base_macros['carbs'] * meal_multiplier,
        'fats': base_macros['fats'] * meal_multiplier,
        'bmi': bmi,
        'bmr': bmr,
        'tdee': tdee
    }

# ============================================================================
# HARD FILTERING (SAFETY CONSTRAINTS)
# ============================================================================

def check_allergy_conflict(ingredients_text: str, user_allergies: List[str]) -> bool:
    """Check if recipe contains any user allergens"""
    allergy_keywords = {
        'nuts': ['nut', 'almond', 'walnut', 'cashew', 'peanut', 'pecan', 'pistachio', 'hazelnut'],
        'lactose': ['milk', 'cheese', 'yogurt', 'cream', 'butter', 'lactose', 'dairy', 'whey'],
        'gluten': ['wheat', 'barley', 'rye', 'oat', 'gluten', 'bread', 'pasta', 'flour'],
        'eggs': ['egg', 'mayonnaise', 'mayo']
    }
    
    ingredients_lower = ingredients_text.lower()
    
    for allergy in user_allergies:
        if allergy == 'none':
            continue
        if allergy in allergy_keywords:
            for keyword in allergy_keywords[allergy]:
                if keyword in ingredients_lower:
                    return True
    
    return False

def check_health_conflict(recipe_diseases: List[str], user_conditions: List[str]) -> bool:
    """Check if recipe conflicts with user's health conditions"""
    condition_keywords = {
        'diabetes': ['diabetes', 'diabetic'],
        'hypertension': ['hypertension', 'high blood pressure', 'heart'],
        'heart disease': ['heart', 'cardiovascular']
    }
    
    for condition in user_conditions:
        if condition in condition_keywords:
            # Check if recipe is NOT recommended for this condition
            for keyword in condition_keywords[condition]:
                if keyword in [d.lower() for d in recipe_diseases]:
                    # Recipe is marked for this disease - check if it's safe or harmful
                    # In our dataset, diseases list shows what conditions it's good for
                    pass
                    
    return False

def check_diet_compliance(recipe_diets: List[str], user_diet: str) -> bool:
    """Check if recipe complies with user's diet type"""
    user_diet_lower = user_diet.lower().strip()
    
    # Map diet types
    diet_mapping = {
        'high protein': ['keto', 'diet'],
        'vegan': ['vegan'],
        'vegetarian': ['vegetarian', 'vegan'],
        'low carb': ['keto', 'diet'],
        'keto': ['keto'],
        'balanced': ['diet'],
        'omnivore': ['diet', 'keto', 'vegan', 'vegetarian'],
        'balanced': ['diet']
    }
    
    allowed_diets = diet_mapping.get(user_diet_lower, ['diet'])
    
    for recipe_diet in recipe_diets:
        if recipe_diet.lower() in allowed_diets:
            return True
    
    return False

def apply_hard_filters(recipes_df: pd.DataFrame, user: UserProfile, target_macros: Dict) -> pd.DataFrame:
    """Apply all hard filtering constraints"""
    df = recipes_df.copy()
    
    # 1. Allergy filtering
    allergy_mask = ~df.apply(
        lambda row: check_allergy_conflict(row['ingredients_text'], user.allergies), 
        axis=1
    )
    df = df[allergy_mask]
    
    # 2. Diet compliance
    diet_mask = df['diet'].apply(lambda x: check_diet_compliance(x, user.diet_type))
    df = df[diet_mask]
    
    # 3. Health condition filtering - exclude recipes that could be harmful
    # For diabetes: avoid high sugar recipes (high carbs)
    if 'diabetes' in user.health_conditions:
        df = df[df['carbs'] <= 40]  # Lower carb limit for diabetes
    
    # For heart disease: prefer lower fat recipes
    if 'heart' in user.health_conditions or 'heart disease' in user.health_conditions:
        df = df[df['fats'] <= 25]
    
    # For hypertension: avoid high sodium (we don't have sodium data, but can limit fats)
    if 'hypertension' in user.health_conditions:
        df = df[df['fats'] <= 20]
    
    # 4. Nutrition limits
    if user.max_calories:
        df = df[df['calories'] <= user.max_calories]
    if user.max_protein:
        df = df[df['protein'] <= user.max_protein]
    if user.max_carbs:
        df = df[df['carbs'] <= user.max_carbs]
    if user.max_fats:
        df = df[df['fats'] <= user.max_fats]
    
    return df

# ============================================================================
# FEATURE ENGINEERING FOR ML
# ============================================================================

def create_user_features(user: UserProfile, target_macros: Dict) -> np.ndarray:
    """Create feature vector for user"""
    features = np.array([
        target_macros['calories'],
        target_macros['protein'],
        target_macros['carbs'],
        target_macros['fats'],
        target_macros['bmi'],
        target_macros['bmr'],
        target_macros['tdee'],
        user.age,
        user.weight,
        user.height,
        1.0 if user.gender == 'male' else 0.0,
        1.0 if 'sedentary' in user.activity_level else 0.0,
        1.0 if 'light' in user.activity_level else 0.0,
        1.0 if 'moderate' in user.activity_level else 0.0,
        1.0 if 'high' in user.activity_level or 'active' in user.activity_level else 0.0,
        1.0 if 'loss' in user.goal or 'lose' in user.goal else 0.0,
        1.0 if 'gain' in user.goal or 'muscle' in user.goal else 0.0,
        1.0 if 'maintain' in user.goal else 0.0,
        1.0 if 'vegan' in user.diet_type else 0.0,
        1.0 if 'vegetarian' in user.diet_type else 0.0,
        1.0 if 'keto' in user.diet_type else 0.0,
        1.0 if 'high protein' in user.diet_type else 0.0,
        1.0 if 'diabetes' in user.health_conditions else 0.0,
        1.0 if 'heart' in user.health_conditions else 0.0,
        1.0 if 'hypertension' in user.health_conditions else 0.0,
        user.meals_per_day
    ])
    return features

def create_meal_features(meal_row: pd.Series) -> np.ndarray:
    """Create feature vector for a meal"""
    features = np.array([
        meal_row['calories'],
        meal_row['protein'],
        meal_row['carbs'],
        meal_row['fats'],
        meal_row['protein'] / max(meal_row['calories'], 1) * 100,  # protein percentage
        meal_row['carbs'] / max(meal_row['calories'], 1) * 100,    # carbs percentage
        meal_row['fats'] / max(meal_row['calories'], 1) * 100,     # fats percentage
        1.0 if 'vegan' in meal_row['diet'] else 0.0,
        1.0 if 'vegetarian' in meal_row['diet'] else 0.0,
        1.0 if 'keto' in meal_row['diet'] else 0.0,
        1.0 if 'diabetes' in meal_row['diseases'] else 0.0,
        1.0 if 'heart' in meal_row['diseases'] else 0.0,
        1.0 if 'nuts' in meal_row['allergy'] else 0.0,
        1.0 if 'lactose' in meal_row['allergy'] else 0.0,
        1.0 if 'gluten' in meal_row['allergy'] else 0.0,
        meal_row.get('time', 20) if pd.notna(meal_row.get('time')) else 20
    ])
    return features

# ============================================================================
# ML MODEL TRAINING
# ============================================================================

class MealRankingModel:
    """ML model for ranking meal recommendations"""
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        
    def _calculate_suitability_score(self, user_features: np.ndarray, meal_features: np.ndarray) -> float:
        """Calculate ground truth suitability score based on macro matching"""
        # User target macros (first 4 elements)
        user_cal, user_prot, user_carb, user_fat = user_features[:4]
        
        # Meal macros (first 4 elements)
        meal_cal, meal_prot, meal_carb, meal_fat = meal_features[:4]
        
        # Calculate how well the meal fits within user's target macros
        # For weight loss: prefer lower calories
        # For muscle gain: prefer higher protein
        
        cal_score = 1.0 - min(abs(meal_cal - user_cal) / max(user_cal, 1), 1.0)
        prot_score = min(meal_prot / max(user_prot, 1), 1.0)
        carb_score = 1.0 - min(abs(meal_carb - user_carb) / max(user_carb, 1), 1.0)
        fat_score = 1.0 - min(abs(meal_fat - user_fat) / max(user_fat, 1), 1.0)
        
        # Additional factors
        protein_density = meal_features[4] / 100  # protein percentage
        is_vegan = meal_features[7]
        is_keto = meal_features[9]
        
        # Weight the scores
        total_score = (
            0.30 * cal_score +
            0.30 * prot_score +
            0.20 * carb_score +
            0.10 * fat_score +
            0.05 * protein_density +
            0.05 * is_vegan
        )
        
        return min(max(total_score, 0.0), 1.0)
    
    def train(self, recipes_df: pd.DataFrame, user_data_df: pd.DataFrame):
        """Train the ranking model using synthetic labels"""
        print("Training meal ranking model...")
        
        # Create training data by combining users with meals
        X_train = []
        y_train = []
        
        # Sample users for training
        sample_users = user_data_df.sample(n=min(100, len(user_data_df)), random_state=42)
        
        for _, user_row in sample_users.iterrows():
            # Create user profile
            try:
                user = UserProfile(
                    age=int(user_row['Age']),
                    gender=str(user_row['Gender']),
                    weight=float(user_row['Weight']),
                    height=float(user_row['Height']),
                    activity_level=str(user_row['Activity Level']),
                    goal=str(user_row['Fitness Goal']),
                    diet_type=str(user_row['Dietary Preference']),
                    allergies=['none'],
                    health_conditions=[]
                )
            except:
                continue
                
            target_macros = get_user_target_macros(user)
            user_features = create_user_features(user, target_macros)
            
            # For each user, create training samples from meals
            for _, meal_row in recipes_df.iterrows():
                meal_features = create_meal_features(meal_row)
                
                # Combine user and meal features
                combined = np.concatenate([user_features, meal_features])
                
                # Calculate suitability score
                suitability = self._calculate_suitability_score(user_features, meal_features)
                
                X_train.append(combined)
                y_train.append(suitability)
        
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Train gradient boosting model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42
        )
        self.model.fit(X_train_scaled, y_train)
        
        self.is_trained = True
        print(f"Model trained on {len(X_train)} samples")
    
    def predict_scores(self, user: UserProfile, recipes_df: pd.DataFrame) -> np.ndarray:
        """Predict suitability scores for all meals"""
        if not self.is_trained:
            # Return heuristic scores if not trained
            return self._heuristic_scores(user, recipes_df)
        
        target_macros = get_user_target_macros(user)
        user_features = create_user_features(user, target_macros)
        
        # Create feature matrix for all meals
        X_pred = []
        for _, meal_row in recipes_df.iterrows():
            meal_features = create_meal_features(meal_row)
            combined = np.concatenate([user_features, meal_features])
            X_pred.append(combined)
        
        X_pred = np.array(X_pred)
        X_pred_scaled = self.scaler.transform(X_pred)
        
        return self.model.predict(X_pred_scaled)
    
    def _heuristic_scores(self, user: UserProfile, recipes_df: pd.DataFrame) -> np.ndarray:
        """Fallback heuristic scoring"""
        target_macros = get_user_target_macros(user)
        
        scores = []
        for _, meal in recipes_df.iterrows():
            # Calculate macro match scores
            cal_diff = abs(meal['calories'] - target_macros['calories']) / max(target_macros['calories'], 1)
            prot_match = min(meal['protein'] / max(target_macros['protein'], 1), 1.0)
            
            score = (1.0 - min(cal_diff, 1.0)) * 0.5 + prot_match * 0.5
            scores.append(score)
        
        return np.array(scores)

# ============================================================================
# RECOMMENDATION ENGINE
# ============================================================================

class MealRecommendationEngine:
    """Main recommendation engine"""
    
    def __init__(self):
        self.recipes_df = None
        self.user_data_df = None
        self.model = None
        
    def initialize(self):
        """Initialize the engine with data and trained model"""
        print("Loading recipes data...")
        self.recipes_df = load_recipes()
        print(f"Loaded {len(self.recipes_df)} recipes")
        
        print("Loading user data...")
        self.user_data_df = load_user_data()
        print(f"Loaded {len(self.user_data_df)} user profiles")
        
        print("Training ranking model...")
        self.model = MealRankingModel()
        self.model.train(self.recipes_df, self.user_data_df)
        
    def get_recommendations(self, user_data: Dict[str, Any], top_k: int = 10) -> Dict[str, Any]:
        """Get meal recommendations for a user"""
        # Process user profile
        user = process_user_profile(user_data)
        
        # Get target macros
        target_macros = get_user_target_macros(user)
        
        # Apply hard filters
        filtered_df = apply_hard_filters(self.recipes_df, user, target_macros)
        
        if len(filtered_df) == 0:
            # No valid meals after filtering
            return {
                "recommendations": [],
                "user_info": {
                    "bmi": round(target_macros['bmi'], 1),
                    "bmr": round(target_macros['bmr'], 0),
                    "tdee": round(target_macros['tdee'], 0),
                    "target_calories": round(target_macros['calories'], 0),
                    "target_protein": round(target_macros['protein'], 0),
                    "target_carbs": round(target_macros['carbs'], 0),
                    "target_fats": round(target_macros['fats'], 0)
                },
                "message": "No meals match your strict criteria. Try relaxing some constraints."
            }
        
        # Get ML scores
        scores = self.model.predict_scores(user, filtered_df)
        filtered_df = filtered_df.copy()
        filtered_df['ml_score'] = scores
        
        # Rank by score
        filtered_df = filtered_df.sort_values('ml_score', ascending=False)
        
        # Get top K recommendations
        top_meals = filtered_df.head(top_k)
        
        # Generate recommendations with reasons
        recommendations = []
        for _, meal in top_meals.iterrows():
            reason = self._generate_reason(meal, user, target_macros)
            
            recommendations.append({
                "name": meal['name'],
                "calories": int(meal['calories']),
                "protein": int(meal['protein']),
                "carbs": int(meal['carbs']),
                "fats": int(meal['fats']),
                "image": meal.get('image', ''),
                "recipe": meal['ingredients'] if isinstance(meal['ingredients'], list) else [],
                "reason": reason
            })
        
        return {
            "recommendations": recommendations,
            "user_info": {
                "bmi": round(target_macros['bmi'], 1),
                "bmr": round(target_macros['bmr'], 0),
                "tdee": round(target_macros['tdee'], 0),
                "target_calories": round(target_macros['calories'], 0),
                "target_protein": round(target_macros['protein'], 0),
                "target_carbs": round(target_macros['carbs'], 0),
                "target_fats": round(target_macros['fats'], 0)
            }
        }
    
    def _generate_reason(self, meal: pd.Series, user: UserProfile, target_macros: Dict) -> str:
        """Generate human-readable reason for recommendation"""
        reasons = []
        
        # Goal alignment
        if 'loss' in user.goal or 'lose' in user.goal:
            if meal['calories'] <= target_macros['calories']:
                reasons.append("fits your calorie target for weight loss")
        elif 'gain' in user.goal or 'muscle' in user.goal:
            if meal['protein'] >= 25:
                reasons.append("high protein for muscle building")
        else:
            reasons.append("balanced nutrition for your goals")
        
        # Macro suitability
        if meal['protein'] >= 20:
            reasons.append(f"{int(meal['protein'])}g protein")
        if meal['calories'] <= target_macros['calories'] * 1.2:
            reasons.append("within calorie limits")
        
        # Diet compliance
        if 'vegan' in meal['diet'] and 'vegan' in user.diet_type:
            reasons.append("vegan")
        elif 'keto' in meal['diet'] and 'keto' in user.diet_type:
            reasons.append("keto-friendly")
        
        # Health conditions
        if 'diabetes' in user.health_conditions:
            if meal['carbs'] <= 30:
                reasons.append("low-carb for diabetes")
        if 'heart' in user.health_conditions or 'heart disease' in user.health_conditions:
            if meal['fats'] <= 20:
                reasons.append("heart-healthy")
        
        # Allergen safety
        if user.allergies == ['none'] or 'none' in user.allergies:
            reasons.append("allergen-safe")
        
        return ", ".join(reasons[:3])

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    # Test the engine
    engine = MealRecommendationEngine()
    engine.initialize()
    
    # Test recommendation
    test_user = {
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
    
    result = engine.get_recommendations(test_user, top_k=5)
    print(json.dumps(result, indent=2))
