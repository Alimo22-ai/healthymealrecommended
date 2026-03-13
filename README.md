# NutriMatch AI - Personalized Meal Recommendation System

A complete Machine Learning-based personalized meal recommendation system that provides tailored meal suggestions based on user profiles, dietary preferences, health conditions, and fitness goals.

## Features

- **Hard Filtering**: Removes meals with allergens, conflicting health conditions, and violating diet types
- **ML Ranking**: Uses Gradient Boosting to score and rank meals based on suitability
- **Feature Engineering**: Calculates BMI, BMR, TDEE, and adjusts macros based on goals
- **Explainable Recommendations**: Generates clear reasons for each recommended meal
- **FastAPI API**: Production-ready REST API endpoint
- **Streamlit UI**: User-friendly web interface

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    User Input                                │
│  Age, Gender, Weight, Height, Activity Level, Goal,         │
│  Diet Type, Allergies, Health Conditions, Nutrition Limits│
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Engineering                             │
│  • BMI Calculation                                           │
│  • BMR (Mifflin-St Jeor)                                     │
│  • TDEE with Activity Multipliers                           │
│  • Macro Adjustment for Goals                               │
│  • Categorical Encoding                                      │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  Hard Filtering Layer                        │
│  • Allergy Safety - Remove allergen-containing meals         │
│  • Health Safety - Remove conflicting meals                 │
│  • Diet Compliance - Filter by diet type                    │
│  • Nutrition Limits - Respect max calories/macros          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                  ML Ranking Model                            │
│  • Gradient Boosting Regressor                              │
│  • Features: User + Meal Combined Vectors                   │
│  • Training: Synthetic labels from macro matching           │
│  • Output: Suitability Score (0-1)                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│               Recommendations Output                         │
│  • Top K Meals Ranked by Score                             │
│  • Reason Generation for Each Meal                         │
│  • Full Recipe Details                                      │
└─────────────────────────────────────────────────────────────┘
```

## Installation

### 1. Install Dependencies

```bash
cd meal_recommender
pip install -r requirements.txt
```

### 2. Verify Data Files

Ensure the following files are in the `user_input_files` directory:

- `recips.json` - Meal/recipe dataset
- `nutrition_dataset.csv` - User profiles for training

## Usage

### Option 1: Run FastAPI Server

```bash
# Start the API server
python -m meal_recommender.api

# The API will be available at http://localhost:8000
# API documentation: http://localhost:8000/docs
```

### Option 2: Run Streamlit App

```bash
# Start Streamlit (requires API to be running)
streamlit run meal_recommender/streamlit_app.py

# The web interface will open at http://localhost:8501
```

### Option 3: Use as Python Library

```python
from meal_recommender import MealRecommendationEngine

# Initialize engine
engine = MealRecommendationEngine()
engine.initialize()

# Get recommendations
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

result = engine.get_recommendations(user_data, top_k=10)
print(json.dumps(result, indent=2))
```

## API Reference

### POST /recommend

Get personalized meal recommendations.

**Request Body:**
```json
{
  "user": {
    "age": 30,
    "gender": "male",
    "weight": 80,
    "height": 180,
    "activity_level": "moderate",
    "goal": "weight loss",
    "diet_type": "balanced",
    "allergies": ["none"],
    "health_conditions": [],
    "meals_per_day": 3,
    "max_calories": 500,
    "max_protein": 50,
    "max_carbs": 60,
    "max_fats": 25,
    "notes": ""
  },
  "top_k": 10
}
```

**Response:**
```json
{
  "recommendations": [
    {
      "name": "Grilled Chicken Breast with Sautéed Vegetables",
      "calories": 430,
      "protein": 40,
      "carbs": 12,
      "fats": 24,
      "image": "https://...",
      "recipe": ["chicken breasts", "olive oil", ...],
      "reason": "fits your calorie target for weight loss, 40g protein, within calorie limits"
    }
  ],
  "user_info": {
    "bmi": 24.7,
    "bmr": 1757.5,
    "tdee": 2724.1,
    "target_calories": 2224.1,
    "target_protein": 166.8,
    "target_carbs": 194.6,
    "target_fats": 77.9
  }
}
```

## User Input Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| age | int | Yes | User age (15-100) |
| gender | string | Yes | "male" or "female" |
| weight | float | Yes | Weight in kg (30-300) |
| height | float | Yes | Height in cm (100-250) |
| activity_level | string | Yes | "sedentary", "light", "moderate", or "high" |
| goal | string | Yes | "lose weight", "gain weight", "maintain weight", "build muscle", "improve health" |
| diet_type | string | Yes | "balanced", "vegan", "vegetarian", "keto", "low carb", "high protein" |
| allergies | list | No | ["nuts", "lactose", "gluten", "eggs", "none"] |
| health_conditions | list | No | ["diabetes", "hypertension", "heart disease"] |
| meals_per_day | int | No | Number of meals per day (2-5), default 3 |
| max_calories | float | No | Maximum calories per meal |
| max_protein | float | No | Maximum protein per meal |
| max_carbs | float | No | Maximum carbs per meal |
| max_fats | float | No | Maximum fats per meal |
| notes | string | No | Additional notes or preferences |

## Hard Constraints

The system enforces the following safety constraints:

1. **Allergy Safety**: Removes any recipe containing allergens from the user's allergy list
2. **Health Safety**: 
   - For diabetes: restricts carbs to ≤40g
   - For heart disease: restricts fats to ≤25g
   - For hypertension: restricts fats to ≤20g
3. **Diet Compliance**: Filters meals based on diet type (vegan, keto, etc.)
4. **Nutrition Limits**: Respects user-specified maximum calories, protein, carbs, and fats

## ML Model Details

- **Algorithm**: Gradient Boosting Regressor
- **Training Data**: Synthetically generated from user profiles and meal data
- **Features**: Combined user profile features + meal nutritional features
- **Target**: Suitability score based on macro matching (0-1)

## Performance

- **API Response Time**: <100ms per request (typical)
- **Model Training Time**: ~10 seconds on startup
- **Memory Usage**: ~100MB for loaded datasets and model

## Project Structure

```
meal_recommender/
├── __init__.py           # Package initialization
├── engine.py             # Core ML recommendation engine
├── api.py                # FastAPI application
├── streamlit_app.py      # Streamlit web interface
├── requirements.txt     # Python dependencies
└── user_input_files/    # Input data directory
    ├── recips.json      # Meal recipes dataset
    └── nutrition_dataset.csv  # User profiles dataset
```

## License

MIT License

## Author

Ali Mohamed Ali Mostafa
