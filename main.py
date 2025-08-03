
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import torch
import os
from contextlib import asynccontextmanager

# Import our model classes and preprocessor
from models import ExpenseClassifier, EntityRecognizer
from preprocessing import ArabicTextPreprocessor
# Import the new post-processor
from post_processing import EntityPostProcessor

# --- 1. Define Paths ---
CLASSIFIER_PATH = "Financial_Models/CAMELBERT_Classifier_SelfTrained_new_dataset_v1"
NER_PATH = "Financial_Models/fine_tuned_camelbert_nerModel"

# --- 2. Create Pydantic Models for Request and Response ---
class ExpenseText(BaseModel):
    text: str

class Entity(BaseModel):
    entity: str
    word: str

class ProcessedExpense(BaseModel):
    category: str
    confidence: float
    entities: list[Entity]

class ClassifierResponse(BaseModel):
    category: str
    confidence: float

# --- 3. Create FastAPI App with Lifespan ---
# This dictionary will hold our loaded models and utilities
ml_models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    print("Loading ML models and preprocessor...")
    ml_models["preprocessor"] = ArabicTextPreprocessor()
    ml_models["classifier"] = ExpenseClassifier(model_path=CLASSIFIER_PATH)
    ml_models["recognizer"] = EntityRecognizer(model_path=NER_PATH)
    ml_models["post_processor"] = EntityPostProcessor()  # Add the post-processor
    print("Models and preprocessor loaded successfully.")
    yield
    # Clean up resources on shutdown
    ml_models.clear()
    print("Models unloaded.")

app = FastAPI(title="Expense Processing API", lifespan=lifespan)

# Add CORS middleware to allow frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 4. Create Endpoints ---
@app.post("/process_expense/", response_model=ProcessedExpense)
async def process_expense(expense: ExpenseText):
    if not expense.text or not expense.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Step 1: Preprocess the raw text
    clean_text = ml_models["preprocessor"].preprocess(expense.text)

    # Step 2: Get category prediction
    classifier_result = ml_models["classifier"].predict(clean_text)
    category = classifier_result['category']
    confidence = classifier_result['confidence']

    # Step 3: Get NER prediction
    entities_raw = ml_models["recognizer"].predict(clean_text)
    
    # Step 4: Post-process the entities
    processed_entities = ml_models["post_processor"].process_entities(entities_raw)

    # Step 5: Assemble and return the final response
    return ProcessedExpense(
        category=category,
        confidence=confidence,
        entities=processed_entities
    )

# New endpoint for just classification (used by the React frontend)
@app.post("/classify", response_model=ClassifierResponse)
async def classify_expense(expense: ExpenseText):
    if not expense.text or not expense.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    # Step 1: Preprocess the raw text
    clean_text = ml_models["preprocessor"].preprocess(expense.text)

    # Step 2: Get category prediction
    classifier_result = ml_models["classifier"].predict(clean_text)
    category = classifier_result['category']
    confidence = classifier_result['confidence']

    # Step 3: Return just the classification result
    return ClassifierResponse(
        category=category,
        confidence=confidence
    )

# This allows running the app directly with: python -m backend.main
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

