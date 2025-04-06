from fastapi import APIRouter, UploadFile, File, HTTPException, Form, Query
from fastapi.responses import JSONResponse
from typing import List, Optional
import os
import uuid
from datetime import datetime
import aiofiles
from PIL import Image
import io

# Import model and utilities
from app.models.classifier import ImageClassifier

# Create router
router = APIRouter()

# Initialize classifiers
general_classifier = ImageClassifier(model_type="mobilenet_v2")
fruit_classifier = ImageClassifier(model_type="fruit_classifier")

# Upload directory
UPLOAD_DIR = "uploads"

@router.post("/classify")
async def classify_image(
    file: UploadFile = File(...),
    model_type: str = Query("fruit_classifier", description="Model type to use: 'mobilenet_v2' or 'fruit_classifier'")
):
    """Endpoint to classify an uploaded image"""
    # Validate file type
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = os.path.join(UPLOAD_DIR, unique_filename)
        
        # Save file
        async with aiofiles.open(file_path, "wb") as out_file:
            content = await file.read()
            await out_file.write(content)
        
        # Process image for classification
        image = Image.open(io.BytesIO(content))
        
        # Get classification results based on selected model
        if model_type == "mobilenet_v2":
            results = general_classifier.classify(image)
        else:  # Default to fruit classifier
            results = fruit_classifier.classify(image)
        
        # Return results
        return {
            "filename": unique_filename,
            "original_filename": file.filename,
            "file_path": file_path,
            "timestamp": datetime.now().isoformat(),
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@router.get("/results/{filename}")
async def get_classification_results(
    filename: str,
    model_type: str = Query("fruit_classifier", description="Model type to use: 'mobilenet_v2' or 'fruit_classifier'")
):
    """Get classification results for a specific image"""
    file_path = os.path.join(UPLOAD_DIR, filename)
    
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="Image not found")
    
    try:
        image = Image.open(file_path)
        
        # Get classification results based on selected model
        if model_type == "mobilenet_v2":
            results = general_classifier.classify(image)
        else:  # Default to fruit classifier
            results = fruit_classifier.classify(image)
        
        return {
            "filename": filename,
            "file_path": file_path,
            "results": results
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving results: {str(e)}")

@router.get("/history")
async def get_classification_history():
    """Get history of classified images"""
    try:
        files = os.listdir(UPLOAD_DIR)
        image_files = [f for f in files if os.path.isfile(os.path.join(UPLOAD_DIR, f)) and 
                      any(f.lower().endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".gif"])]
        
        history = []
        for filename in image_files:
            file_path = os.path.join(UPLOAD_DIR, filename)
            history.append({
                "filename": filename,
                "file_path": file_path,
                "timestamp": datetime.fromtimestamp(os.path.getctime(file_path)).isoformat()
            })
        
        return {"history": history}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving history: {str(e)}")