import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input, decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os
import pathlib

class ImageClassifier:
    """Class for image classification using TensorFlow and pre-trained models"""
    
    def __init__(self, model_type="mobilenet_v2"):
        """Initialize the classifier with a pre-trained model or custom model
        
        Args:
            model_type (str): Type of model to use - 'mobilenet_v2' or 'fruit_classifier'
        """
        self.model_type = model_type
        
        # Define fruit classes for the custom model
        self.fruit_classes = [
            'apple fruit', 'banana fruit', 'cherry fruit', 'chickoo fruit',
            'grapes fruit', 'kiwi fruit', 'mango fruit', 'orange fruit', 'strawberry fruit'
        ]
        
        # Load appropriate model
        if model_type == "mobilenet_v2":
            self.model = MobileNetV2(weights='imagenet')
            print(f"Loaded {model_type} model for general image classification")
        elif model_type == "fruit_classifier":
            
            model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 
                                     "models", "fruit_classifier_model.h5")
            
            if not os.path.exists(model_path):
                raise FileNotFoundError(f"Custom fruit classifier model not found at {model_path}")
                
            self.model = load_model(model_path)
            print(f"Loaded custom fruit classifier model from {model_path}")
        else:
            # Default to MobileNetV2 if specified model is not available
            self.model = MobileNetV2(weights='imagenet')
            self.model_type = "mobilenet_v2"
            print(f"Unknown model type '{model_type}', defaulting to mobilenet_v2")
        
        # Warm up the model
        self._warmup()
    
    def _warmup(self):
        """Warm up the model with a dummy prediction"""
        dummy_input = np.zeros((1, 224, 224, 3))
        self.model.predict(dummy_input)
    
    def preprocess(self, img):
        """Preprocess the image for the model"""
        # Resize image to expected dimensions
        img = img.resize((224, 224))
        
        # Convert to array and preprocess
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        
        # Apply appropriate preprocessing based on model type
        if self.model_type == "mobilenet_v2":
            img_array = preprocess_input(img_array)  # MobileNetV2 specific preprocessing
        else:
            # For custom model, just normalize to [0,1]
            img_array = img_array / 255.0
        
        return img_array
    
    def classify(self, img, top_k=5):
        """Classify an image and return top k predictions"""
        # Preprocess the image
        processed_img = self.preprocess(img)
        
        # Make prediction
        predictions = self.model.predict(processed_img)
        
        # Decode and format results based on model type
        if self.model_type == "mobilenet_v2":
            # Use ImageNet decoder for MobileNetV2
            results = decode_predictions(predictions, top=top_k)[0]
            formatted_results = [
                {
                    "class_id": class_id,
                    "class_name": class_name,
                    "confidence": float(score)  # Convert numpy float to Python float for JSON serialization
                }
                for (class_id, class_name, score) in results
            ]
        else:
            # Custom decoding for fruit classifier
            # Get indices of top k predictions
            top_indices = np.argsort(predictions[0])[-top_k:][::-1]
            
            formatted_results = [
                {
                    "class_id": str(idx),
                    "class_name": self.fruit_classes[idx],
                    "confidence": float(predictions[0][idx])  # Convert numpy float to Python float
                }
                for idx in top_indices
            ]
        
        return formatted_results
    
    def classify_from_file(self, file_path, top_k=5):
        """Classify an image from a file path"""
        img = Image.open(file_path).convert('RGB')
        return self.classify(img, top_k=top_k)
    
    def classify_from_bytes(self, image_bytes, top_k=5):
        """Classify an image from bytes"""
        img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        return self.classify(img, top_k=top_k)