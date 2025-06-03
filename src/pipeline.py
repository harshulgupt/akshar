"""
Main pipeline integration for Brahmi to Devanagari transliteration
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
from PIL import Image
import json
import pandas as pd
import time
from datetime import datetime
import unicodedata
from src.ocr_module import BrahmiOCR
from src.transliteration_model import BrahmiToDevanagariTransliterator
from src.variant_support import MultiVariantTransliterator
from src.confidence_and_correction import ConfidenceScorer
from src.feedback_loop import FeedbackHandler, ContinualLearningManager
from src.explainable_ai import AttentionVisualizer, ModelExplainer

class BrahmiToDevanagariPipeline:
    """
    End-to-end pipeline for Brahmi to Devanagari transliteration.
    """
    
    def __init__(self, ocr_model_path=None, transliteration_model_path=None,
                 variant_model_paths=None, use_multi_variant=False,
                 feedback_threshold=10, auto_update_interval=24,
                 model_save_dir='models'):
        """
        Initialize the pipeline.
        
        Args:
            ocr_model_path (str, optional): Path to OCR model
            transliteration_model_path (str, optional): Path to transliteration model
            variant_model_paths (dict, optional): Dictionary mapping variant names to model paths
            use_multi_variant (bool): Whether to use multi-variant support
            feedback_threshold (int): Number of corrections needed to trigger model update
            auto_update_interval (int): Interval in hours for automatic model update
            model_save_dir (str): Directory to save updated models
        """
        # Initialize OCR module
        self.ocr_module = BrahmiOCR(model_path=ocr_model_path)
        
        # Initialize transliteration module
        if use_multi_variant and variant_model_paths:
            self.transliterator = MultiVariantTransliterator(
                default_model_path=transliteration_model_path,
                variant_model_paths=variant_model_paths
            )
        else:
            self.transliterator = BrahmiToDevanagariTransliterator(
                model_path=transliteration_model_path
            )
        
        # Initialize confidence scorer
        self.confidence_scorer = ConfidenceScorer(self.transliterator)
        
        # Initialize feedback handler
        self.feedback_handler = FeedbackHandler(
            self.transliterator,
            correction_threshold=feedback_threshold,
            auto_update_interval=auto_update_interval,
            model_save_dir=os.path.join(model_save_dir, 'feedback')
        )
        
        # Initialize continual learning manager
        self.continual_learning_manager = ContinualLearningManager(
            self.transliterator,
            self.feedback_handler,
            model_save_dir=os.path.join(model_save_dir, 'continual')
        )
        
        # Initialize explainable AI components
        self.attention_visualizer = AttentionVisualizer(self.transliterator)
        self.model_explainer = ModelExplainer(self.transliterator, self.attention_visualizer)
        
        # Create model save directory if specified
        if model_save_dir:
            os.makedirs(model_save_dir, exist_ok=True)
    
    def preprocess_image(self, image):
        """
        Preprocess an image for OCR.
        
        Args:
            image (numpy.ndarray or str): Input image or path to image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Load image if path is provided
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Failed to load image: {image}")
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Ensure image is in RGB format
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        # Resize image if needed
        if image.shape[0] > 1000 or image.shape[1] > 1000:
            scale = min(1000 / image.shape[0], 1000 / image.shape[1])
            new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
            image = cv2.resize(image, new_size)
        
        # Apply preprocessing
        # - Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # - Apply adaptive thresholding
        binary = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        
        # - Remove noise
        kernel = np.ones((2, 2), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # - Normalize
        normalized = binary / 255.0
        
        return normalized
    
    def segment_characters(self, image):
        """
        Segment characters from an image.
        
        Args:
            image (numpy.ndarray): Preprocessed image
            
        Returns:
            list: List of character images
        """
        # Convert to binary if not already
        if image.max() <= 1.0:
            binary = (image * 255).astype(np.uint8)
        else:
            binary = image.astype(np.uint8)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort contours from left to right, top to bottom
        def sort_key(contour):
            x, y, w, h = cv2.boundingRect(contour)
            return y // 30 * 1000 + x  # Group by rows (assuming row height < 30 pixels)
        
        contours = sorted(contours, key=sort_key)
        
        # Extract character images
        char_images = []
        
        for contour in contours:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Skip very small contours (noise)
            if w < 5 or h < 5:
                continue
            
            # Extract character image
            char_img = binary[y:y+h, x:x+w]
            
            # Pad to square
            size = max(w, h) + 4
            padded = np.zeros((size, size), dtype=np.uint8)
            
            # Center character in padded image
            x_offset = (size - w) // 2
            y_offset = (size - h) // 2
            padded[y_offset:y_offset+h, x_offset:x_offset+w] = char_img
            
            # Resize to standard size
            resized = cv2.resize(padded, (28, 28))
            
            char_images.append(resized)
        
        return char_images
    
    def process_image(self, image):
        """
        Process an image through the pipeline.
        
        Args:
            image (numpy.ndarray or str): Input image or path to image
            
        Returns:
            dict: Processing results
        """
        # Measure processing time
        start_time = time.time()
        
        # Preprocess image
        preprocessed = self.preprocess_image(image)
        
        # Segment characters
        char_images = self.segment_characters(preprocessed)
        
        if not char_images:
            return {
                'error': 'No characters found in image',
                'transliterated_text': '',
                'confidence_scores': [],
                'processing_time': time.time() - start_time
            }
        
        # Recognize characters
        char_images_array = np.array(char_images) / 255.0
        char_images_array = np.expand_dims(char_images_array, axis=-1)
        
        recognized_chars = self.ocr_module.recognize_characters(char_images_array)
        
        # Join characters into text
        brahmi_text = ''.join(recognized_chars)
        
        # Transliterate text
        if isinstance(self.transliterator, MultiVariantTransliterator):
            result = self.transliterator.transliterate(brahmi_text)
            devanagari_text = result['transliterated_text']
            variant = result.get('detected_variant', 'Unknown')
            variant_confidence = result.get('variant_confidence', 0.0)
        else:
            devanagari_text = self.transliterator.transliterate(brahmi_text)
            variant = None
            variant_confidence = None
        
        # Calculate confidence scores
        confidence_scores = self.confidence_scorer.calculate_character_confidence(
            brahmi_text, devanagari_text)
        
        # Calculate average confidence
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Create result
        result = {
            'brahmi_text': brahmi_text,
            'transliterated_text': devanagari_text,
            'confidence_scores': confidence_scores,
            'confidence': avg_confidence,
            'processing_time': processing_time
        }
        
        # Add variant information if available
        if variant:
            result['detected_variant'] = variant
            result['variant_confidence'] = variant_confidence
        
        return result
    
    def process_correction(self, brahmi_text, original_text, corrected_text):
        """
        Process a user correction.
        
        Args:
            brahmi_text (str): Brahmi text
            original_text (str): Original transliterated text
            corrected_text (str): Corrected transliterated text
            
        Returns:
            bool: Whether the model was updated
        """
        # Create correction
        correction = {
            'brahmi_text': brahmi_text,
            'original_text': original_text,
            'corrected_text': corrected_text,
            'timestamp': pd.Timestamp.now()
        }
        
        # Process correction
        return self.feedback_handler.process_correction(correction)
    
    def explain_transliteration(self, result, output_dir=None):
        """
        Explain the transliteration process.
        
        Args:
            result (dict): Processing result from process_image
            output_dir (str, optional): Directory to save explanation files
            
        Returns:
            dict: Explanation data
        """
        brahmi_text = result['brahmi_text']
        devanagari_text = result['transliterated_text']
        
        # Generate explanation
        return self.model_explainer.explain_transliteration(brahmi_text, output_dir)
    
    def create_explanation_report(self, result, output_path):
        """
        Create a comprehensive explanation report.
        
        Args:
            result (dict): Processing result from process_image
            output_path (str): Path to save the report
            
        Returns:
            str: Path to the report
        """
        brahmi_text = result['brahmi_text']
        
        # Create explanation report
        return self.model_explainer.create_explanation_report(brahmi_text, output_path)
    
    def save_models(self, save_dir):
        """
        Save all models.
        
        Args:
            save_dir (str): Directory to save models
            
        Returns:
            dict: Paths to saved models
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save OCR model
        ocr_path = os.path.join(save_dir, 'ocr_model')
        self.ocr_module.save_model(ocr_path)
        
        # Save transliteration model
        if isinstance(self.transliterator, MultiVariantTransliterator):
            # Save default transliterator
            default_path = os.path.join(save_dir, 'transliterator_default')
            self.transliterator.default_transliterator.save_model(default_path)
            
            # Save variant transliterators
            variant_paths = {}
            
            for variant_name, variant_transliterator in self.transliterator.transliterators.items():
                variant_path = os.path.join(save_dir, f'transliterator_{variant_name}')
                variant_transliterator.save_model(variant_path)
                variant_paths[variant_name] = variant_path
            
            transliterator_path = {
                'default': default_path,
                'variants': variant_paths
            }
        else:
            # Save single transliterator
            transliterator_path = os.path.join(save_dir, 'transliterator')
            self.transliterator.save_model(transliterator_path)
        
        # Return paths
        return {
            'ocr_model': ocr_path,
            'transliterator': transliterator_path
        }
    
    def load_models(self, load_dir):
        """
        Load all models.
        
        Args:
            load_dir (str): Directory to load models from
            
        Returns:
            bool: Whether models were loaded successfully
        """
        try:
            # Load OCR model
            ocr_path = os.path.join(load_dir, 'ocr_model')
            if os.path.exists(ocr_path):
                self.ocr_module.load_model(ocr_path)
            
            # Check if multi-variant transliterator
            default_path = os.path.join(load_dir, 'transliterator_default')
            if os.path.exists(default_path):
                # Load default transliterator
                self.transliterator.default_transliterator.load_model(default_path)
                
                # Load variant transliterators
                for variant_name, variant_transliterator in self.transliterator.transliterators.items():
                    variant_path = os.path.join(load_dir, f'transliterator_{variant_name}')
                    if os.path.exists(variant_path):
                        variant_transliterator.load_model(variant_path)
            else:
                # Load single transliterator
                transliterator_path = os.path.join(load_dir, 'transliterator')
                if os.path.exists(transliterator_path):
                    self.transliterator.load_model(transliterator_path)
            
            return True
            
        except Exception as e:
            print(f"Error loading models: {str(e)}")
            return False


if __name__ == "__main__":
    # Example usage
    pipeline = BrahmiToDevanagariPipeline()
    
    # Process an image
    result = pipeline.process_image("path/to/brahmi_image.jpg")
    
    # Print the transliterated text
    print(f"Brahmi text: {result['brahmi_text']}")
    print(f"Transliterated text: {result['transliterated_text']}")
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Processing time: {result['processing_time']:.4f} seconds")
    
    # Generate explanation
    explanation = pipeline.explain_transliteration(result, "explanation_output")
    
    # Create explanation report
    report_path = pipeline.create_explanation_report(result, "explanation_report.html")
    
    print(f"Explanation report saved to: {report_path}")
