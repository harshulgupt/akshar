"""
Validation and evaluation of the Brahmi to Devanagari transliteration pipeline
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json
import pandas as pd
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from tqdm import tqdm
import time
from datetime import datetime
import unicodedata
from src.ocr_module import BrahmiOCR
from src.transliteration_model import BrahmiToDevanagariTransliterator
from src.variant_support import MultiVariantTransliterator
from src.confidence_and_correction import ConfidenceScorer
from src.feedback_loop import FeedbackHandler
from src.explainable_ai import AttentionVisualizer, ModelExplainer
from src.pipeline import BrahmiToDevanagariPipeline
from src.data_augmentation import ImageAugmenter, TextAugmenter, SemiSupervisedLearner

class PipelineValidator:
    """
    Validator for the Brahmi to Devanagari transliteration pipeline.
    """
    
    def __init__(self, pipeline, test_data_dir=None, test_data_file=None):
        """
        Initialize the pipeline validator.
        
        Args:
            pipeline (BrahmiToDevanagariPipeline): Pipeline to validate
            test_data_dir (str, optional): Directory containing test images
            test_data_file (str, optional): File containing test text pairs
        """
        self.pipeline = pipeline
        self.test_data_dir = test_data_dir
        self.test_data_file = test_data_file
        
        # Initialize metrics
        self.ocr_metrics = {}
        self.transliteration_metrics = {}
        self.pipeline_metrics = {}
        self.confidence_metrics = {}
        self.variant_metrics = {}
        self.timing_metrics = {}
    
    def validate_ocr_module(self, test_images, test_labels):
        """
        Validate the OCR module.
        
        Args:
            test_images (numpy.ndarray): Test images
            test_labels (numpy.ndarray): Test labels
            
        Returns:
            dict: OCR metrics
        """
        print("Validating OCR module...")
        
        # Get OCR module
        ocr_module = self.pipeline.ocr_module
        
        # Predict labels
        start_time = time.time()
        predicted_labels = ocr_module.predict(test_images)
        end_time = time.time()
        
        # Calculate metrics
        accuracy = accuracy_score(test_labels, predicted_labels)
        precision, recall, f1, _ = precision_recall_fscore_support(
            test_labels, predicted_labels, average='weighted')
        
        # Calculate confusion matrix
        cm = confusion_matrix(test_labels, predicted_labels)
        
        # Calculate timing metrics
        inference_time = end_time - start_time
        avg_inference_time = inference_time / len(test_images)
        
        # Store metrics
        self.ocr_metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': cm,
            'inference_time': inference_time,
            'avg_inference_time': avg_inference_time
        }
        
        print(f"OCR Accuracy: {accuracy:.4f}")
        print(f"OCR F1 Score: {f1:.4f}")
        print(f"OCR Average Inference Time: {avg_inference_time:.4f} seconds")
        
        return self.ocr_metrics
    
    def validate_transliteration_module(self, test_brahmi_texts, test_devanagari_texts):
        """
        Validate the transliteration module.
        
        Args:
            test_brahmi_texts (list): Test Brahmi texts
            test_devanagari_texts (list): Test Devanagari texts
            
        Returns:
            dict: Transliteration metrics
        """
        print("Validating transliteration module...")
        
        # Get transliteration module
        if hasattr(self.pipeline, 'transliterator'):
            transliterator = self.pipeline.transliterator
        else:
            print("Transliteration module not found in pipeline")
            return {}
        
        # Predict transliterations
        predicted_texts = []
        character_accuracies = []
        inference_times = []
        
        for brahmi_text in tqdm(test_brahmi_texts, desc="Transliterating"):
            # Measure inference time
            start_time = time.time()
            
            # Transliterate text
            if isinstance(transliterator, MultiVariantTransliterator):
                result = transliterator.transliterate(brahmi_text)
                predicted_text = result['transliterated_text']
            else:
                predicted_text = transliterator.transliterate(brahmi_text)
            
            end_time = time.time()
            inference_time = end_time - start_time
            inference_times.append(inference_time)
            
            predicted_texts.append(predicted_text)
        
        # Calculate text-level accuracy
        text_matches = sum(1 for pred, true in zip(predicted_texts, test_devanagari_texts) 
                          if pred == true)
        text_accuracy = text_matches / len(test_brahmi_texts) if test_brahmi_texts else 0
        
        # Calculate character-level metrics
        all_true_chars = []
        all_pred_chars = []
        
        for true_text, pred_text in zip(test_devanagari_texts, predicted_texts):
            # Pad shorter text with spaces
            max_len = max(len(true_text), len(pred_text))
            true_padded = true_text.ljust(max_len)
            pred_padded = pred_text.ljust(max_len)
            
            all_true_chars.extend(list(true_padded))
            all_pred_chars.extend(list(pred_padded))
            
            # Calculate character accuracy for this text
            char_matches = sum(1 for t, p in zip(true_padded, pred_padded) if t == p)
            char_accuracy = char_matches / max_len if max_len > 0 else 1.0
            character_accuracies.append(char_accuracy)
        
        # Calculate overall character-level metrics
        char_accuracy = accuracy_score(all_true_chars, all_pred_chars)
        char_precision, char_recall, char_f1, _ = precision_recall_fscore_support(
            all_true_chars, all_pred_chars, average='weighted')
        
        # Calculate timing metrics
        total_inference_time = sum(inference_times)
        avg_inference_time = total_inference_time / len(test_brahmi_texts) if test_brahmi_texts else 0
        
        # Store metrics
        self.transliteration_metrics = {
            'text_accuracy': text_accuracy,
            'char_accuracy': char_accuracy,
            'char_precision': char_precision,
            'char_recall': char_recall,
            'char_f1': char_f1,
            'character_accuracies': character_accuracies,
            'avg_character_accuracy': sum(character_accuracies) / len(character_accuracies) if character_accuracies else 0,
            'total_inference_time': total_inference_time,
            'avg_inference_time': avg_inference_time
        }
        
        print(f"Transliteration Text Accuracy: {text_accuracy:.4f}")
        print(f"Transliteration Character Accuracy: {char_accuracy:.4f}")
        print(f"Transliteration Character F1 Score: {char_f1:.4f}")
        print(f"Transliteration Average Inference Time: {avg_inference_time:.4f} seconds")
        
        return self.transliteration_metrics
    
    def validate_confidence_scoring(self, test_brahmi_texts, test_devanagari_texts):
        """
        Validate the confidence scoring module.
        
        Args:
            test_brahmi_texts (list): Test Brahmi texts
            test_devanagari_texts (list): Test Devanagari texts
            
        Returns:
            dict: Confidence metrics
        """
        print("Validating confidence scoring...")
        
        # Get confidence scorer
        if hasattr(self.pipeline, 'confidence_scorer'):
            confidence_scorer = self.pipeline.confidence_scorer
        else:
            print("Confidence scorer not found in pipeline")
            return {}
        
        # Calculate confidence scores
        confidence_scores = []
        character_confidences = []
        
        for brahmi_text, devanagari_text in tqdm(zip(test_brahmi_texts, test_devanagari_texts), 
                                               desc="Calculating confidence"):
            # Calculate character confidence scores
            char_confidences = confidence_scorer.calculate_character_confidence(
                brahmi_text, devanagari_text)
            
            # Calculate average confidence
            avg_confidence = sum(char_confidences) / len(char_confidences) if char_confidences else 0
            
            confidence_scores.append(avg_confidence)
            character_confidences.extend(char_confidences)
        
        # Calculate metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        avg_char_confidence = sum(character_confidences) / len(character_confidences) if character_confidences else 0
        
        # Store metrics
        self.confidence_metrics = {
            'avg_confidence': avg_confidence,
            'avg_char_confidence': avg_char_confidence,
            'confidence_scores': confidence_scores,
            'character_confidences': character_confidences
        }
        
        print(f"Average Confidence Score: {avg_confidence:.4f}")
        print(f"Average Character Confidence: {avg_char_confidence:.4f}")
        
        return self.confidence_metrics
    
    def validate_variant_support(self, test_brahmi_texts, variant_labels=None):
        """
        Validate the variant support module.
        
        Args:
            test_brahmi_texts (list): Test Brahmi texts
            variant_labels (list, optional): True variant labels
            
        Returns:
            dict: Variant metrics
        """
        print("Validating variant support...")
        
        # Check if pipeline has multi-variant support
        if hasattr(self.pipeline, 'transliterator') and isinstance(self.pipeline.transliterator, MultiVariantTransliterator):
            transliterator = self.pipeline.transliterator
        else:
            print("Multi-variant transliterator not found in pipeline")
            return {}
        
        # Predict variants
        predicted_variants = []
        variant_confidences = []
        
        for brahmi_text in tqdm(test_brahmi_texts, desc="Detecting variants"):
            # Transliterate text
            result = transliterator.transliterate(brahmi_text)
            
            # Extract variant information
            variant = result.get('detected_variant', 'Unknown')
            confidence = result.get('variant_confidence', 0.0)
            
            predicted_variants.append(variant)
            variant_confidences.append(confidence)
        
        # Calculate variant accuracy if labels are provided
        variant_accuracy = 0.0
        
        if variant_labels:
            variant_matches = sum(1 for pred, true in zip(predicted_variants, variant_labels) 
                                if pred == true)
            variant_accuracy = variant_matches / len(test_brahmi_texts) if test_brahmi_texts else 0
        
        # Calculate metrics
        avg_variant_confidence = sum(variant_confidences) / len(variant_confidences) if variant_confidences else 0
        
        # Store metrics
        self.variant_metrics = {
            'variant_accuracy': variant_accuracy,
            'avg_variant_confidence': avg_variant_confidence,
            'predicted_variants': predicted_variants,
            'variant_confidences': variant_confidences
        }
        
        if variant_labels:
            print(f"Variant Detection Accuracy: {variant_accuracy:.4f}")
        
        print(f"Average Variant Confidence: {avg_variant_confidence:.4f}")
        
        return self.variant_metrics
    
    def validate_end_to_end_pipeline(self, test_image_paths, test_devanagari_texts):
        """
        Validate the end-to-end pipeline.
        
        Args:
            test_image_paths (list): Paths to test images
            test_devanagari_texts (list): Expected Devanagari texts
            
        Returns:
            dict: Pipeline metrics
        """
        print("Validating end-to-end pipeline...")
        
        # Process images through the pipeline
        predicted_texts = []
        confidence_scores = []
        processing_times = []
        
        for image_path in tqdm(test_image_paths, desc="Processing images"):
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Failed to load image: {image_path}")
                continue
            
            # Convert to RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Measure processing time
            start_time = time.time()
            
            # Process image through pipeline
            result = self.pipeline.process_image(image)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Extract results
            predicted_text = result.get('transliterated_text', '')
            confidence = result.get('confidence', 0.0)
            
            predicted_texts.append(predicted_text)
            confidence_scores.append(confidence)
            processing_times.append(processing_time)
        
        # Calculate text-level accuracy
        text_matches = sum(1 for pred, true in zip(predicted_texts, test_devanagari_texts) 
                          if pred == true)
        text_accuracy = text_matches / len(test_image_paths) if test_image_paths else 0
        
        # Calculate character-level metrics
        all_true_chars = []
        all_pred_chars = []
        
        for true_text, pred_text in zip(test_devanagari_texts, predicted_texts):
            # Pad shorter text with spaces
            max_len = max(len(true_text), len(pred_text))
            true_padded = true_text.ljust(max_len)
            pred_padded = pred_text.ljust(max_len)
            
            all_true_chars.extend(list(true_padded))
            all_pred_chars.extend(list(pred_padded))
        
        # Calculate overall character-level metrics
        char_accuracy = accuracy_score(all_true_chars, all_pred_chars)
        char_precision, char_recall, char_f1, _ = precision_recall_fscore_support(
            all_true_chars, all_pred_chars, average='weighted')
        
        # Calculate timing metrics
        total_processing_time = sum(processing_times)
        avg_processing_time = total_processing_time / len(test_image_paths) if test_image_paths else 0
        
        # Calculate confidence metrics
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        # Store metrics
        self.pipeline_metrics = {
            'text_accuracy': text_accuracy,
            'char_accuracy': char_accuracy,
            'char_precision': char_precision,
            'char_recall': char_recall,
            'char_f1': char_f1,
            'avg_confidence': avg_confidence,
            'total_processing_time': total_processing_time,
            'avg_processing_time': avg_processing_time
        }
        
        print(f"Pipeline Text Accuracy: {text_accuracy:.4f}")
        print(f"Pipeline Character Accuracy: {char_accuracy:.4f}")
        print(f"Pipeline Character F1 Score: {char_f1:.4f}")
        print(f"Pipeline Average Confidence: {avg_confidence:.4f}")
        print(f"Pipeline Average Processing Time: {avg_processing_time:.4f} seconds")
        
        return self.pipeline_metrics
    
    def validate_feedback_loop(self, test_brahmi_texts, test_devanagari_texts, 
                              corrected_devanagari_texts):
        """
        Validate the feedback loop.
        
        Args:
            test_brahmi_texts (list): Test Brahmi texts
            test_devanagari_texts (list): Initial Devanagari texts
            corrected_devanagari_texts (list): Corrected Devanagari texts
            
        Returns:
            dict: Feedback metrics
        """
        print("Validating feedback loop...")
        
        # Get feedback handler
        if hasattr(self.pipeline, 'feedback_handler'):
            feedback_handler = self.pipeline.feedback_handler
        else:
            print("Feedback handler not found in pipeline")
            return {}
        
        # Get transliterator
        if hasattr(self.pipeline, 'transliterator'):
            transliterator = self.pipeline.transliterator
        else:
            print("Transliterator not found in pipeline")
            return {}
        
        # Measure initial performance
        initial_predicted_texts = []
        
        for brahmi_text in test_brahmi_texts:
            # Transliterate text
            if isinstance(transliterator, MultiVariantTransliterator):
                result = transliterator.transliterate(brahmi_text)
                predicted_text = result['transliterated_text']
            else:
                predicted_text = transliterator.transliterate(brahmi_text)
            
            initial_predicted_texts.append(predicted_text)
        
        # Calculate initial accuracy
        initial_text_matches = sum(1 for pred, true in zip(initial_predicted_texts, test_devanagari_texts) 
                                  if pred == true)
        initial_text_accuracy = initial_text_matches / len(test_brahmi_texts) if test_brahmi_texts else 0
        
        # Process corrections
        for i, (brahmi_text, original_text, corrected_text) in enumerate(
            zip(test_brahmi_texts, test_devanagari_texts, corrected_devanagari_texts)):
            
            # Skip if no correction needed
            if original_text == corrected_text:
                continue
            
            # Create correction
            correction = {
                'brahmi_text': brahmi_text,
                'original_text': original_text,
                'corrected_text': corrected_text,
                'timestamp': pd.Timestamp.now()
            }
            
            # Process correction
            feedback_handler.process_correction(correction)
            
            # Update model if enough corrections
            if (i + 1) % feedback_handler.correction_threshold == 0:
                feedback_handler.update_model()
        
        # Force model update
        feedback_handler.update_model()
        
        # Measure final performance
        final_predicted_texts = []
        
        for brahmi_text in test_brahmi_texts:
            # Transliterate text
            if isinstance(transliterator, MultiVariantTransliterator):
                result = transliterator.transliterate(brahmi_text)
                predicted_text = result['transliterated_text']
            else:
                predicted_text = transliterator.transliterate(brahmi_text)
            
            final_predicted_texts.append(predicted_text)
        
        # Calculate final accuracy
        final_text_matches = sum(1 for pred, true in zip(final_predicted_texts, corrected_devanagari_texts) 
                                if pred == true)
        final_text_accuracy = final_text_matches / len(test_brahmi_texts) if test_brahmi_texts else 0
        
        # Calculate improvement
        accuracy_improvement = final_text_accuracy - initial_text_accuracy
        
        # Store metrics
        feedback_metrics = {
            'initial_text_accuracy': initial_text_accuracy,
            'final_text_accuracy': final_text_accuracy,
            'accuracy_improvement': accuracy_improvement,
            'correction_count': len([1 for o, c in zip(test_devanagari_texts, corrected_devanagari_texts) if o != c])
        }
        
        print(f"Initial Text Accuracy: {initial_text_accuracy:.4f}")
        print(f"Final Text Accuracy: {final_text_accuracy:.4f}")
        print(f"Accuracy Improvement: {accuracy_improvement:.4f}")
        
        return feedback_metrics
    
    def validate_explainable_ai(self, test_brahmi_texts, output_dir=None):
        """
        Validate the explainable AI module.
        
        Args:
            test_brahmi_texts (list): Test Brahmi texts
            output_dir (str, optional): Directory to save explanation files
            
        Returns:
            dict: Explainable AI metrics
        """
        print("Validating explainable AI...")
        
        # Get model explainer
        if hasattr(self.pipeline, 'model_explainer'):
            model_explainer = self.pipeline.model_explainer
        else:
            print("Model explainer not found in pipeline")
            return {}
        
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Generate explanations
        explanation_times = []
        
        for i, brahmi_text in enumerate(tqdm(test_brahmi_texts[:5], desc="Generating explanations")):
            # Measure explanation time
            start_time = time.time()
            
            # Generate explanation
            if output_dir:
                explanation_dir = os.path.join(output_dir, f"explanation_{i}")
                os.makedirs(explanation_dir, exist_ok=True)
                
                explanation = model_explainer.explain_transliteration(brahmi_text, explanation_dir)
            else:
                explanation = model_explainer.explain_transliteration(brahmi_text)
            
            end_time = time.time()
            explanation_time = end_time - start_time
            
            explanation_times.append(explanation_time)
        
        # Calculate timing metrics
        total_explanation_time = sum(explanation_times)
        avg_explanation_time = total_explanation_time / len(explanation_times) if explanation_times else 0
        
        # Store metrics
        explainable_ai_metrics = {
            'total_explanation_time': total_explanation_time,
            'avg_explanation_time': avg_explanation_time
        }
        
        print(f"Average Explanation Time: {avg_explanation_time:.4f} seconds")
        
        return explainable_ai_metrics
    
    def validate_all(self, test_images=None, test_labels=None, 
                    test_brahmi_texts=None, test_devanagari_texts=None,
                    test_image_paths=None, corrected_devanagari_texts=None,
                    variant_labels=None, output_dir=None):
        """
        Validate all components of the pipeline.
        
        Args:
            test_images (numpy.ndarray, optional): Test images for OCR
            test_labels (numpy.ndarray, optional): Test labels for OCR
            test_brahmi_texts (list, optional): Test Brahmi texts
            test_devanagari_texts (list, optional): Test Devanagari texts
            test_image_paths (list, optional): Paths to test images for end-to-end validation
            corrected_devanagari_texts (list, optional): Corrected Devanagari texts for feedback loop
            variant_labels (list, optional): True variant labels
            output_dir (str, optional): Directory to save validation results
            
        Returns:
            dict: All validation metrics
        """
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Validate OCR module
        if test_images is not None and test_labels is not None:
            self.validate_ocr_module(test_images, test_labels)
        
        # Validate transliteration module
        if test_brahmi_texts is not None and test_devanagari_texts is not None:
            self.validate_transliteration_module(test_brahmi_texts, test_devanagari_texts)
        
        # Validate confidence scoring
        if test_brahmi_texts is not None and test_devanagari_texts is not None:
            self.validate_confidence_scoring(test_brahmi_texts, test_devanagari_texts)
        
        # Validate variant support
        if test_brahmi_texts is not None:
            self.validate_variant_support(test_brahmi_texts, variant_labels)
        
        # Validate end-to-end pipeline
        if test_image_paths is not None and test_devanagari_texts is not None:
            self.validate_end_to_end_pipeline(test_image_paths, test_devanagari_texts)
        
        # Validate feedback loop
        if test_brahmi_texts is not None and test_devanagari_texts is not None and corrected_devanagari_texts is not None:
            self.validate_feedback_loop(test_brahmi_texts, test_devanagari_texts, corrected_devanagari_texts)
        
        # Validate explainable AI
        if test_brahmi_texts is not None:
            explainable_ai_dir = os.path.join(output_dir, 'explainable_ai') if output_dir else None
            self.validate_explainable_ai(test_brahmi_texts, explainable_ai_dir)
        
        # Combine all metrics
        all_metrics = {
            'ocr_metrics': self.ocr_metrics,
            'transliteration_metrics': self.transliteration_metrics,
            'confidence_metrics': self.confidence_metrics,
            'variant_metrics': self.variant_metrics,
            'pipeline_metrics': self.pipeline_metrics,
            'timing_metrics': self.timing_metrics
        }
        
        # Save metrics if output directory is specified
        if output_dir:
            metrics_path = os.path.join(output_dir, 'validation_metrics.json')
            
            with open(metrics_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                metrics_json = {}
                
                for category, metrics in all_metrics.items():
                    metrics_json[category] = {}
                    
                    for key, value in metrics.items():
                        if isinstance(value, np.ndarray):
                            metrics_json[category][key] = value.tolist()
                        elif isinstance(value, list) and value and isinstance(value[0], np.ndarray):
                            metrics_json[category][key] = [v.tolist() for v in value]
                        else:
                            metrics_json[category][key] = value
                
                json.dump(metrics_json, f, indent=4)
        
        return all_metrics
    
    def plot_metrics(self, output_dir=None):
        """
        Plot validation metrics.
        
        Args:
            output_dir (str, optional): Directory to save plots
            
        Returns:
            dict: Paths to saved plots
        """
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        plot_paths = {}
        
        # Plot OCR confusion matrix
        if 'confusion_matrix' in self.ocr_metrics:
            plt.figure(figsize=(10, 8))
            sns.heatmap(self.ocr_metrics['confusion_matrix'], annot=True, fmt='d', cmap='Blues')
            plt.xlabel('Predicted Label')
            plt.ylabel('True Label')
            plt.title('OCR Confusion Matrix')
            
            if output_dir:
                plot_path = os.path.join(output_dir, 'ocr_confusion_matrix.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['ocr_confusion_matrix'] = plot_path
            
            plt.close()
        
        # Plot transliteration character accuracies
        if 'character_accuracies' in self.transliteration_metrics:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.transliteration_metrics['character_accuracies'], bins=20, kde=True)
            plt.xlabel('Character Accuracy')
            plt.ylabel('Frequency')
            plt.title('Distribution of Character Accuracies')
            
            if output_dir:
                plot_path = os.path.join(output_dir, 'character_accuracies.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['character_accuracies'] = plot_path
            
            plt.close()
        
        # Plot confidence scores
        if 'confidence_scores' in self.confidence_metrics:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.confidence_metrics['confidence_scores'], bins=20, kde=True)
            plt.xlabel('Confidence Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Confidence Scores')
            
            if output_dir:
                plot_path = os.path.join(output_dir, 'confidence_scores.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['confidence_scores'] = plot_path
            
            plt.close()
        
        # Plot variant confidences
        if 'variant_confidences' in self.variant_metrics:
            plt.figure(figsize=(10, 6))
            sns.histplot(self.variant_metrics['variant_confidences'], bins=20, kde=True)
            plt.xlabel('Variant Confidence')
            plt.ylabel('Frequency')
            plt.title('Distribution of Variant Confidences')
            
            if output_dir:
                plot_path = os.path.join(output_dir, 'variant_confidences.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['variant_confidences'] = plot_path
            
            plt.close()
        
        # Plot processing times
        if 'avg_processing_time' in self.pipeline_metrics:
            # Create bar chart of average processing times
            plt.figure(figsize=(10, 6))
            
            times = []
            labels = []
            
            if 'avg_inference_time' in self.ocr_metrics:
                times.append(self.ocr_metrics['avg_inference_time'])
                labels.append('OCR')
            
            if 'avg_inference_time' in self.transliteration_metrics:
                times.append(self.transliteration_metrics['avg_inference_time'])
                labels.append('Transliteration')
            
            times.append(self.pipeline_metrics['avg_processing_time'])
            labels.append('Full Pipeline')
            
            plt.bar(labels, times)
            plt.xlabel('Component')
            plt.ylabel('Average Processing Time (seconds)')
            plt.title('Average Processing Times')
            
            if output_dir:
                plot_path = os.path.join(output_dir, 'processing_times.png')
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plot_paths['processing_times'] = plot_path
            
            plt.close()
        
        return plot_paths
    
    def create_validation_report(self, output_path):
        """
        Create a comprehensive validation report.
        
        Args:
            output_path (str): Path to save the report
            
        Returns:
            str: Path to the report
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate plots
        plots_dir = os.path.join(os.path.dirname(output_path), 'validation_plots')
        plot_paths = self.plot_metrics(plots_dir)
        
        # Create HTML report
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Brahmi to Devanagari Transliteration Pipeline Validation Report</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }
                h1, h2, h3 {
                    color: #2c3e50;
                }
                .section {
                    margin-bottom: 30px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                table {
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }
                th, td {
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }
                th {
                    background-color: #f2f2f2;
                }
                .metric {
                    font-weight: bold;
                }
                .value {
                    text-align: right;
                }
                .visualization {
                    text-align: center;
                    margin: 20px 0;
                }
                .visualization img {
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }
                .footer {
                    margin-top: 50px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 14px;
                }
            </style>
        </head>
        <body>
            <h1>Brahmi to Devanagari Transliteration Pipeline Validation Report</h1>
            
            <div class="section">
                <h2>Executive Summary</h2>
                <p>This report presents the validation results for the Brahmi to Devanagari transliteration pipeline. The pipeline includes OCR for Brahmi script recognition, sequence-to-sequence transliteration to Devanagari, support for multiple Brahmi variants, confidence scoring, user correction interface, self-improving feedback loop, and explainable AI tools.</p>
                
                <table>
                    <tr>
                        <th>Component</th>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
        """
        
        # Add OCR metrics
        if self.ocr_metrics:
            html += f"""
                    <tr>
                        <td rowspan="3">OCR Module</td>
                        <td class="metric">Accuracy</td>
                        <td class="value">{self.ocr_metrics.get('accuracy', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric">F1 Score</td>
                        <td class="value">{self.ocr_metrics.get('f1', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric">Avg. Inference Time</td>
                        <td class="value">{self.ocr_metrics.get('avg_inference_time', 0):.4f} seconds</td>
                    </tr>
            """
        
        # Add transliteration metrics
        if self.transliteration_metrics:
            html += f"""
                    <tr>
                        <td rowspan="4">Transliteration Module</td>
                        <td class="metric">Text Accuracy</td>
                        <td class="value">{self.transliteration_metrics.get('text_accuracy', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric">Character Accuracy</td>
                        <td class="value">{self.transliteration_metrics.get('char_accuracy', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric">Character F1 Score</td>
                        <td class="value">{self.transliteration_metrics.get('char_f1', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric">Avg. Inference Time</td>
                        <td class="value">{self.transliteration_metrics.get('avg_inference_time', 0):.4f} seconds</td>
                    </tr>
            """
        
        # Add pipeline metrics
        if self.pipeline_metrics:
            html += f"""
                    <tr>
                        <td rowspan="4">End-to-End Pipeline</td>
                        <td class="metric">Text Accuracy</td>
                        <td class="value">{self.pipeline_metrics.get('text_accuracy', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric">Character Accuracy</td>
                        <td class="value">{self.pipeline_metrics.get('char_accuracy', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric">Avg. Confidence</td>
                        <td class="value">{self.pipeline_metrics.get('avg_confidence', 0):.4f}</td>
                    </tr>
                    <tr>
                        <td class="metric">Avg. Processing Time</td>
                        <td class="value">{self.pipeline_metrics.get('avg_processing_time', 0):.4f} seconds</td>
                    </tr>
            """
        
        html += """
                </table>
            </div>
        """
        
        # Add OCR section
        if self.ocr_metrics:
            html += """
            <div class="section">
                <h2>OCR Module Validation</h2>
                <p>The OCR module is responsible for recognizing Brahmi characters from input images. It uses a CNN-based architecture with transfer learning from MobileNetV2.</p>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """
            
            for metric, value in self.ocr_metrics.items():
                if metric != 'confusion_matrix' and not isinstance(value, (list, np.ndarray)):
                    html += f"""
                    <tr>
                        <td class="metric">{metric.replace('_', ' ').title()}</td>
                        <td class="value">{value:.4f if isinstance(value, float) else value}</td>
                    </tr>
                    """
            
            html += """
                </table>
            """
            
            # Add confusion matrix visualization
            if 'ocr_confusion_matrix' in plot_paths:
                html += f"""
                <div class="visualization">
                    <h3>Confusion Matrix</h3>
                    <img src="{os.path.relpath(plot_paths['ocr_confusion_matrix'], os.path.dirname(output_path))}" alt="OCR Confusion Matrix">
                </div>
                """
            
            html += """
            </div>
            """
        
        # Add transliteration section
        if self.transliteration_metrics:
            html += """
            <div class="section">
                <h2>Transliteration Module Validation</h2>
                <p>The transliteration module converts recognized Brahmi characters into Devanagari Unicode text. It uses a sequence-to-sequence model with attention mechanisms.</p>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """
            
            for metric, value in self.transliteration_metrics.items():
                if not isinstance(value, (list, np.ndarray)):
                    html += f"""
                    <tr>
                        <td class="metric">{metric.replace('_', ' ').title()}</td>
                        <td class="value">{value:.4f if isinstance(value, float) else value}</td>
                    </tr>
                    """
            
            html += """
                </table>
            """
            
            # Add character accuracies visualization
            if 'character_accuracies' in plot_paths:
                html += f"""
                <div class="visualization">
                    <h3>Distribution of Character Accuracies</h3>
                    <img src="{os.path.relpath(plot_paths['character_accuracies'], os.path.dirname(output_path))}" alt="Character Accuracies">
                </div>
                """
            
            html += """
            </div>
            """
        
        # Add confidence scoring section
        if self.confidence_metrics:
            html += """
            <div class="section">
                <h2>Confidence Scoring Validation</h2>
                <p>The confidence scoring module provides confidence scores for each character in the transliteration output. It helps identify potential errors and areas for improvement.</p>
                
                <table>
                    <tr>
                        <th>Metric</th>
                        <th>Value</th>
                    </tr>
            """
            
            for metric, value in self.confidence_metrics.items():
                if not isinstance(value, (list, np.ndarray)):
                    html += f"""
                    <tr>
                        <td class="metric">{metric.replace('_', ' ').title()}</td>
                        <td class="value">{value:.4f if isinstance(value, float) else value}</td>
                    </tr>
                    """
            
            html += """
                </table>
            """
            
            # Add confidence scores visualization
            if 'confidence_scores' in plot_paths:
                html += f"""
                <div class="visualization">
                    <h3>Distribution of Confidence Scores</h3>
                    <img src="{os.path.relpath(plot_paths['confidence_scores'], os.path.dirname(output_path))}" alt="Confidence Scores">
                </div>
                """
            
            html += """
            </div>
            """
        
        # Add processing times section
        if 'processing_times' in plot_paths:
            html += f"""
            <div class="section">
                <h2>Processing Times</h2>
                <p>This section shows the average processing times for different components of the pipeline.</p>
                
                <div class="visualization">
                    <img src="{os.path.relpath(plot_paths['processing_times'], os.path.dirname(output_path))}" alt="Processing Times">
                </div>
            </div>
            """
        
        # Add footer
        html += f"""
            <div class="footer">
                <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html)
        
        return output_path


class DatasetValidator:
    """
    Validator for the Brahmi to Devanagari transliteration datasets.
    """
    
    def __init__(self, ocr_data_dir=None, transliteration_data_file=None):
        """
        Initialize the dataset validator.
        
        Args:
            ocr_data_dir (str, optional): Directory containing OCR dataset
            transliteration_data_file (str, optional): File containing transliteration dataset
        """
        self.ocr_data_dir = ocr_data_dir
        self.transliteration_data_file = transliteration_data_file
        
        # Initialize metrics
        self.ocr_dataset_metrics = {}
        self.transliteration_dataset_metrics = {}
    
    def validate_ocr_dataset(self):
        """
        Validate the OCR dataset.
        
        Returns:
            dict: OCR dataset metrics
        """
        print("Validating OCR dataset...")
        
        if not self.ocr_data_dir or not os.path.exists(self.ocr_data_dir):
            print(f"OCR data directory not found: {self.ocr_data_dir}")
            return {}
        
        # Count images and classes
        class_counts = {}
        image_sizes = []
        total_images = 0
        
        for class_dir in os.listdir(self.ocr_data_dir):
            class_path = os.path.join(self.ocr_data_dir, class_dir)
            
            if not os.path.isdir(class_path):
                continue
            
            # Count images in this class
            images = [f for f in os.listdir(class_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            class_counts[class_dir] = len(images)
            total_images += len(images)
            
            # Sample image sizes
            for image_file in images[:10]:  # Sample up to 10 images per class
                image_path = os.path.join(class_path, image_file)
                try:
                    image = cv2.imread(image_path)
                    if image is not None:
                        image_sizes.append(image.shape[:2])  # (height, width)
                except Exception as e:
                    print(f"Error reading image {image_path}: {str(e)}")
        
        # Calculate metrics
        num_classes = len(class_counts)
        avg_images_per_class = total_images / num_classes if num_classes > 0 else 0
        min_class_size = min(class_counts.values()) if class_counts else 0
        max_class_size = max(class_counts.values()) if class_counts else 0
        
        # Calculate average image size
        avg_height = sum(h for h, w in image_sizes) / len(image_sizes) if image_sizes else 0
        avg_width = sum(w for h, w in image_sizes) / len(image_sizes) if image_sizes else 0
        
        # Store metrics
        self.ocr_dataset_metrics = {
            'total_images': total_images,
            'num_classes': num_classes,
            'avg_images_per_class': avg_images_per_class,
            'min_class_size': min_class_size,
            'max_class_size': max_class_size,
            'class_distribution': class_counts,
            'avg_image_height': avg_height,
            'avg_image_width': avg_width
        }
        
        print(f"Total images: {total_images}")
        print(f"Number of classes: {num_classes}")
        print(f"Average images per class: {avg_images_per_class:.2f}")
        print(f"Min class size: {min_class_size}")
        print(f"Max class size: {max_class_size}")
        print(f"Average image size: {avg_width:.2f} x {avg_height:.2f}")
        
        return self.ocr_dataset_metrics
    
    def validate_transliteration_dataset(self):
        """
        Validate the transliteration dataset.
        
        Returns:
            dict: Transliteration dataset metrics
        """
        print("Validating transliteration dataset...")
        
        if not self.transliteration_data_file or not os.path.exists(self.transliteration_data_file):
            print(f"Transliteration data file not found: {self.transliteration_data_file}")
            return {}
        
        # Load dataset
        try:
            if self.transliteration_data_file.endswith('.csv'):
                df = pd.read_csv(self.transliteration_data_file)
            elif self.transliteration_data_file.endswith('.json'):
                df = pd.read_json(self.transliteration_data_file)
            else:
                print(f"Unsupported file format: {self.transliteration_data_file}")
                return {}
        except Exception as e:
            print(f"Error loading transliteration dataset: {str(e)}")
            return {}
        
        # Extract Brahmi and Devanagari columns
        brahmi_col = None
        devanagari_col = None
        
        for col in df.columns:
            if 'brahmi' in col.lower():
                brahmi_col = col
            elif 'devanagari' in col.lower():
                devanagari_col = col
        
        if not brahmi_col or not devanagari_col:
            print("Could not identify Brahmi and Devanagari columns")
            return {}
        
        # Calculate metrics
        total_pairs = len(df)
        
        # Calculate text lengths
        brahmi_lengths = df[brahmi_col].str.len()
        devanagari_lengths = df[devanagari_col].str.len()
        
        avg_brahmi_length = brahmi_lengths.mean()
        avg_devanagari_length = devanagari_lengths.mean()
        
        min_brahmi_length = brahmi_lengths.min()
        max_brahmi_length = brahmi_lengths.max()
        
        min_devanagari_length = devanagari_lengths.min()
        max_devanagari_length = devanagari_lengths.max()
        
        # Count unique characters
        unique_brahmi_chars = set()
        unique_devanagari_chars = set()
        
        for text in df[brahmi_col]:
            unique_brahmi_chars.update(text)
        
        for text in df[devanagari_col]:
            unique_devanagari_chars.update(text)
        
        # Store metrics
        self.transliteration_dataset_metrics = {
            'total_pairs': total_pairs,
            'avg_brahmi_length': avg_brahmi_length,
            'avg_devanagari_length': avg_devanagari_length,
            'min_brahmi_length': min_brahmi_length,
            'max_brahmi_length': max_brahmi_length,
            'min_devanagari_length': min_devanagari_length,
            'max_devanagari_length': max_devanagari_length,
            'unique_brahmi_chars': len(unique_brahmi_chars),
            'unique_devanagari_chars': len(unique_devanagari_chars)
        }
        
        print(f"Total pairs: {total_pairs}")
        print(f"Average Brahmi length: {avg_brahmi_length:.2f}")
        print(f"Average Devanagari length: {avg_devanagari_length:.2f}")
        print(f"Unique Brahmi characters: {len(unique_brahmi_chars)}")
        print(f"Unique Devanagari characters: {len(unique_devanagari_chars)}")
        
        return self.transliteration_dataset_metrics
    
    def validate_all(self, output_path=None):
        """
        Validate all datasets.
        
        Args:
            output_path (str, optional): Path to save validation results
            
        Returns:
            dict: All dataset metrics
        """
        # Validate OCR dataset
        if self.ocr_data_dir:
            self.validate_ocr_dataset()
        
        # Validate transliteration dataset
        if self.transliteration_data_file:
            self.validate_transliteration_dataset()
        
        # Combine all metrics
        all_metrics = {
            'ocr_dataset_metrics': self.ocr_dataset_metrics,
            'transliteration_dataset_metrics': self.transliteration_dataset_metrics
        }
        
        # Save metrics if output path is specified
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w') as f:
                json.dump(all_metrics, f, indent=4)
        
        return all_metrics


if __name__ == "__main__":
    # Example usage
    from src.pipeline import BrahmiToDevanagariPipeline
    
    # Create pipeline
    pipeline = BrahmiToDevanagariPipeline()
    
    # Create validator
    validator = PipelineValidator(pipeline)
    
    # Validate pipeline
    validator.validate_all(
        output_dir='validation_results'
    )
    
    # Create validation report
    validator.create_validation_report('validation_results/validation_report.html')
