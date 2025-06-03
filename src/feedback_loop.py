"""
Self-improving feedback loop for Brahmi to Devanagari transliteration
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import unicodedata
import time
from datetime import datetime
from src.transliteration_model import BrahmiToDevanagariTransliterator
from src.variant_support import MultiVariantTransliterator
from src.confidence_and_correction import ConfidenceScorer

class FeedbackHandler:
    """
    Handler for user feedback to improve the transliteration model.
    """
    
    def __init__(self, transliterator, correction_threshold=10, 
                 auto_update_interval=None, model_save_dir=None):
        """
        Initialize the feedback handler.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
            correction_threshold (int): Number of corrections needed to trigger model update
            auto_update_interval (int): Interval in hours for automatic model update (None to disable)
            model_save_dir (str): Directory to save updated models
        """
        self.transliterator = transliterator
        self.correction_threshold = correction_threshold
        self.auto_update_interval = auto_update_interval
        self.model_save_dir = model_save_dir
        
        # Initialize correction database
        self.corrections = []
        self.last_update_time = time.time()
        
        # Create model save directory if specified
        if self.model_save_dir:
            os.makedirs(self.model_save_dir, exist_ok=True)
    
    def process_correction(self, correction):
        """
        Process a user correction.
        
        Args:
            correction (dict): Correction data with brahmi_text, original_text, and corrected_text
            
        Returns:
            bool: Whether the model was updated
        """
        # Add correction to database
        self.corrections.append(correction)
        
        # Check if update is needed
        if self._should_update_model():
            return self.update_model()
        
        return False
    
    def _should_update_model(self):
        """
        Check if the model should be updated.
        
        Returns:
            bool: Whether the model should be updated
        """
        # Check if enough corrections have been collected
        if len(self.corrections) >= self.correction_threshold:
            return True
        
        # Check if auto-update interval has passed
        if self.auto_update_interval is not None:
            current_time = time.time()
            hours_since_update = (current_time - self.last_update_time) / 3600
            
            if hours_since_update >= self.auto_update_interval:
                return True
        
        return False
    
    def update_model(self):
        """
        Update the model based on collected corrections.
        
        Returns:
            bool: Whether the update was successful
        """
        if not self.corrections:
            return False
        
        try:
            # Extract correction data
            brahmi_texts = [c['brahmi_text'] for c in self.corrections]
            corrected_texts = [c['corrected_text'] for c in self.corrections]
            
            # Check if using multi-variant transliterator
            if isinstance(self.transliterator, MultiVariantTransliterator):
                # Update each variant transliterator
                for variant_name, variant_transliterator in self.transliterator.transliterators.items():
                    self._update_single_transliterator(
                        variant_transliterator, brahmi_texts, corrected_texts, variant_name)
                
                # Update default transliterator
                self._update_single_transliterator(
                    self.transliterator.default_transliterator, 
                    brahmi_texts, corrected_texts, 'default')
            else:
                # Update single transliterator
                self._update_single_transliterator(
                    self.transliterator, brahmi_texts, corrected_texts)
            
            # Update last update time
            self.last_update_time = time.time()
            
            # Clear processed corrections
            self.corrections = []
            
            return True
            
        except Exception as e:
            print(f"Error updating model: {str(e)}")
            return False
    
    def _update_single_transliterator(self, transliterator, brahmi_texts, 
                                     corrected_texts, variant_name=None):
        """
        Update a single transliterator model.
        
        Args:
            transliterator (BrahmiToDevanagariTransliterator): Transliterator to update
            brahmi_texts (list): List of Brahmi texts
            corrected_texts (list): List of corrected Devanagari texts
            variant_name (str, optional): Variant name for multi-variant transliterators
        """
        # Preprocess data
        (encoder_input_train, decoder_input_train, decoder_target_train,
         encoder_input_val, decoder_input_val, decoder_target_val) = transliterator.preprocess_data(
            brahmi_texts, corrected_texts, test_size=0.2)
        
        # Fine-tune the model with a low learning rate
        transliterator.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train for a few epochs
        transliterator.train(
            encoder_input_train, decoder_input_train, decoder_target_train,
            encoder_input_val, decoder_input_val, decoder_target_val,
            epochs=5
        )
        
        # Save updated model if directory is specified
        if self.model_save_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            if variant_name:
                save_path = os.path.join(
                    self.model_save_dir, 
                    f"transliterator_{variant_name}_{timestamp}")
            else:
                save_path = os.path.join(
                    self.model_save_dir, 
                    f"transliterator_{timestamp}")
                
            transliterator.save_model(save_path)
    
    def save_corrections(self, file_path):
        """
        Save collected corrections to a file.
        
        Args:
            file_path (str): Path to save the corrections
        """
        # Convert to DataFrame
        df = pd.DataFrame(self.corrections)
        
        # Save to file
        df.to_csv(file_path, index=False)
    
    def load_corrections(self, file_path):
        """
        Load corrections from a file.
        
        Args:
            file_path (str): Path to load the corrections from
        """
        # Load from file
        df = pd.read_csv(file_path)
        
        # Convert to list of dictionaries
        self.corrections = df.to_dict('records')


class ActiveLearningSelector:
    """
    Selector for active learning samples based on model uncertainty.
    """
    
    def __init__(self, transliterator, confidence_scorer):
        """
        Initialize the active learning selector.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
            confidence_scorer (ConfidenceScorer): Confidence scorer
        """
        self.transliterator = transliterator
        self.confidence_scorer = confidence_scorer
    
    def select_samples(self, brahmi_texts, selection_count=10, threshold=0.8):
        """
        Select samples for active learning based on model uncertainty.
        
        Args:
            brahmi_texts (list): List of Brahmi texts
            selection_count (int): Number of samples to select
            threshold (float): Confidence threshold (0.0 to 1.0)
            
        Returns:
            list: Selected Brahmi texts
        """
        # Calculate confidence for each text
        confidences = []
        
        for text in brahmi_texts:
            # Transliterate text
            if isinstance(self.transliterator, MultiVariantTransliterator):
                result = self.transliterator.transliterate(text)
                devanagari_text = result['transliterated_text']
            else:
                devanagari_text = self.transliterator.transliterate(text)
            
            # Calculate confidence scores
            confidence_scores = self.confidence_scorer.calculate_character_confidence(
                text, devanagari_text)
            
            # Calculate average confidence
            avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 1.0
            
            confidences.append(avg_confidence)
        
        # Create DataFrame with texts and confidences
        df = pd.DataFrame({
            'text': brahmi_texts,
            'confidence': confidences
        })
        
        # Filter by threshold
        low_confidence_df = df[df['confidence'] < threshold]
        
        # If not enough low confidence samples, take the lowest confidence samples
        if len(low_confidence_df) < selection_count:
            df = df.sort_values('confidence')
            selected_texts = df['text'].head(selection_count).tolist()
        else:
            # Randomly select from low confidence samples
            selected_texts = low_confidence_df['text'].sample(
                min(selection_count, len(low_confidence_df))).tolist()
        
        return selected_texts


class FewShotFineTuner:
    """
    Few-shot fine-tuning for the transliteration model.
    """
    
    def __init__(self, transliterator):
        """
        Initialize the few-shot fine-tuner.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
        """
        self.transliterator = transliterator
    
    def fine_tune(self, brahmi_texts, devanagari_texts, epochs=3, learning_rate=0.0001):
        """
        Perform few-shot fine-tuning on the transliteration model.
        
        Args:
            brahmi_texts (list): List of Brahmi texts
            devanagari_texts (list): List of corresponding Devanagari texts
            epochs (int): Number of fine-tuning epochs
            learning_rate (float): Learning rate for fine-tuning
            
        Returns:
            dict: Fine-tuning metrics
        """
        # Check if using multi-variant transliterator
        if isinstance(self.transliterator, MultiVariantTransliterator):
            # Fine-tune each variant transliterator
            metrics = {}
            
            for variant_name, variant_transliterator in self.transliterator.transliterators.items():
                variant_metrics = self._fine_tune_single_transliterator(
                    variant_transliterator, brahmi_texts, devanagari_texts, 
                    epochs, learning_rate)
                
                metrics[variant_name] = variant_metrics
            
            # Fine-tune default transliterator
            default_metrics = self._fine_tune_single_transliterator(
                self.transliterator.default_transliterator, 
                brahmi_texts, devanagari_texts, epochs, learning_rate)
            
            metrics['default'] = default_metrics
            
            return metrics
        else:
            # Fine-tune single transliterator
            return self._fine_tune_single_transliterator(
                self.transliterator, brahmi_texts, devanagari_texts, 
                epochs, learning_rate)
    
    def _fine_tune_single_transliterator(self, transliterator, brahmi_texts, 
                                        devanagari_texts, epochs, learning_rate):
        """
        Fine-tune a single transliterator model.
        
        Args:
            transliterator (BrahmiToDevanagariTransliterator): Transliterator to fine-tune
            brahmi_texts (list): List of Brahmi texts
            devanagari_texts (list): List of corresponding Devanagari texts
            epochs (int): Number of fine-tuning epochs
            learning_rate (float): Learning rate for fine-tuning
            
        Returns:
            dict: Fine-tuning metrics
        """
        # Preprocess data
        (encoder_input_train, decoder_input_train, decoder_target_train,
         encoder_input_val, decoder_input_val, decoder_target_val) = transliterator.preprocess_data(
            brahmi_texts, devanagari_texts, test_size=0.2)
        
        # Set a low learning rate for fine-tuning
        transliterator.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Fine-tune the model
        history = transliterator.train(
            encoder_input_train, decoder_input_train, decoder_target_train,
            encoder_input_val, decoder_input_val, decoder_target_val,
            epochs=epochs
        )
        
        # Extract metrics
        metrics = {
            'final_loss': history.history['loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        }
        
        return metrics


class ContinualLearningManager:
    """
    Manager for continual learning of the transliteration model.
    """
    
    def __init__(self, transliterator, feedback_handler, active_learning_selector=None,
                 few_shot_fine_tuner=None, model_save_dir=None):
        """
        Initialize the continual learning manager.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
            feedback_handler (FeedbackHandler): Feedback handler
            active_learning_selector (ActiveLearningSelector, optional): Active learning selector
            few_shot_fine_tuner (FewShotFineTuner, optional): Few-shot fine-tuner
            model_save_dir (str, optional): Directory to save models
        """
        self.transliterator = transliterator
        self.feedback_handler = feedback_handler
        self.active_learning_selector = active_learning_selector
        self.few_shot_fine_tuner = few_shot_fine_tuner
        self.model_save_dir = model_save_dir
        
        # Create model save directory if specified
        if self.model_save_dir:
            os.makedirs(self.model_save_dir, exist_ok=True)
        
        # Initialize learning history
        self.learning_history = []
    
    def update_from_corrections(self):
        """
        Update the model from collected corrections.
        
        Returns:
            bool: Whether the update was successful
        """
        return self.feedback_handler.update_model()
    
    def update_from_active_learning(self, brahmi_texts, annotator_func, 
                                   selection_count=10, threshold=0.8):
        """
        Update the model using active learning.
        
        Args:
            brahmi_texts (list): List of Brahmi texts
            annotator_func (callable): Function to annotate selected texts
            selection_count (int): Number of samples to select
            threshold (float): Confidence threshold (0.0 to 1.0)
            
        Returns:
            bool: Whether the update was successful
        """
        if self.active_learning_selector is None:
            return False
        
        try:
            # Select samples for active learning
            selected_texts = self.active_learning_selector.select_samples(
                brahmi_texts, selection_count, threshold)
            
            # Annotate selected texts
            annotated_pairs = []
            
            for text in selected_texts:
                # Get annotation from annotator function
                devanagari_text = annotator_func(text)
                
                if devanagari_text:
                    annotated_pairs.append((text, devanagari_text))
            
            if not annotated_pairs:
                return False
            
            # Extract texts and annotations
            selected_brahmi_texts = [pair[0] for pair in annotated_pairs]
            selected_devanagari_texts = [pair[1] for pair in annotated_pairs]
            
            # Fine-tune the model
            if self.few_shot_fine_tuner is not None:
                metrics = self.few_shot_fine_tuner.fine_tune(
                    selected_brahmi_texts, selected_devanagari_texts)
                
                # Record learning event
                learning_event = {
                    'type': 'active_learning',
                    'timestamp': datetime.now(),
                    'sample_count': len(selected_brahmi_texts),
                    'metrics': metrics
                }
                
                self.learning_history.append(learning_event)
                
                # Save updated model if directory is specified
                if self.model_save_dir:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    save_path = os.path.join(
                        self.model_save_dir, 
                        f"transliterator_active_learning_{timestamp}")
                    
                    if isinstance(self.transliterator, MultiVariantTransliterator):
                        self.transliterator.save(save_path)
                    else:
                        self.transliterator.save_model(save_path)
                
                return True
            
            return False
            
        except Exception as e:
            print(f"Error in active learning update: {str(e)}")
            return False
    
    def update_from_few_shot(self, brahmi_texts, devanagari_texts, epochs=3):
        """
        Update the model using few-shot fine-tuning.
        
        Args:
            brahmi_texts (list): List of Brahmi texts
            devanagari_texts (list): List of corresponding Devanagari texts
            epochs (int): Number of fine-tuning epochs
            
        Returns:
            bool: Whether the update was successful
        """
        if self.few_shot_fine_tuner is None:
            return False
        
        try:
            # Fine-tune the model
            metrics = self.few_shot_fine_tuner.fine_tune(
                brahmi_texts, devanagari_texts, epochs=epochs)
            
            # Record learning event
            learning_event = {
                'type': 'few_shot',
                'timestamp': datetime.now(),
                'sample_count': len(brahmi_texts),
                'metrics': metrics
            }
            
            self.learning_history.append(learning_event)
            
            # Save updated model if directory is specified
            if self.model_save_dir:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(
                    self.model_save_dir, 
                    f"transliterator_few_shot_{timestamp}")
                
                if isinstance(self.transliterator, MultiVariantTransliterator):
                    self.transliterator.save(save_path)
                else:
                    self.transliterator.save_model(save_path)
            
            return True
            
        except Exception as e:
            print(f"Error in few-shot update: {str(e)}")
            return False
    
    def save_learning_history(self, file_path):
        """
        Save learning history to a file.
        
        Args:
            file_path (str): Path to save the learning history
        """
        # Convert timestamps to strings
        history_copy = []
        
        for event in self.learning_history:
            event_copy = event.copy()
            event_copy['timestamp'] = event_copy['timestamp'].isoformat()
            history_copy.append(event_copy)
        
        # Save to file
        with open(file_path, 'w') as f:
            json.dump(history_copy, f, indent=4)
    
    def load_learning_history(self, file_path):
        """
        Load learning history from a file.
        
        Args:
            file_path (str): Path to load the learning history from
        """
        # Load from file
        with open(file_path, 'r') as f:
            history_copy = json.load(f)
        
        # Convert timestamps back to datetime objects
        self.learning_history = []
        
        for event_copy in history_copy:
            event = event_copy.copy()
            event['timestamp'] = datetime.fromisoformat(event['timestamp'])
            self.learning_history.append(event)
    
    def plot_learning_curve(self, save_path=None):
        """
        Plot the learning curve.
        
        Args:
            save_path (str, optional): Path to save the plot
        """
        if not self.learning_history:
            print("No learning history available")
            return
        
        # Extract timestamps and metrics
        timestamps = [event['timestamp'] for event in self.learning_history]
        
        # Check if using multi-variant transliterator
        if isinstance(self.transliterator, MultiVariantTransliterator):
            # Plot metrics for default transliterator
            accuracies = []
            
            for event in self.learning_history:
                if 'default' in event['metrics']:
                    accuracies.append(event['metrics']['default']['final_val_accuracy'])
                else:
                    # Use average of variant accuracies
                    variant_accuracies = [
                        metrics['final_val_accuracy'] 
                        for variant, metrics in event['metrics'].items()
                        if variant != 'default'
                    ]
                    
                    if variant_accuracies:
                        accuracies.append(sum(variant_accuracies) / len(variant_accuracies))
                    else:
                        accuracies.append(None)
        else:
            # Plot metrics for single transliterator
            accuracies = [event['metrics']['final_val_accuracy'] 
                         for event in self.learning_history]
        
        # Create figure
        plt.figure(figsize=(10, 6))
        plt.plot(timestamps, accuracies, 'o-', label='Validation Accuracy')
        plt.xlabel('Time')
        plt.ylabel('Accuracy')
        plt.title('Learning Curve')
        plt.legend()
        plt.grid(True)
        
        # Format x-axis
        plt.gcf().autofmt_xdate()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()


if __name__ == "__main__":
    # Example usage
    from src.transliteration_model import BrahmiToDevanagariTransliterator
    from src.confidence_and_correction import ConfidenceScorer
    
    # Create a transliterator
    transliterator = BrahmiToDevanagariTransliterator()
    
    # Create a confidence scorer
    confidence_scorer = ConfidenceScorer(transliterator)
    
    # Create a feedback handler
    feedback_handler = FeedbackHandler(
        transliterator, 
        correction_threshold=10,
        auto_update_interval=24,  # Update every 24 hours
        model_save_dir='models/feedback'
    )
    
    # Create an active learning selector
    active_learning_selector = ActiveLearningSelector(
        transliterator, confidence_scorer)
    
    # Create a few-shot fine-tuner
    few_shot_fine_tuner = FewShotFineTuner(transliterator)
    
    # Create a continual learning manager
    continual_learning_manager = ContinualLearningManager(
        transliterator,
        feedback_handler,
        active_learning_selector,
        few_shot_fine_tuner,
        model_save_dir='models/continual'
    )
    
    # Example of processing a correction
    correction = {
        'brahmi_text': '𑀓𑀸𑀯𑀺',
        'original_text': 'कावि',
        'corrected_text': 'कावी',
        'timestamp': pd.Timestamp.now()
    }
    
    feedback_handler.process_correction(correction)
    
    # Example of active learning update
    def mock_annotator(text):
        # In a real scenario, this would be a human annotator
        return 'कावी' if text == '𑀓𑀸𑀯𑀺' else None
    
    brahmi_texts = ['𑀓𑀸𑀯𑀺', '𑀅𑀲𑁄𑀓', '𑀥𑀼𑀤𑁆𑀥']
    
    continual_learning_manager.update_from_active_learning(
        brahmi_texts, mock_annotator, selection_count=2)
    
    # Example of few-shot update
    brahmi_texts = ['𑀓𑀸𑀯𑀺', '𑀅𑀲𑁄𑀓']
    devanagari_texts = ['कावी', 'अशोक']
    
    continual_learning_manager.update_from_few_shot(
        brahmi_texts, devanagari_texts, epochs=3)
