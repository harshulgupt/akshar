"""
Data augmentation and semi-supervised learning techniques for Brahmi to Devanagari transliteration
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import cv2
from PIL import Image, ImageOps, ImageEnhance, ImageFilter
import random
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import unicodedata
from tqdm import tqdm
import albumentations as A
from src.transliteration_model import BrahmiToDevanagariTransliterator
from src.variant_support import MultiVariantTransliterator

class ImageAugmenter:
    """
    Augmenter for Brahmi character images to improve OCR model robustness.
    """
    
    def __init__(self, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
                 zoom_range=0.1, shear_range=10, brightness_range=(0.8, 1.2),
                 noise_range=(0, 0.05), blur_range=(0, 1.0)):
        """
        Initialize the image augmenter.
        
        Args:
            rotation_range (float): Maximum rotation angle in degrees
            width_shift_range (float): Maximum horizontal shift as fraction of width
            height_shift_range (float): Maximum vertical shift as fraction of height
            zoom_range (float): Maximum zoom as fraction of original size
            shear_range (float): Maximum shear angle in degrees
            brightness_range (tuple): Range of brightness adjustment (min, max)
            noise_range (tuple): Range of noise intensity (min, max)
            blur_range (tuple): Range of blur intensity (min, max)
        """
        self.rotation_range = rotation_range
        self.width_shift_range = width_shift_range
        self.height_shift_range = height_shift_range
        self.zoom_range = zoom_range
        self.shear_range = shear_range
        self.brightness_range = brightness_range
        self.noise_range = noise_range
        self.blur_range = blur_range
        
        # Set up albumentations transform pipeline
        self.transform = A.Compose([
            A.Rotate(limit=rotation_range, p=0.7),
            A.ShiftScaleRotate(
                shift_limit=max(width_shift_range, height_shift_range),
                scale_limit=zoom_range,
                rotate_limit=0,  # Already handled by Rotate
                p=0.7
            ),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_range[1] - 1.0,
                contrast_limit=0.2,
                p=0.7
            ),
            A.GaussNoise(
                var_limit=(
                    int(noise_range[0] * 255 * 255),
                    int(noise_range[1] * 255 * 255)
                ),
                p=0.5
            ),
            A.GaussianBlur(
                blur_limit=(3, 7),
                sigma_limit=blur_range,
                p=0.5
            ),
            A.ElasticTransform(
                alpha=1,
                sigma=50,
                alpha_affine=50,
                p=0.3
            ),
            A.GridDistortion(p=0.3),
            A.OpticalDistortion(p=0.3),
            A.CLAHE(p=0.3),
            A.Sharpen(p=0.3),
            A.Cutout(
                num_holes=1,
                max_h_size=10,
                max_w_size=10,
                p=0.2
            )
        ])
    
    def augment_image(self, image):
        """
        Apply augmentation to a single image.
        
        Args:
            image (numpy.ndarray): Input image (H, W, C) or (H, W)
            
        Returns:
            numpy.ndarray: Augmented image
        """
        # Ensure image has 3 dimensions
        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=-1)
        
        # Apply transformations
        augmented = self.transform(image=image)['image']
        
        # Return image with original dimensions
        if len(image.shape) == 3 and image.shape[2] == 1:
            augmented = augmented.squeeze(axis=-1)
        
        return augmented
    
    def augment_batch(self, images, count=1):
        """
        Apply augmentation to a batch of images.
        
        Args:
            images (numpy.ndarray): Batch of images (N, H, W, C) or (N, H, W)
            count (int): Number of augmented versions to generate per image
            
        Returns:
            numpy.ndarray: Augmented images
        """
        # Ensure images have 4 dimensions
        original_shape = images.shape
        if len(original_shape) == 3:
            images = np.expand_dims(images, axis=-1)
        
        # Generate augmented images
        augmented_images = []
        
        for image in images:
            for _ in range(count):
                augmented = self.augment_image(image)
                augmented_images.append(augmented)
        
        # Stack augmented images
        augmented_batch = np.stack(augmented_images)
        
        # Return images with original dimensions
        if len(original_shape) == 3:
            augmented_batch = augmented_batch.squeeze(axis=-1)
        
        return augmented_batch
    
    def visualize_augmentations(self, image, count=5, save_path=None):
        """
        Visualize augmentations of a single image.
        
        Args:
            image (numpy.ndarray): Input image
            count (int): Number of augmented versions to generate
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        # Generate augmented images
        augmented_images = [image]
        
        for _ in range(count):
            augmented = self.augment_image(image)
            augmented_images.append(augmented)
        
        # Create figure
        fig, axes = plt.subplots(1, count + 1, figsize=(3 * (count + 1), 3))
        
        # Plot original and augmented images
        for i, (ax, img) in enumerate(zip(axes, augmented_images)):
            if len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1):
                ax.imshow(img, cmap='gray')
            else:
                ax.imshow(img)
            
            ax.set_title('Original' if i == 0 else f'Augmented {i}')
            ax.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


class TextAugmenter:
    """
    Augmenter for Brahmi and Devanagari text to improve transliteration model robustness.
    """
    
    def __init__(self, brahmi_chars=None, devanagari_chars=None, 
                 char_swap_prob=0.05, char_drop_prob=0.03, char_repeat_prob=0.03):
        """
        Initialize the text augmenter.
        
        Args:
            brahmi_chars (list, optional): List of Brahmi characters for random insertion
            devanagari_chars (list, optional): List of Devanagari characters for random insertion
            char_swap_prob (float): Probability of swapping adjacent characters
            char_drop_prob (float): Probability of dropping a character
            char_repeat_prob (float): Probability of repeating a character
        """
        self.brahmi_chars = brahmi_chars or []
        self.devanagari_chars = devanagari_chars or []
        self.char_swap_prob = char_swap_prob
        self.char_drop_prob = char_drop_prob
        self.char_repeat_prob = char_repeat_prob
    
    def augment_text(self, text, is_brahmi=True):
        """
        Apply augmentation to a single text.
        
        Args:
            text (str): Input text
            is_brahmi (bool): Whether the text is in Brahmi script
            
        Returns:
            str: Augmented text
        """
        chars = list(text)
        
        # Apply character swapping
        if len(chars) >= 2:
            for i in range(len(chars) - 1):
                if random.random() < self.char_swap_prob:
                    chars[i], chars[i + 1] = chars[i + 1], chars[i]
        
        # Apply character dropping
        if len(chars) >= 3:  # Ensure we don't drop too many characters
            i = 0
            while i < len(chars):
                if random.random() < self.char_drop_prob:
                    chars.pop(i)
                else:
                    i += 1
        
        # Apply character repetition
        i = 0
        while i < len(chars):
            if random.random() < self.char_repeat_prob:
                chars.insert(i, chars[i])
                i += 2
            else:
                i += 1
        
        # Apply random character insertion
        char_pool = self.brahmi_chars if is_brahmi else self.devanagari_chars
        if char_pool:
            for i in range(len(chars) + 1):
                if random.random() < 0.02:  # Low probability for insertion
                    chars.insert(i, random.choice(char_pool))
        
        return ''.join(chars)
    
    def augment_pair(self, brahmi_text, devanagari_text):
        """
        Apply augmentation to a pair of Brahmi and Devanagari texts.
        
        Args:
            brahmi_text (str): Brahmi text
            devanagari_text (str): Devanagari text
            
        Returns:
            tuple: (augmented_brahmi_text, augmented_devanagari_text)
        """
        # Augment Brahmi text
        augmented_brahmi = self.augment_text(brahmi_text, is_brahmi=True)
        
        # Augment Devanagari text
        augmented_devanagari = self.augment_text(devanagari_text, is_brahmi=False)
        
        return augmented_brahmi, augmented_devanagari
    
    def augment_dataset(self, brahmi_texts, devanagari_texts, count=1):
        """
        Apply augmentation to a dataset of Brahmi-Devanagari pairs.
        
        Args:
            brahmi_texts (list): List of Brahmi texts
            devanagari_texts (list): List of Devanagari texts
            count (int): Number of augmented versions to generate per pair
            
        Returns:
            tuple: (augmented_brahmi_texts, augmented_devanagari_texts)
        """
        augmented_brahmi = []
        augmented_devanagari = []
        
        for brahmi, devanagari in zip(brahmi_texts, devanagari_texts):
            # Add original pair
            augmented_brahmi.append(brahmi)
            augmented_devanagari.append(devanagari)
            
            # Add augmented pairs
            for _ in range(count):
                aug_brahmi, aug_devanagari = self.augment_pair(brahmi, devanagari)
                augmented_brahmi.append(aug_brahmi)
                augmented_devanagari.append(aug_devanagari)
        
        return augmented_brahmi, augmented_devanagari


class SemiSupervisedLearner:
    """
    Semi-supervised learning for Brahmi to Devanagari transliteration.
    """
    
    def __init__(self, transliterator, confidence_threshold=0.8):
        """
        Initialize the semi-supervised learner.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
            confidence_threshold (float): Confidence threshold for pseudo-labeling
        """
        self.transliterator = transliterator
        self.confidence_threshold = confidence_threshold
    
    def generate_pseudo_labels(self, unlabeled_brahmi_texts, confidence_scorer=None):
        """
        Generate pseudo-labels for unlabeled Brahmi texts.
        
        Args:
            unlabeled_brahmi_texts (list): List of unlabeled Brahmi texts
            confidence_scorer (ConfidenceScorer, optional): Confidence scorer
            
        Returns:
            tuple: (pseudo_labeled_brahmi_texts, pseudo_labeled_devanagari_texts, confidences)
        """
        pseudo_labeled_brahmi = []
        pseudo_labeled_devanagari = []
        confidences = []
        
        for text in tqdm(unlabeled_brahmi_texts, desc="Generating pseudo-labels"):
            # Transliterate text
            if isinstance(self.transliterator, MultiVariantTransliterator):
                result = self.transliterator.transliterate(text)
                devanagari_text = result['transliterated_text']
            else:
                devanagari_text = self.transliterator.transliterate(text)
            
            # Calculate confidence
            if confidence_scorer is not None:
                confidence_scores = confidence_scorer.calculate_character_confidence(
                    text, devanagari_text)
                avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            else:
                # If no confidence scorer is provided, use a default high confidence
                avg_confidence = 0.9
            
            # Add to pseudo-labeled dataset if confidence is high enough
            if avg_confidence >= self.confidence_threshold:
                pseudo_labeled_brahmi.append(text)
                pseudo_labeled_devanagari.append(devanagari_text)
                confidences.append(avg_confidence)
        
        return pseudo_labeled_brahmi, pseudo_labeled_devanagari, confidences
    
    def train_with_pseudo_labels(self, labeled_brahmi_texts, labeled_devanagari_texts,
                                unlabeled_brahmi_texts, confidence_scorer=None,
                                pseudo_label_weight=0.5, epochs=5):
        """
        Train the transliteration model with pseudo-labels.
        
        Args:
            labeled_brahmi_texts (list): List of labeled Brahmi texts
            labeled_devanagari_texts (list): List of labeled Devanagari texts
            unlabeled_brahmi_texts (list): List of unlabeled Brahmi texts
            confidence_scorer (ConfidenceScorer, optional): Confidence scorer
            pseudo_label_weight (float): Weight for pseudo-labeled data in loss function
            epochs (int): Number of training epochs
            
        Returns:
            dict: Training metrics
        """
        # Generate pseudo-labels
        pseudo_brahmi, pseudo_devanagari, confidences = self.generate_pseudo_labels(
            unlabeled_brahmi_texts, confidence_scorer)
        
        print(f"Generated {len(pseudo_brahmi)} pseudo-labels from {len(unlabeled_brahmi_texts)} unlabeled texts")
        
        if not pseudo_brahmi:
            print("No pseudo-labels generated, skipping training")
            return {}
        
        # Combine labeled and pseudo-labeled data
        combined_brahmi = labeled_brahmi_texts + pseudo_brahmi
        combined_devanagari = labeled_devanagari_texts + pseudo_devanagari
        
        # Create sample weights (1.0 for labeled, pseudo_label_weight for pseudo-labeled)
        sample_weights = np.ones(len(combined_brahmi))
        sample_weights[len(labeled_brahmi_texts):] = pseudo_label_weight
        
        # Train the model
        if isinstance(self.transliterator, MultiVariantTransliterator):
            # Train each variant transliterator
            metrics = {}
            
            for variant_name, variant_transliterator in self.transliterator.transliterators.items():
                variant_metrics = self._train_single_transliterator(
                    variant_transliterator, combined_brahmi, combined_devanagari, 
                    sample_weights, epochs)
                
                metrics[variant_name] = variant_metrics
            
            # Train default transliterator
            default_metrics = self._train_single_transliterator(
                self.transliterator.default_transliterator, 
                combined_brahmi, combined_devanagari, sample_weights, epochs)
            
            metrics['default'] = default_metrics
            
            return metrics
        else:
            # Train single transliterator
            return self._train_single_transliterator(
                self.transliterator, combined_brahmi, combined_devanagari, 
                sample_weights, epochs)
    
    def _train_single_transliterator(self, transliterator, brahmi_texts, 
                                    devanagari_texts, sample_weights, epochs):
        """
        Train a single transliterator model.
        
        Args:
            transliterator (BrahmiToDevanagariTransliterator): Transliterator to train
            brahmi_texts (list): List of Brahmi texts
            devanagari_texts (list): List of Devanagari texts
            sample_weights (numpy.ndarray): Sample weights
            epochs (int): Number of training epochs
            
        Returns:
            dict: Training metrics
        """
        # Preprocess data
        (encoder_input_train, decoder_input_train, decoder_target_train,
         encoder_input_val, decoder_input_val, decoder_target_val) = transliterator.preprocess_data(
            brahmi_texts, devanagari_texts, test_size=0.2)
        
        # Split sample weights into train and validation
        train_indices = np.arange(len(brahmi_texts))
        train_indices, val_indices = train_test_split(
            train_indices, test_size=0.2, random_state=42)
        
        train_weights = sample_weights[train_indices]
        
        # Compile the model with sample weights
        transliterator.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Train the model with sample weights
        history = transliterator.model.fit(
            [encoder_input_train, decoder_input_train],
            decoder_target_train,
            batch_size=32,
            epochs=epochs,
            validation_data=([encoder_input_val, decoder_input_val], decoder_target_val),
            sample_weight=train_weights
        )
        
        # Extract metrics
        metrics = {
            'final_loss': history.history['loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'final_val_loss': history.history['val_loss'][-1],
            'final_val_accuracy': history.history['val_accuracy'][-1]
        }
        
        return metrics


class DataAugmentationManager:
    """
    Manager for data augmentation and semi-supervised learning.
    """
    
    def __init__(self, image_augmenter=None, text_augmenter=None, 
                 semi_supervised_learner=None):
        """
        Initialize the data augmentation manager.
        
        Args:
            image_augmenter (ImageAugmenter, optional): Image augmenter
            text_augmenter (TextAugmenter, optional): Text augmenter
            semi_supervised_learner (SemiSupervisedLearner, optional): Semi-supervised learner
        """
        self.image_augmenter = image_augmenter
        self.text_augmenter = text_augmenter
        self.semi_supervised_learner = semi_supervised_learner
    
    def augment_ocr_dataset(self, images, labels, count=1):
        """
        Augment OCR dataset.
        
        Args:
            images (numpy.ndarray): Images
            labels (numpy.ndarray): Labels
            count (int): Number of augmented versions to generate per image
            
        Returns:
            tuple: (augmented_images, augmented_labels)
        """
        if self.image_augmenter is None:
            print("Image augmenter not initialized")
            return images, labels
        
        # Augment images
        augmented_images = self.image_augmenter.augment_batch(images, count)
        
        # Repeat labels for each augmented image
        augmented_labels = np.repeat(labels, count + 1, axis=0)
        
        return augmented_images, augmented_labels
    
    def augment_transliteration_dataset(self, brahmi_texts, devanagari_texts, count=1):
        """
        Augment transliteration dataset.
        
        Args:
            brahmi_texts (list): List of Brahmi texts
            devanagari_texts (list): List of Devanagari texts
            count (int): Number of augmented versions to generate per pair
            
        Returns:
            tuple: (augmented_brahmi_texts, augmented_devanagari_texts)
        """
        if self.text_augmenter is None:
            print("Text augmenter not initialized")
            return brahmi_texts, devanagari_texts
        
        # Augment texts
        augmented_brahmi, augmented_devanagari = self.text_augmenter.augment_dataset(
            brahmi_texts, devanagari_texts, count)
        
        return augmented_brahmi, augmented_devanagari
    
    def train_with_semi_supervised_learning(self, labeled_brahmi_texts, 
                                           labeled_devanagari_texts,
                                           unlabeled_brahmi_texts,
                                           confidence_scorer=None,
                                           pseudo_label_weight=0.5,
                                           epochs=5):
        """
        Train with semi-supervised learning.
        
        Args:
            labeled_brahmi_texts (list): List of labeled Brahmi texts
            labeled_devanagari_texts (list): List of labeled Devanagari texts
            unlabeled_brahmi_texts (list): List of unlabeled Brahmi texts
            confidence_scorer (ConfidenceScorer, optional): Confidence scorer
            pseudo_label_weight (float): Weight for pseudo-labeled data in loss function
            epochs (int): Number of training epochs
            
        Returns:
            dict: Training metrics
        """
        if self.semi_supervised_learner is None:
            print("Semi-supervised learner not initialized")
            return {}
        
        # Train with pseudo-labels
        return self.semi_supervised_learner.train_with_pseudo_labels(
            labeled_brahmi_texts, labeled_devanagari_texts,
            unlabeled_brahmi_texts, confidence_scorer,
            pseudo_label_weight, epochs)
    
    def save_augmented_dataset(self, images, labels, save_dir, prefix='augmented'):
        """
        Save augmented OCR dataset.
        
        Args:
            images (numpy.ndarray): Images
            labels (numpy.ndarray): Labels
            save_dir (str): Directory to save the dataset
            prefix (str): Prefix for saved files
            
        Returns:
            str: Path to the saved dataset
        """
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Save images
        for i, (image, label) in enumerate(zip(images, labels)):
            # Create label directory if it doesn't exist
            label_dir = os.path.join(save_dir, str(label))
            os.makedirs(label_dir, exist_ok=True)
            
            # Save image
            image_path = os.path.join(label_dir, f"{prefix}_{i}.png")
            
            # Ensure image is in the correct format
            if len(image.shape) == 2:
                # Grayscale image
                cv2.imwrite(image_path, image)
            else:
                # RGB image
                cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        
        # Save metadata
        metadata = {
            'num_images': len(images),
            'num_classes': len(np.unique(labels)),
            'image_shape': images.shape[1:],
            'prefix': prefix
        }
        
        metadata_path = os.path.join(save_dir, f"{prefix}_metadata.json")
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
        
        return save_dir
    
    def save_augmented_transliteration_dataset(self, brahmi_texts, devanagari_texts, 
                                              save_path, format='csv'):
        """
        Save augmented transliteration dataset.
        
        Args:
            brahmi_texts (list): List of Brahmi texts
            devanagari_texts (list): List of Devanagari texts
            save_path (str): Path to save the dataset
            format (str): Format to save the dataset ('csv' or 'jsonl')
            
        Returns:
            str: Path to the saved dataset
        """
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        if format.lower() == 'csv':
            # Save as CSV
            df = pd.DataFrame({
                'brahmi_text': brahmi_texts,
                'devanagari_text': devanagari_texts
            })
            
            df.to_csv(save_path, index=False)
        
        elif format.lower() == 'jsonl':
            # Save as JSONL
            with open(save_path, 'w', encoding='utf-8') as f:
                for brahmi, devanagari in zip(brahmi_texts, devanagari_texts):
                    json.dump({
                        'brahmi_text': brahmi,
                        'devanagari_text': devanagari
                    }, f, ensure_ascii=False)
                    f.write('\n')
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        return save_path


class MixedDataAugmentation:
    """
    Mixed data augmentation for Brahmi to Devanagari transliteration.
    """
    
    def __init__(self, image_augmenter=None, text_augmenter=None):
        """
        Initialize the mixed data augmentation.
        
        Args:
            image_augmenter (ImageAugmenter, optional): Image augmenter
            text_augmenter (TextAugmenter, optional): Text augmenter
        """
        self.image_augmenter = image_augmenter
        self.text_augmenter = text_augmenter
    
    def generate_synthetic_data(self, brahmi_chars, devanagari_chars, 
                               char_mapping, count=1000, max_length=5):
        """
        Generate synthetic data for transliteration.
        
        Args:
            brahmi_chars (list): List of Brahmi characters
            devanagari_chars (list): List of Devanagari characters
            char_mapping (dict): Mapping from Brahmi to Devanagari characters
            count (int): Number of synthetic pairs to generate
            max_length (int): Maximum length of generated texts
            
        Returns:
            tuple: (synthetic_brahmi_texts, synthetic_devanagari_texts)
        """
        synthetic_brahmi = []
        synthetic_devanagari = []
        
        for _ in range(count):
            # Generate random length
            length = random.randint(1, max_length)
            
            # Generate random Brahmi text
            brahmi_text = ''.join(random.choice(brahmi_chars) for _ in range(length))
            
            # Generate corresponding Devanagari text
            devanagari_text = ''
            
            for char in brahmi_text:
                if char in char_mapping:
                    devanagari_text += char_mapping[char]
                else:
                    # Use a random Devanagari character if mapping not found
                    devanagari_text += random.choice(devanagari_chars)
            
            synthetic_brahmi.append(brahmi_text)
            synthetic_devanagari.append(devanagari_text)
        
        return synthetic_brahmi, synthetic_devanagari
    
    def generate_mixed_augmentation(self, brahmi_texts, devanagari_texts, 
                                   brahmi_images=None, labels=None,
                                   text_count=1, image_count=1):
        """
        Generate mixed augmentation for transliteration.
        
        Args:
            brahmi_texts (list): List of Brahmi texts
            devanagari_texts (list): List of Devanagari texts
            brahmi_images (numpy.ndarray, optional): Brahmi character images
            labels (numpy.ndarray, optional): Labels for Brahmi character images
            text_count (int): Number of text augmentations per pair
            image_count (int): Number of image augmentations per image
            
        Returns:
            dict: Augmented data
        """
        augmented_data = {}
        
        # Augment text data
        if self.text_augmenter is not None:
            augmented_brahmi, augmented_devanagari = self.text_augmenter.augment_dataset(
                brahmi_texts, devanagari_texts, text_count)
            
            augmented_data['augmented_brahmi_texts'] = augmented_brahmi
            augmented_data['augmented_devanagari_texts'] = augmented_devanagari
        
        # Augment image data
        if self.image_augmenter is not None and brahmi_images is not None and labels is not None:
            augmented_images, augmented_labels = self.image_augmenter.augment_batch(
                brahmi_images, image_count)
            
            augmented_data['augmented_brahmi_images'] = augmented_images
            augmented_data['augmented_labels'] = augmented_labels
        
        return augmented_data


if __name__ == "__main__":
    # Example usage
    from src.transliteration_model import BrahmiToDevanagariTransliterator
    from src.confidence_and_correction import ConfidenceScorer
    
    # Create a transliterator
    transliterator = BrahmiToDevanagariTransliterator()
    
    # Create a confidence scorer
    confidence_scorer = ConfidenceScorer(transliterator)
    
    # Create an image augmenter
    image_augmenter = ImageAugmenter(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        shear_range=10,
        brightness_range=(0.8, 1.2),
        noise_range=(0, 0.05),
        blur_range=(0, 1.0)
    )
    
    # Create a text augmenter
    brahmi_chars = ['𑀓', '𑀸', '𑀯', '𑀺']
    devanagari_chars = ['क', 'ा', 'व', 'ी']
    
    text_augmenter = TextAugmenter(
        brahmi_chars=brahmi_chars,
        devanagari_chars=devanagari_chars,
        char_swap_prob=0.05,
        char_drop_prob=0.03,
        char_repeat_prob=0.03
    )
    
    # Create a semi-supervised learner
    semi_supervised_learner = SemiSupervisedLearner(
        transliterator,
        confidence_threshold=0.8
    )
    
    # Create a data augmentation manager
    augmentation_manager = DataAugmentationManager(
        image_augmenter=image_augmenter,
        text_augmenter=text_augmenter,
        semi_supervised_learner=semi_supervised_learner
    )
    
    # Example data
    brahmi_texts = ['𑀓𑀸𑀯𑀺', '𑀅𑀲𑁄𑀓']
    devanagari_texts = ['कावी', 'अशोक']
    
    # Augment transliteration dataset
    augmented_brahmi, augmented_devanagari = augmentation_manager.augment_transliteration_dataset(
        brahmi_texts, devanagari_texts, count=2)
    
    print(f"Original texts: {brahmi_texts} -> {devanagari_texts}")
    print(f"Augmented texts: {augmented_brahmi} -> {augmented_devanagari}")
    
    # Example of semi-supervised learning
    unlabeled_brahmi_texts = ['𑀓𑀸𑀯𑀺𑀢𑀸', '𑀅𑀲𑁄𑀓𑀦']
    
    metrics = augmentation_manager.train_with_semi_supervised_learning(
        brahmi_texts, devanagari_texts, unlabeled_brahmi_texts,
        confidence_scorer=confidence_scorer, epochs=1)
    
    print(f"Semi-supervised learning metrics: {metrics}")
    
    # Example of mixed data augmentation
    mixed_augmenter = MixedDataAugmentation(
        image_augmenter=image_augmenter,
        text_augmenter=text_augmenter
    )
    
    # Generate synthetic data
    char_mapping = dict(zip(brahmi_chars, devanagari_chars))
    
    synthetic_brahmi, synthetic_devanagari = mixed_augmenter.generate_synthetic_data(
        brahmi_chars, devanagari_chars, char_mapping, count=5, max_length=3)
    
    print(f"Synthetic data: {synthetic_brahmi} -> {synthetic_devanagari}")
