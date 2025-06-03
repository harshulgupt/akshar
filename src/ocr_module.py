"""
Custom CNN-based OCR Module for Brahmi Script Recognition
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2

class BrahmiOCR:
    """
    CNN-based OCR module for Brahmi script recognition with support for
    multiple variants and confidence scoring.
    """
    
    def __init__(self, num_classes=170, input_shape=(224, 224, 3), 
                 model_type='mobilenet', variant_classes=None):
        """
        Initialize the OCR module.
        
        Args:
            num_classes (int): Number of character classes to recognize
            input_shape (tuple): Input image dimensions (height, width, channels)
            model_type (str): Type of CNN architecture ('mobilenet', 'custom')
            variant_classes (int): Number of script variant classes (if None, variant classification is disabled)
        """
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.model_type = model_type
        self.variant_classes = variant_classes
        self.model = None
        self.variant_model = None
        self.history = None
        self.class_indices = None
        
    def build_model(self):
        """
        Build the CNN model architecture based on specified type.
        """
        if self.model_type == 'mobilenet':
            # Use MobileNetV2 as base model (based on research findings)
            base_model = MobileNetV2(
                input_shape=self.input_shape,
                include_top=False,
                weights='imagenet'
            )
            
            # Freeze early layers for transfer learning
            for layer in base_model.layers[:100]:
                layer.trainable = False
                
            # Build the model
            inputs = tf.keras.Input(shape=self.input_shape)
            x = inputs
            
            # Preprocessing layer
            x = layers.Rescaling(1./255)(x)
            
            # Base model
            x = base_model(x, training=False)
            x = layers.GlobalAveragePooling2D()(x)
            x = layers.Dropout(0.2)(x)
            x = layers.Dense(1024, activation='relu')(x)
            x = layers.Dropout(0.5)(x)
            
            # Main classification head
            outputs = layers.Dense(self.num_classes, activation='softmax', name='character_output')(x)
            
            # Multi-task learning for variant classification if specified
            if self.variant_classes:
                variant_output = layers.Dense(self.variant_classes, activation='softmax', name='variant_output')(x)
                self.model = models.Model(inputs=inputs, outputs=[outputs, variant_output])
                self.model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    loss={
                        'character_output': 'categorical_crossentropy',
                        'variant_output': 'categorical_crossentropy'
                    },
                    metrics={
                        'character_output': ['accuracy', 'top_k_categorical_accuracy'],
                        'variant_output': ['accuracy']
                    },
                    loss_weights={
                        'character_output': 1.0,
                        'variant_output': 0.2
                    }
                )
            else:
                self.model = models.Model(inputs=inputs, outputs=outputs)
                self.model.compile(
                    optimizer=optimizers.Adam(learning_rate=0.001),
                    loss='categorical_crossentropy',
                    metrics=['accuracy', 'top_k_categorical_accuracy']
                )
                
        elif self.model_type == 'custom':
            # Custom CNN architecture
            self.model = models.Sequential([
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.input_shape),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.MaxPooling2D((2, 2)),
                layers.Flatten(),
                layers.Dropout(0.5),
                layers.Dense(512, activation='relu'),
                layers.Dense(self.num_classes, activation='softmax')
            ])
            
            self.model.compile(
                optimizer=optimizers.Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy', 'top_k_categorical_accuracy']
            )
        
        return self.model
    
    def preprocess_image(self, image_path):
        """
        Preprocess a single image for inference.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            np.array: Preprocessed image
        """
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.input_shape[0], self.input_shape[1]))
        
        # Apply preprocessing
        img = self._apply_preprocessing(img)
        
        return np.expand_dims(img, axis=0)
    
    def _apply_preprocessing(self, img):
        """
        Apply preprocessing steps to an image.
        
        Args:
            img (np.array): Input image
            
        Returns:
            np.array: Preprocessed image
        """
        # Convert to grayscale if needed
        if self.input_shape[2] == 1 and len(img.shape) == 3 and img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = np.expand_dims(img, axis=-1)
        
        # Normalize
        img = img.astype(np.float32) / 255.0
        
        # Apply thresholding if grayscale
        if self.input_shape[2] == 1:
            _, img = cv2.threshold(img, 0.5, 1.0, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return img
    
    def prepare_data_generators(self, train_dir, validation_dir, batch_size=32, augmentation=True):
        """
        Prepare data generators for training and validation.
        
        Args:
            train_dir (str): Directory containing training data
            validation_dir (str): Directory containing validation data
            batch_size (int): Batch size for training
            augmentation (bool): Whether to apply data augmentation
            
        Returns:
            tuple: (train_generator, validation_generator)
        """
        if augmentation:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                shear_range=0.1,
                zoom_range=0.1,
                horizontal_flip=False,
                fill_mode='nearest'
            )
        else:
            train_datagen = ImageDataGenerator(rescale=1./255)
            
        validation_datagen = ImageDataGenerator(rescale=1./255)
        
        train_generator = train_datagen.flow_from_directory(
            train_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb' if self.input_shape[2] == 3 else 'grayscale'
        )
        
        validation_generator = validation_datagen.flow_from_directory(
            validation_dir,
            target_size=(self.input_shape[0], self.input_shape[1]),
            batch_size=batch_size,
            class_mode='categorical',
            color_mode='rgb' if self.input_shape[2] == 3 else 'grayscale'
        )
        
        # Store class indices for later use
        self.class_indices = train_generator.class_indices
        
        return train_generator, validation_generator
    
    def train(self, train_generator, validation_generator, epochs=50, 
              callbacks=None, save_dir=None):
        """
        Train the OCR model.
        
        Args:
            train_generator: Training data generator
            validation_generator: Validation data generator
            epochs (int): Number of training epochs
            callbacks (list): List of Keras callbacks
            save_dir (str): Directory to save model checkpoints
            
        Returns:
            History object
        """
        if self.model is None:
            self.build_model()
            
        if callbacks is None:
            callbacks = []
            
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            model_checkpoint = ModelCheckpoint(
                os.path.join(save_dir, 'brahmi_ocr_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(model_checkpoint)
            
        # Add early stopping and learning rate reduction
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks.extend([early_stopping, reduce_lr])
        
        # Train the model
        self.history = self.model.fit(
            train_generator,
            validation_data=validation_generator,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Save final model
        if save_dir:
            self.model.save(os.path.join(save_dir, 'brahmi_ocr_final.h5'))
            
        return self.history
    
    def predict(self, image_path):
        """
        Predict character class for a single image with confidence score.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            tuple: (predicted_class, confidence_score, attention_map)
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        # Preprocess the image
        img = self.preprocess_image(image_path)
        
        # Get predictions
        if self.variant_classes:
            char_preds, variant_preds = self.model.predict(img)
            predicted_class_idx = np.argmax(char_preds[0])
            confidence_score = char_preds[0][predicted_class_idx]
            
            variant_idx = np.argmax(variant_preds[0])
            variant_confidence = variant_preds[0][variant_idx]
            
            # Get class name from index
            predicted_class = None
            variant_class = None
            
            if self.class_indices:
                # Invert the dictionary to get index->class mapping
                idx_to_class = {v: k for k, v in self.class_indices.items()}
                if predicted_class_idx in idx_to_class:
                    predicted_class = idx_to_class[predicted_class_idx]
            
            # Generate attention map (feature importance visualization)
            attention_map = self._generate_attention_map(img)
            
            return {
                'character': predicted_class,
                'character_idx': int(predicted_class_idx),
                'character_confidence': float(confidence_score),
                'variant_idx': int(variant_idx),
                'variant_confidence': float(variant_confidence),
                'attention_map': attention_map
            }
        else:
            preds = self.model.predict(img)
            predicted_class_idx = np.argmax(preds[0])
            confidence_score = preds[0][predicted_class_idx]
            
            # Get class name from index
            predicted_class = None
            if self.class_indices:
                # Invert the dictionary to get index->class mapping
                idx_to_class = {v: k for k, v in self.class_indices.items()}
                if predicted_class_idx in idx_to_class:
                    predicted_class = idx_to_class[predicted_class_idx]
            
            # Generate attention map (feature importance visualization)
            attention_map = self._generate_attention_map(img)
            
            return {
                'character': predicted_class,
                'character_idx': int(predicted_class_idx),
                'character_confidence': float(confidence_score),
                'attention_map': attention_map
            }
    
    def _generate_attention_map(self, img):
        """
        Generate an attention map for explainability.
        
        Args:
            img (np.array): Input image
            
        Returns:
            np.array: Attention map highlighting important regions
        """
        # Implement Grad-CAM or similar technique for visualization
        # This is a placeholder - actual implementation would use techniques like Grad-CAM
        # to visualize which parts of the image influenced the prediction
        
        # Placeholder return
        return np.zeros((self.input_shape[0], self.input_shape[1]))
    
    def evaluate(self, test_generator):
        """
        Evaluate the model on test data.
        
        Args:
            test_generator: Test data generator
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        # Get predictions
        y_pred = []
        y_true = []
        
        # Reset the generator
        test_generator.reset()
        
        # Get predictions for all batches
        for i in range(len(test_generator)):
            x, y = test_generator[i]
            pred = self.model.predict(x)
            
            if self.variant_classes:
                pred = pred[0]  # Get character predictions
                
            y_pred.extend(np.argmax(pred, axis=1))
            y_true.extend(np.argmax(y, axis=1))
            
            if i >= len(test_generator) - 1:
                break
                
        # Calculate metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        conf_matrix = confusion_matrix(y_true, y_pred)
        
        # Evaluate using model.evaluate
        if self.variant_classes:
            loss, char_loss, var_loss, char_acc, char_top_k, var_acc = self.model.evaluate(test_generator)
            metrics = {
                'loss': loss,
                'character_loss': char_loss,
                'variant_loss': var_loss,
                'character_accuracy': char_acc,
                'character_top_k_accuracy': char_top_k,
                'variant_accuracy': var_acc,
                'classification_report': report,
                'confusion_matrix': conf_matrix
            }
        else:
            loss, acc, top_k = self.model.evaluate(test_generator)
            metrics = {
                'loss': loss,
                'accuracy': acc,
                'top_k_accuracy': top_k,
                'classification_report': report,
                'confusion_matrix': conf_matrix
            }
            
        return metrics
    
    def save_model(self, save_path):
        """
        Save the trained model.
        
        Args:
            save_path (str): Path to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        self.model.save(save_path)
        
        # Save class indices
        if self.class_indices:
            import json
            with open(os.path.join(os.path.dirname(save_path), 'class_indices.json'), 'w') as f:
                json.dump(self.class_indices, f)
    
    def load_model(self, model_path, class_indices_path=None):
        """
        Load a trained model.
        
        Args:
            model_path (str): Path to the saved model
            class_indices_path (str): Path to the saved class indices
        """
        self.model = tf.keras.models.load_model(model_path)
        
        # Load class indices if provided
        if class_indices_path and os.path.exists(class_indices_path):
            import json
            with open(class_indices_path, 'r') as f:
                self.class_indices = json.load(f)
        
    def plot_training_history(self, save_path=None):
        """
        Plot training history.
        
        Args:
            save_path (str): Path to save the plot
        """
        if self.history is None:
            raise ValueError("No training history available")
            
        # Plot accuracy
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        
        if self.variant_classes:
            plt.plot(self.history.history['character_output_accuracy'])
            plt.plot(self.history.history['val_character_output_accuracy'])
            plt.title('Character Recognition Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
        else:
            plt.plot(self.history.history['accuracy'])
            plt.plot(self.history.history['val_accuracy'])
            plt.title('Model Accuracy')
            plt.ylabel('Accuracy')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)
        
        if self.variant_classes:
            plt.plot(self.history.history['character_output_loss'])
            plt.plot(self.history.history['val_character_output_loss'])
            plt.title('Character Recognition Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
        else:
            plt.plot(self.history.history['loss'])
            plt.plot(self.history.history['val_loss'])
            plt.title('Model Loss')
            plt.ylabel('Loss')
            plt.xlabel('Epoch')
            plt.legend(['Train', 'Validation'], loc='upper left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            
        plt.show()


def create_train_val_split(data_dir, output_dir, val_split=0.2, seed=42):
    """
    Create train/validation split from a directory of class folders.
    
    Args:
        data_dir (str): Directory containing class folders
        output_dir (str): Directory to output the split datasets
        val_split (float): Fraction of data to use for validation
        seed (int): Random seed for reproducibility
    """
    import shutil
    import random
    
    # Set random seed
    random.seed(seed)
    
    # Create output directories
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    
    # Get class folders
    class_folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]
    
    for class_folder in class_folders:
        # Create class folders in train and val
        os.makedirs(os.path.join(train_dir, class_folder), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_folder), exist_ok=True)
        
        # Get all files in the class folder
        files = [f for f in os.listdir(os.path.join(data_dir, class_folder)) 
                if os.path.isfile(os.path.join(data_dir, class_folder, f))]
        
        # Shuffle files
        random.shuffle(files)
        
        # Split files
        split_idx = int(len(files) * (1 - val_split))
        train_files = files[:split_idx]
        val_files = files[split_idx:]
        
        # Copy files to train and val directories
        for f in train_files:
            shutil.copy(
                os.path.join(data_dir, class_folder, f),
                os.path.join(train_dir, class_folder, f)
            )
            
        for f in val_files:
            shutil.copy(
                os.path.join(data_dir, class_folder, f),
                os.path.join(val_dir, class_folder, f)
            )
    
    print(f"Split {len(class_folders)} classes into train and validation sets")
    print(f"Train directory: {train_dir}")
    print(f"Validation directory: {val_dir}")


if __name__ == "__main__":
    # Example usage
    ocr = BrahmiOCR(num_classes=170, input_shape=(224, 224, 3), model_type='mobilenet')
    model = ocr.build_model()
    print(model.summary())
