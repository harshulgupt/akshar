"""
Script to train the Brahmi OCR module using the prepared dataset
"""

import os
import sys
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import json
import shutil

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ocr_module import BrahmiOCR, create_train_val_split

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configuration
DATA_DIR = '/home/ubuntu/brahmi_transliteration_project/data/extracted'
PROCESSED_DIR = '/home/ubuntu/brahmi_transliteration_project/data/processed'
MODELS_DIR = '/home/ubuntu/brahmi_transliteration_project/models'
RESULTS_DIR = '/home/ubuntu/brahmi_transliteration_project/results'

# Create necessary directories
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

def prepare_dataset():
    """
    Prepare the dataset for training by creating train/validation splits
    """
    print("Preparing dataset...")
    
    # Create train/validation split from the training data
    train_data_dir = os.path.join(DATA_DIR, 'train')
    processed_split_dir = os.path.join(PROCESSED_DIR, 'split')
    
    # Create the split
    create_train_val_split(
        data_dir=train_data_dir,
        output_dir=processed_split_dir,
        val_split=0.2,
        seed=42
    )
    
    # Count classes and samples
    train_dir = os.path.join(processed_split_dir, 'train')
    val_dir = os.path.join(processed_split_dir, 'val')
    
    num_classes = len(os.listdir(train_dir))
    num_train_samples = sum([len(os.listdir(os.path.join(train_dir, c))) for c in os.listdir(train_dir)])
    num_val_samples = sum([len(os.listdir(os.path.join(val_dir, c))) for c in os.listdir(val_dir)])
    
    print(f"Dataset prepared with {num_classes} classes")
    print(f"Training samples: {num_train_samples}")
    print(f"Validation samples: {num_val_samples}")
    
    # Save dataset statistics
    stats = {
        'num_classes': num_classes,
        'num_train_samples': num_train_samples,
        'num_val_samples': num_val_samples,
        'class_list': os.listdir(train_dir)
    }
    
    with open(os.path.join(PROCESSED_DIR, 'dataset_stats.json'), 'w') as f:
        json.dump(stats, f, indent=4)
    
    return train_dir, val_dir, num_classes

def train_ocr_model(train_dir, val_dir, num_classes):
    """
    Train the OCR model using the prepared dataset
    """
    print("Training OCR model...")
    
    # Initialize the OCR model
    ocr = BrahmiOCR(
        num_classes=num_classes,
        input_shape=(224, 224, 3),
        model_type='mobilenet',
        variant_classes=None  # No variant classification for initial model
    )
    
    # Build the model
    model = ocr.build_model()
    print(model.summary())
    
    # Prepare data generators
    train_generator, validation_generator = ocr.prepare_data_generators(
        train_dir=train_dir,
        validation_dir=val_dir,
        batch_size=32,
        augmentation=True
    )
    
    # Train the model
    history = ocr.train(
        train_generator=train_generator,
        validation_generator=validation_generator,
        epochs=50,
        save_dir=MODELS_DIR
    )
    
    # Plot and save training history
    ocr.plot_training_history(save_path=os.path.join(RESULTS_DIR, 'training_history.png'))
    
    # Save the class indices
    with open(os.path.join(MODELS_DIR, 'class_indices.json'), 'w') as f:
        json.dump(ocr.class_indices, f, indent=4)
    
    return ocr, train_generator, validation_generator

def evaluate_model(ocr, validation_generator):
    """
    Evaluate the trained model on the validation set
    """
    print("Evaluating model...")
    
    # Evaluate the model
    metrics = ocr.evaluate(validation_generator)
    
    # Save evaluation metrics
    with open(os.path.join(RESULTS_DIR, 'evaluation_metrics.json'), 'w') as f:
        # Convert numpy types to Python native types for JSON serialization
        metrics_json = {}
        for k, v in metrics.items():
            if k not in ['classification_report', 'confusion_matrix']:
                metrics_json[k] = float(v)
            elif k == 'classification_report':
                metrics_json[k] = v
            else:
                # Skip confusion matrix for JSON (can be large)
                pass
        
        json.dump(metrics_json, f, indent=4)
    
    # Save confusion matrix as numpy array
    np.save(os.path.join(RESULTS_DIR, 'confusion_matrix.npy'), metrics['confusion_matrix'])
    
    print(f"Validation accuracy: {metrics.get('accuracy', metrics.get('character_accuracy', 0)):.4f}")
    
    return metrics

def test_inference(ocr, test_dir):
    """
    Test inference on a few sample images
    """
    print("Testing inference...")
    
    # Get a few test images
    test_images = []
    for class_dir in os.listdir(test_dir):
        class_path = os.path.join(test_dir, class_dir)
        if os.path.isdir(class_path):
            images = os.listdir(class_path)
            if images:
                test_images.append(os.path.join(class_path, images[0]))
                if len(test_images) >= 5:
                    break
    
    # Run inference on test images
    results = []
    for img_path in test_images:
        result = ocr.predict(img_path)
        results.append({
            'image_path': img_path,
            'predicted_class': result['character'],
            'confidence': result['character_confidence']
        })
        print(f"Image: {os.path.basename(img_path)}")
        print(f"Predicted class: {result['character']}")
        print(f"Confidence: {result['character_confidence']:.4f}")
        print("---")
    
    # Save inference results
    with open(os.path.join(RESULTS_DIR, 'inference_samples.json'), 'w') as f:
        json.dump(results, f, indent=4)

def main():
    """
    Main function to run the training pipeline
    """
    print("Starting Brahmi OCR training pipeline...")
    
    # Prepare dataset
    train_dir, val_dir, num_classes = prepare_dataset()
    
    # Train model
    ocr, train_generator, validation_generator = train_ocr_model(train_dir, val_dir, num_classes)
    
    # Evaluate model
    metrics = evaluate_model(ocr, validation_generator)
    
    # Test inference
    test_dir = os.path.join(DATA_DIR, 'test')
    test_inference(ocr, test_dir)
    
    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()
