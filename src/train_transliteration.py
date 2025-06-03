"""
Script to train the Brahmi to Devanagari transliteration model
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.transliteration_model import BrahmiToDevanagariTransliterator, create_parallel_corpus, load_parallel_corpus

def prepare_synthetic_corpus(output_file):
    """
    Prepare a synthetic parallel corpus of Brahmi and Devanagari character sequences.
    
    Args:
        output_file (str): Path to output file
        
    Returns:
        pd.DataFrame: Parallel corpus
    """
    # Define Brahmi characters and their Devanagari equivalents
    brahmi_chars = [
        '𑀅', '𑀆', '𑀇', '𑀈', '𑀉', '𑀊', '𑀋', '𑀌', '𑀍', '𑀎',
        '𑀏', '𑀐', '𑀑', '𑀒', '𑀓', '𑀔', '𑀕', '𑀖', '𑀗', '𑀘',
        '𑀙', '𑀚', '𑀛', '𑀜', '𑀝', '𑀞', '𑀟', '𑀠', '𑀡', '𑀢',
        '𑀣', '𑀤', '𑀥', '𑀦', '𑀧', '𑀨', '𑀩', '𑀪', '𑀫', '𑀬',
        '𑀭', '𑀮', '𑀯', '𑀰', '𑀱', '𑀲', '𑀳', '𑀴', '𑀵', '𑀶',
        '𑀷', '𑀸', '𑀹', '𑀺', '𑀻', '𑀼', '𑀽', '𑀾', '𑀿', '𑁀',
        '𑁁', '𑁂', '𑁃', '𑁄', '𑁅', '𑁆', '𑁇', '𑁈', '𑁉', '𑁊',
        '𑁋', '𑁌', '𑁍', '𑁎', '𑁏', '𑁐', '𑁑', '𑁒', '𑁓', '𑁔',
        '𑁕', '𑁖', '𑁗', '𑁘', '𑁙', '𑁚', '𑁛', '𑁜', '𑁝', '𑁞',
        '𑁟', '𑁠', '𑁡', '𑁢', '𑁣', '𑁤', '𑁥', '𑁦', '𑁧', '𑁨'
    ]
    
    devanagari_chars = [
        'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ', 'ऋ', 'ॠ', 'ऌ', 'ॡ',
        'ए', 'ऐ', 'ओ', 'औ', 'क', 'ख', 'ग', 'घ', 'ङ', 'च',
        'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण', 'त',
        'थ', 'द', 'ध', 'न', 'प', 'फ', 'ब', 'भ', 'म', 'य',
        'र', 'ल', 'व', 'श', 'ष', 'स', 'ह', 'ळ', 'क्ष', 'ज्ञ',
        '०', 'ा', 'ि', 'ी', 'ु', 'ू', 'ृ', 'ॄ', 'ॢ', 'ॣ',
        'े', 'ै', 'ो', 'ौ', '्', 'ॐ', '।', '॥', '१', '२',
        '३', '४', '५', '६', '७', '८', '९', '०', '॰', '॰',
        '॰', '॰', '॰', '॰', '॰', '॰', '॰', '॰', '॰', '॰',
        '॰', '॰', '॰', '॰', '॰', '॰', '॰', '॰', '॰', '॰'
    ]
    
    # Create synthetic corpus
    corpus = create_parallel_corpus(
        brahmi_chars, 
        devanagari_chars, 
        output_file,
        num_samples=5000,  # Reduced from 10000
        max_length=8       # Reduced from 10
    )
    
    print(f"Created synthetic corpus with {len(corpus)} samples")
    print("Sample entries:")
    print(corpus.head())
    
    return corpus

def prepare_academic_corpus(output_file):
    """
    Prepare an academic parallel corpus of Brahmi and Devanagari character sequences.
    
    Args:
        output_file (str): Path to output file
        
    Returns:
        pd.DataFrame: Parallel corpus
    """
    # Define academic examples
    brahmi_texts = [
        '𑀓𑀸𑀯𑀺',
        '𑀅𑀲𑁄𑀓',
        '𑀥𑀼𑀤𑁆𑀥',
        '𑀤𑀫𑁆𑀫',
        '𑀲𑀁𑀖',
        '𑀧𑁆𑀭𑀸𑀳𑁆𑀫𑀺',
        '𑀤𑁂𑀯𑀦𑀸𑀕𑀭𑀻',
        '𑀮𑀺𑀧𑀺',
        '𑀅𑀓𑁆𑀱𑀭',
        '𑀯𑀭𑁆𑀡𑀫𑀸𑀮𑀸'
    ]
    
    devanagari_texts = [
        'कावी',
        'अशोक',
        'बुद्ध',
        'धम्म',
        'संघ',
        'ब्राह्मि',
        'देवनागरी',
        'लिपि',
        'अक्षर',
        'वर्णमाला'
    ]
    
    # Create DataFrame
    corpus = pd.DataFrame({
        'brahmi': brahmi_texts,
        'devanagari': devanagari_texts
    })
    
    # Save to file
    corpus.to_csv(output_file, index=False)
    
    print(f"Created academic corpus with {len(corpus)} samples")
    print("Sample entries:")
    print(corpus.head())
    
    return corpus

def merge_corpora(synthetic_corpus, academic_corpus, output_file):
    """
    Merge synthetic and academic corpora.
    
    Args:
        synthetic_corpus (pd.DataFrame): Synthetic corpus
        academic_corpus (pd.DataFrame): Academic corpus
        output_file (str): Path to output file
        
    Returns:
        pd.DataFrame: Merged corpus
    """
    # Concatenate corpora
    merged_corpus = pd.concat([synthetic_corpus, academic_corpus])
    
    # Remove duplicates
    merged_corpus = merged_corpus.drop_duplicates()
    
    # Shuffle corpus
    merged_corpus = merged_corpus.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Save to file
    merged_corpus.to_csv(output_file, index=False)
    
    print(f"Created merged corpus with {len(merged_corpus)} samples")
    
    return merged_corpus

def train_transliteration_model(corpus, max_length=50):
    """
    Train the transliteration model.
    
    Args:
        corpus (pd.DataFrame): Parallel corpus
        max_length (int): Maximum sequence length
        
    Returns:
        BrahmiToDevanagariTransliterator: Trained transliterator
    """
    print("Training transliteration model...")
    
    # Extract texts
    brahmi_texts = corpus['brahmi'].tolist()
    devanagari_texts = corpus['devanagari'].tolist()
    
    # Create transliterator
    transliterator = BrahmiToDevanagariTransliterator(
        embedding_dim=128,  # Reduced from 256
        units=256,          # Reduced from 512
        batch_size=16       # Reduced from 64
    )
    
    # Preprocess data
    (encoder_input_train, decoder_input_train, decoder_target_train,
     encoder_input_test, decoder_input_test, decoder_target_test) = transliterator.preprocess_data(
        brahmi_texts, devanagari_texts, max_length=max_length
    )
    
    # Build model
    model = transliterator.build_model()
    
    # Create model directory
    model_dir = os.path.join('/home/ubuntu/brahmi_transliteration_project/models', 'transliteration')
    os.makedirs(model_dir, exist_ok=True)
    
    # Train model
    history = transliterator.train(
        encoder_input_train, decoder_input_train, decoder_target_train,
        encoder_input_test, decoder_input_test, decoder_target_test,
        epochs=20,  # Reduced from 50
        save_dir=model_dir
    )
    
    # Plot training history
    transliterator.plot_training_history(
        save_path=os.path.join(model_dir, 'training_history.png')
    )
    
    # Evaluate model
    metrics = transliterator.evaluate(
        encoder_input_test, decoder_input_test, decoder_target_test
    )
    
    print(f"Evaluation metrics: {metrics}")
    
    # Save model
    transliterator.save_model(model_dir)
    
    # Test transliteration
    test_brahmi_texts = [
        '𑀓𑀸𑀯𑀺',
        '𑀅𑀲𑁄𑀓',
        '𑀥𑀼𑀤𑁆𑀥'
    ]
    
    print("Testing transliteration:")
    for text in test_brahmi_texts:
        transliterated = transliterator.transliterate(text)
        print(f"Brahmi: {text} -> Devanagari: {transliterated}")
    
    return transliterator

def main():
    """
    Main function.
    """
    print("Starting Brahmi to Devanagari transliteration model training pipeline...")
    
    # Create data directory
    data_dir = os.path.join('/home/ubuntu/brahmi_transliteration_project/data', 'parallel_corpus')
    os.makedirs(data_dir, exist_ok=True)
    
    # Prepare synthetic corpus
    print("Preparing synthetic parallel corpus...")
    synthetic_corpus_path = os.path.join(data_dir, 'synthetic_corpus.csv')
    synthetic_corpus = prepare_synthetic_corpus(synthetic_corpus_path)
    
    # Prepare academic corpus
    print("Preparing academic corpus...")
    academic_corpus_path = os.path.join(data_dir, 'academic_corpus.csv')
    academic_corpus = prepare_academic_corpus(academic_corpus_path)
    
    # Merge corpora
    print("Merging corpora...")
    merged_corpus_path = os.path.join(data_dir, 'merged_corpus.csv')
    merged_corpus = merge_corpora(synthetic_corpus, academic_corpus, merged_corpus_path)
    
    # Train transliteration model
    transliterator = train_transliteration_model(merged_corpus)
    
    print("Transliteration model training pipeline completed successfully.")

if __name__ == "__main__":
    main()
