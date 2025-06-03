"""
Sequence-to-Sequence Transliteration Model for Brahmi to Devanagari conversion
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import json
import pandas as pd
from sklearn.model_selection import train_test_split
import unicodedata

class BrahmiToDevanagariTransliterator:
    """
    Sequence-to-sequence transliteration model for converting Brahmi characters
    to Devanagari Unicode text, with attention mechanisms for contextual awareness.
    """
    
    def __init__(self, embedding_dim=128, units=256, batch_size=16, 
                 brahmi_vocab_size=None, devanagari_vocab_size=None):
        """
        Initialize the transliteration model.
        
        Args:
            embedding_dim (int): Dimension of the embedding vectors
            units (int): Number of units in the LSTM/GRU layers
            batch_size (int): Batch size for training
            brahmi_vocab_size (int): Size of the Brahmi vocabulary
            devanagari_vocab_size (int): Size of the Devanagari vocabulary
        """
        self.embedding_dim = embedding_dim
        self.units = units
        self.batch_size = batch_size
        self.brahmi_vocab_size = brahmi_vocab_size
        self.devanagari_vocab_size = devanagari_vocab_size
        
        # Tokenizers
        self.brahmi_tokenizer = None
        self.devanagari_tokenizer = None
        
        # Special tokens
        self.START_TOKEN = '<start>'
        self.END_TOKEN = '<end>'
        self.PAD_TOKEN = '<pad>'
        self.UNK_TOKEN = '<unk>'
        
        # Models
        self.encoder = None
        self.decoder = None
        self.model = None
        self.history = None
        
    def create_tokenizers(self, brahmi_texts, devanagari_texts):
        """
        Create tokenizers for Brahmi and Devanagari texts.
        
        Args:
            brahmi_texts (list): List of Brahmi text sequences
            devanagari_texts (list): List of Devanagari text sequences
            
        Returns:
            tuple: (brahmi_tokenizer, devanagari_tokenizer)
        """
        # Create tokenizers
        self.brahmi_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=self.UNK_TOKEN)
        self.devanagari_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=self.UNK_TOKEN)
        
        # Fit tokenizers on texts
        self.brahmi_tokenizer.fit_on_texts(brahmi_texts)
        self.devanagari_tokenizer.fit_on_texts([self.START_TOKEN + ' ' + text + ' ' + self.END_TOKEN for text in devanagari_texts])
        
        # Update vocabulary sizes
        self.brahmi_vocab_size = len(self.brahmi_tokenizer.word_index) + 1  # +1 for padding
        self.devanagari_vocab_size = len(self.devanagari_tokenizer.word_index) + 1  # +1 for padding
        
        # Save word indices
        self.brahmi_word_index = self.brahmi_tokenizer.word_index
        self.devanagari_word_index = self.devanagari_tokenizer.word_index
        
        # Create reverse mappings
        self.brahmi_index_word = {v: k for k, v in self.brahmi_word_index.items()}
        self.devanagari_index_word = {v: k for k, v in self.devanagari_word_index.items()}
        
        return self.brahmi_tokenizer, self.devanagari_tokenizer
    
    def save_tokenizers(self, save_dir):
        """
        Save tokenizers to files.
        
        Args:
            save_dir (str): Directory to save tokenizers
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Save Brahmi tokenizer
        brahmi_tokenizer_json = {
            'word_index': self.brahmi_tokenizer.word_index,
            'index_word': self.brahmi_index_word,
            'vocab_size': self.brahmi_vocab_size
        }
        with open(os.path.join(save_dir, 'brahmi_tokenizer.json'), 'w', encoding='utf-8') as f:
            json.dump(brahmi_tokenizer_json, f, ensure_ascii=False, indent=4)
        
        # Save Devanagari tokenizer
        devanagari_tokenizer_json = {
            'word_index': self.devanagari_tokenizer.word_index,
            'index_word': self.devanagari_index_word,
            'vocab_size': self.devanagari_vocab_size
        }
        with open(os.path.join(save_dir, 'devanagari_tokenizer.json'), 'w', encoding='utf-8') as f:
            json.dump(devanagari_tokenizer_json, f, ensure_ascii=False, indent=4)
    
    def load_tokenizers(self, load_dir):
        """
        Load tokenizers from files.
        
        Args:
            load_dir (str): Directory to load tokenizers from
        """
        # Load Brahmi tokenizer
        with open(os.path.join(load_dir, 'brahmi_tokenizer.json'), 'r', encoding='utf-8') as f:
            brahmi_tokenizer_json = json.load(f)
        
        self.brahmi_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=self.UNK_TOKEN)
        self.brahmi_tokenizer.word_index = brahmi_tokenizer_json['word_index']
        self.brahmi_index_word = brahmi_tokenizer_json['index_word']
        self.brahmi_vocab_size = brahmi_tokenizer_json['vocab_size']
        
        # Load Devanagari tokenizer
        with open(os.path.join(load_dir, 'devanagari_tokenizer.json'), 'r', encoding='utf-8') as f:
            devanagari_tokenizer_json = json.load(f)
        
        self.devanagari_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='', oov_token=self.UNK_TOKEN)
        self.devanagari_tokenizer.word_index = devanagari_tokenizer_json['word_index']
        self.devanagari_index_word = devanagari_tokenizer_json['index_word']
        self.devanagari_vocab_size = devanagari_tokenizer_json['vocab_size']
    
    def preprocess_data(self, brahmi_texts, devanagari_texts, max_length=50, test_size=0.2, random_state=42):
        """
        Preprocess data for training.
        
        Args:
            brahmi_texts (list): List of Brahmi text sequences
            devanagari_texts (list): List of Devanagari text sequences
            max_length (int): Maximum sequence length
            test_size (float): Proportion of data to use for testing
            random_state (int): Random seed for reproducibility
            
        Returns:
            tuple: (encoder_input_train, decoder_input_train, decoder_target_train,
                   encoder_input_test, decoder_input_test, decoder_target_test)
        """
        # Create tokenizers if not already created
        if self.brahmi_tokenizer is None or self.devanagari_tokenizer is None:
            self.create_tokenizers(brahmi_texts, devanagari_texts)
        
        # Convert texts to sequences
        brahmi_sequences = self.brahmi_tokenizer.texts_to_sequences(brahmi_texts)
        
        # Add start and end tokens to Devanagari texts
        devanagari_texts_with_tokens = [self.START_TOKEN + ' ' + text + ' ' + self.END_TOKEN for text in devanagari_texts]
        devanagari_sequences = self.devanagari_tokenizer.texts_to_sequences(devanagari_texts_with_tokens)
        
        # Pad sequences
        brahmi_padded = tf.keras.preprocessing.sequence.pad_sequences(
            brahmi_sequences, maxlen=max_length, padding='post')
        devanagari_padded = tf.keras.preprocessing.sequence.pad_sequences(
            devanagari_sequences, maxlen=max_length, padding='post')
        
        # Create decoder input and target sequences
        decoder_input = devanagari_padded[:, :-1]  # Remove the last token (end)
        decoder_target = devanagari_padded[:, 1:]  # Remove the first token (start)
        
        # Split data into train and test sets
        (encoder_input_train, encoder_input_test,
         decoder_input_train, decoder_input_test,
         decoder_target_train, decoder_target_test) = train_test_split(
            brahmi_padded, decoder_input, decoder_target,
            test_size=test_size, random_state=random_state)
        
        return (encoder_input_train, decoder_input_train, decoder_target_train,
                encoder_input_test, decoder_input_test, decoder_target_test)
    
    def build_encoder(self):
        """
        Build the encoder model.
        
        Returns:
            tf.keras.Model: Encoder model
        """
        # Encoder
        encoder_inputs = tf.keras.Input(shape=(None,), name='encoder_inputs')
        encoder_embedding = layers.Embedding(
            self.brahmi_vocab_size, self.embedding_dim, name='encoder_embedding')(encoder_inputs)
        
        # Bidirectional LSTM for better context capture
        encoder_lstm = layers.Bidirectional(
            layers.LSTM(self.units, return_sequences=True, return_state=True, name='encoder_lstm'),
            name='bidirectional_encoder')
        
        encoder_outputs, forward_h, forward_c, backward_h, backward_c = encoder_lstm(encoder_embedding)
        
        # Concatenate the forward and backward states
        state_h = layers.Concatenate()([forward_h, backward_h])
        state_c = layers.Concatenate()([forward_c, backward_c])
        
        encoder_states = [state_h, state_c]
        
        self.encoder = tf.keras.Model(encoder_inputs, [encoder_outputs, *encoder_states], name='encoder')
        return self.encoder
    
    def build_decoder(self):
        """
        Build the decoder model with attention.
        
        Returns:
            tf.keras.Model: Decoder model
        """
        # Decoder inputs
        decoder_inputs = tf.keras.Input(shape=(None,), name='decoder_inputs')
        encoder_outputs = tf.keras.Input(shape=(None, self.units * 2), name='encoder_outputs')
        encoder_h = tf.keras.Input(shape=(self.units * 2,), name='encoder_h')
        encoder_c = tf.keras.Input(shape=(self.units * 2,), name='encoder_c')
        
        # Embedding layer
        decoder_embedding = layers.Embedding(
            self.devanagari_vocab_size, self.embedding_dim, name='decoder_embedding')(decoder_inputs)
        
        # LSTM layer
        decoder_lstm = layers.LSTM(
            self.units * 2, return_sequences=True, return_state=True, name='decoder_lstm')
        
        # Initial call to LSTM with encoder states
        decoder_outputs, decoder_h, decoder_c = decoder_lstm(
            decoder_embedding, initial_state=[encoder_h, encoder_c])
        
        # Use Keras built-in Attention layer
        attention = layers.Attention(name='attention_layer')
        context_vector = attention([decoder_outputs, encoder_outputs])
        
        # Concatenate decoder outputs with attention context
        decoder_combined = layers.Concatenate(axis=-1)([decoder_outputs, context_vector])
        
        # Output layer
        decoder_dense = layers.Dense(self.devanagari_vocab_size, activation='softmax', name='decoder_dense')
        decoder_outputs = decoder_dense(decoder_combined)
        
        # Create the decoder model
        self.decoder = tf.keras.Model(
            [decoder_inputs, encoder_outputs, encoder_h, encoder_c],
            decoder_outputs,
            name='decoder')
        
        return self.decoder
    
    def build_model(self):
        """
        Build the full sequence-to-sequence model with attention.
        
        Returns:
            tf.keras.Model: Full model
        """
        # Build encoder and decoder if not already built
        if self.encoder is None:
            self.build_encoder()
        
        if self.decoder is None:
            self.build_decoder()
        
        # Define inputs
        encoder_inputs = tf.keras.Input(shape=(None,), name='encoder_inputs')
        decoder_inputs = tf.keras.Input(shape=(None,), name='decoder_inputs')
        
        # Get encoder outputs and states
        encoder_outputs, encoder_h, encoder_c = self.encoder(encoder_inputs)
        
        # Get decoder outputs
        decoder_outputs = self.decoder(
            [decoder_inputs, encoder_outputs, encoder_h, encoder_c])
        
        # Create the full model
        self.model = tf.keras.Model(
            [encoder_inputs, decoder_inputs],
            decoder_outputs,
            name='brahmi_to_devanagari_transliterator')
        
        # Compile the model
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])
        
        return self.model
    
    def train(self, encoder_input_train, decoder_input_train, decoder_target_train,
              encoder_input_val, decoder_input_val, decoder_target_val,
              epochs=30, callbacks=None, save_dir=None):
        """
        Train the transliteration model.
        
        Args:
            encoder_input_train: Training data for encoder input
            decoder_input_train: Training data for decoder input
            decoder_target_train: Training data for decoder target
            encoder_input_val: Validation data for encoder input
            decoder_input_val: Validation data for decoder input
            decoder_target_val: Validation data for decoder target
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
                os.path.join(save_dir, 'transliteration_model_best.h5'),
                monitor='val_accuracy',
                save_best_only=True,
                mode='max',
                verbose=1
            )
            callbacks.append(model_checkpoint)
            
        # Add early stopping and learning rate reduction
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6,
            verbose=1
        )
        
        callbacks.extend([early_stopping, reduce_lr])
        
        # Train the model
        self.history = self.model.fit(
            [encoder_input_train, decoder_input_train],
            decoder_target_train,
            validation_data=(
                [encoder_input_val, decoder_input_val],
                decoder_target_val
            ),
            epochs=epochs,
            batch_size=self.batch_size,
            callbacks=callbacks
        )
        
        # Save final model
        if save_dir:
            self.model.save(os.path.join(save_dir, 'transliteration_model_final.h5'))
            
        return self.history
    
    def transliterate(self, brahmi_text, max_length=50):
        """
        Transliterate Brahmi text to Devanagari.
        
        Args:
            brahmi_text (str): Brahmi text to transliterate
            max_length (int): Maximum sequence length
            
        Returns:
            str: Transliterated text
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        # Tokenize input text
        brahmi_sequence = self.brahmi_tokenizer.texts_to_sequences([brahmi_text])[0]
        brahmi_padded = tf.keras.preprocessing.sequence.pad_sequences(
            [brahmi_sequence], maxlen=max_length, padding='post')
        
        # Initialize decoder input with start token
        start_token_id = self.devanagari_tokenizer.word_index[self.START_TOKEN]
        decoder_input = tf.expand_dims([start_token_id], 0)
        
        # Initialize result
        result = []
        
        # Get encoder outputs and states
        encoder_outputs, encoder_h, encoder_c = self.encoder(brahmi_padded)
        
        # Inference loop (only used during inference, not during model building)
        for t in range(max_length):
            # Get decoder output
            decoder_output = self.decoder(
                [decoder_input, encoder_outputs, encoder_h, encoder_c])
            
            # Get predicted token
            predicted_id = tf.argmax(decoder_output[0, -1, :]).numpy()
            
            # Append to result
            if predicted_id in self.devanagari_index_word:
                predicted_word = self.devanagari_index_word[predicted_id]
                result.append(predicted_word)
            
            # Break if end token is predicted
            if predicted_id == self.devanagari_tokenizer.word_index.get(self.END_TOKEN, 0):
                break
                
            # Update decoder input
            decoder_input = tf.concat([decoder_input, tf.expand_dims([predicted_id], 0)], axis=-1)
        
        # Remove start and end tokens
        result = [token for token in result if token not in [self.START_TOKEN, self.END_TOKEN, self.PAD_TOKEN]]
        
        # Join result
        transliterated_text = ''.join(result)
        
        return transliterated_text
    
    def save_model(self, save_dir):
        """
        Save the trained model.
        
        Args:
            save_dir (str): Directory to save the model
        """
        if self.model is None:
            raise ValueError("No model to save")
            
        os.makedirs(save_dir, exist_ok=True)
        
        # Save encoder
        self.encoder.save(os.path.join(save_dir, 'encoder.h5'))
        
        # Save decoder
        self.decoder.save(os.path.join(save_dir, 'decoder.h5'))
        
        # Save full model
        self.model.save(os.path.join(save_dir, 'full_model.h5'))
        
        # Save tokenizers
        self.save_tokenizers(save_dir)
        
        # Save model parameters
        model_params = {
            'embedding_dim': self.embedding_dim,
            'units': self.units,
            'batch_size': self.batch_size,
            'brahmi_vocab_size': self.brahmi_vocab_size,
            'devanagari_vocab_size': self.devanagari_vocab_size
        }
        
        with open(os.path.join(save_dir, 'model_params.json'), 'w') as f:
            json.dump(model_params, f)
    
    def load_model(self, load_dir):
        """
        Load a trained model.
        
        Args:
            load_dir (str): Directory to load the model from
        """
        # Load model parameters
        with open(os.path.join(load_dir, 'model_params.json'), 'r') as f:
            model_params = json.load(f)
            
        self.embedding_dim = model_params['embedding_dim']
        self.units = model_params['units']
        self.batch_size = model_params['batch_size']
        self.brahmi_vocab_size = model_params['brahmi_vocab_size']
        self.devanagari_vocab_size = model_params['devanagari_vocab_size']
        
        # Load tokenizers
        self.load_tokenizers(load_dir)
        
        # Load encoder
        self.encoder = tf.keras.models.load_model(os.path.join(load_dir, 'encoder.h5'))
        
        # Load decoder
        self.decoder = tf.keras.models.load_model(os.path.join(load_dir, 'decoder.h5'))
        
        # Load full model
        self.model = tf.keras.models.load_model(os.path.join(load_dir, 'full_model.h5'))
    
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
        plt.plot(self.history.history['accuracy'])
        plt.plot(self.history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'], loc='upper left')
        
        # Plot loss
        plt.subplot(1, 2, 2)
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
    
    def evaluate(self, encoder_input_test, decoder_input_test, decoder_target_test):
        """
        Evaluate the model on test data.
        
        Args:
            encoder_input_test: Test data for encoder input
            decoder_input_test: Test data for decoder input
            decoder_target_test: Test data for decoder target
            
        Returns:
            dict: Evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded")
            
        # Evaluate the model
        results = self.model.evaluate(
            [encoder_input_test, decoder_input_test],
            decoder_target_test,
            batch_size=self.batch_size,
            verbose=1
        )
        
        # Create metrics dictionary
        metrics = {
            'loss': results[0],
            'accuracy': results[1]
        }
        
        return metrics


def create_parallel_corpus(brahmi_chars, devanagari_chars, output_file, num_samples=5000, max_length=8):
    """
    Create a synthetic parallel corpus of Brahmi and Devanagari character sequences.
    
    Args:
        brahmi_chars (list): List of Brahmi characters
        devanagari_chars (list): List of corresponding Devanagari characters
        output_file (str): Path to output file
        num_samples (int): Number of samples to generate
        max_length (int): Maximum sequence length
    """
    # Create character mapping
    char_map = dict(zip(brahmi_chars, devanagari_chars))
    
    # Generate random sequences
    brahmi_sequences = []
    devanagari_sequences = []
    
    for _ in range(num_samples):
        # Generate random length
        length = np.random.randint(1, max_length + 1)
        
        # Generate random sequence
        brahmi_seq = np.random.choice(brahmi_chars, size=length)
        
        # Transliterate to Devanagari
        devanagari_seq = [char_map.get(char, char) for char in brahmi_seq]
        
        # Join sequences
        brahmi_sequences.append(''.join(brahmi_seq))
        devanagari_sequences.append(''.join(devanagari_seq))
    
    # Create DataFrame
    df = pd.DataFrame({
        'brahmi': brahmi_sequences,
        'devanagari': devanagari_sequences
    })
    
    # Save to file
    df.to_csv(output_file, index=False)
    
    return df


def load_parallel_corpus(file_path):
    """
    Load a parallel corpus from a file.
    
    Args:
        file_path (str): Path to the corpus file
        
    Returns:
        tuple: (brahmi_texts, devanagari_texts)
    """
    # Load corpus
    df = pd.read_csv(file_path)
    
    # Extract texts
    brahmi_texts = df['brahmi'].tolist()
    devanagari_texts = df['devanagari'].tolist()
    
    return brahmi_texts, devanagari_texts


def normalize_unicode(text):
    """
    Normalize Unicode text.
    
    Args:
        text (str): Text to normalize
        
    Returns:
        str: Normalized text
    """
    return unicodedata.normalize('NFC', text)


if __name__ == "__main__":
    # Example usage
    transliterator = BrahmiToDevanagariTransliterator(
        embedding_dim=128,
        units=256,
        batch_size=16
    )
    
    # Print model summary
    model = transliterator.build_model()
    print(model.summary())
