"""
Confidence scoring and user correction interface for Brahmi to Devanagari transliteration
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import json
import pandas as pd
import tkinter as tk
from tkinter import ttk, scrolledtext
from PIL import Image, ImageTk
import cv2
import unicodedata
from src.transliteration_model import BrahmiToDevanagariTransliterator
from src.variant_support import MultiVariantTransliterator

class ConfidenceScorer:
    """
    Confidence scoring for Brahmi to Devanagari transliteration.
    """
    
    def __init__(self, transliterator):
        """
        Initialize the confidence scorer.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
        """
        self.transliterator = transliterator
        
    def calculate_character_confidence(self, brahmi_text, devanagari_text, attention_weights=None):
        """
        Calculate confidence scores for each character in the transliteration.
        
        Args:
            brahmi_text (str): Original Brahmi text
            devanagari_text (str): Transliterated Devanagari text
            attention_weights (numpy.ndarray, optional): Attention weights from the model
            
        Returns:
            list: List of confidence scores for each character
        """
        # If attention weights are provided, use them to calculate confidence
        if attention_weights is not None:
            # Normalize attention weights to get confidence scores
            confidence_scores = []
            for i in range(len(devanagari_text)):
                if i < attention_weights.shape[0]:
                    # Use maximum attention weight as confidence score
                    confidence = np.max(attention_weights[i])
                    confidence_scores.append(float(confidence))
                else:
                    # Default confidence for characters beyond attention matrix
                    confidence_scores.append(0.5)
            
            return confidence_scores
        
        # If no attention weights, use a simple heuristic based on character frequency
        else:
            # For demonstration, assign random confidence scores
            # In a real implementation, this would use more sophisticated methods
            confidence_scores = []
            for _ in range(len(devanagari_text)):
                # Generate a random confidence score between 0.7 and 1.0
                confidence = 0.7 + 0.3 * np.random.random()
                confidence_scores.append(float(confidence))
            
            return confidence_scores
    
    def highlight_low_confidence_characters(self, devanagari_text, confidence_scores, threshold=0.8):
        """
        Identify characters with confidence scores below the threshold.
        
        Args:
            devanagari_text (str): Transliterated Devanagari text
            confidence_scores (list): List of confidence scores for each character
            threshold (float): Confidence threshold (0.0 to 1.0)
            
        Returns:
            list: List of indices of low-confidence characters
        """
        low_confidence_indices = []
        
        for i, confidence in enumerate(confidence_scores):
            if confidence < threshold:
                low_confidence_indices.append(i)
                
        return low_confidence_indices
    
    def format_with_confidence(self, devanagari_text, confidence_scores, threshold=0.8):
        """
        Format the transliterated text with confidence indicators.
        
        Args:
            devanagari_text (str): Transliterated Devanagari text
            confidence_scores (list): List of confidence scores for each character
            threshold (float): Confidence threshold (0.0 to 1.0)
            
        Returns:
            dict: Formatted text with confidence indicators
        """
        formatted_result = {
            'text': devanagari_text,
            'confidence_scores': confidence_scores,
            'low_confidence_indices': self.highlight_low_confidence_characters(
                devanagari_text, confidence_scores, threshold),
            'average_confidence': sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
        }
        
        return formatted_result


class UserCorrectionInterface:
    """
    User interface for correcting Brahmi to Devanagari transliterations.
    """
    
    def __init__(self, transliterator, confidence_scorer, feedback_handler=None):
        """
        Initialize the user correction interface.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
            confidence_scorer (ConfidenceScorer): Confidence scorer
            feedback_handler: Handler for user feedback (for self-improvement)
        """
        self.transliterator = transliterator
        self.confidence_scorer = confidence_scorer
        self.feedback_handler = feedback_handler
        self.correction_history = []
        
    def create_gui(self):
        """
        Create a graphical user interface for transliteration correction.
        
        Returns:
            tk.Tk: Root window of the GUI
        """
        # Create root window
        root = tk.Tk()
        root.title("Brahmi to Devanagari Transliteration Correction")
        root.geometry("800x600")
        
        # Create main frame
        main_frame = ttk.Frame(root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Create input frame
        input_frame = ttk.LabelFrame(main_frame, text="Input", padding="10")
        input_frame.pack(fill=tk.X, pady=5)
        
        # Create input text area
        ttk.Label(input_frame, text="Brahmi Text:").pack(anchor=tk.W)
        self.input_text = scrolledtext.ScrolledText(input_frame, height=5)
        self.input_text.pack(fill=tk.X)
        
        # Create transliteration button
        ttk.Button(input_frame, text="Transliterate", 
                  command=self.on_transliterate).pack(pady=5)
        
        # Create output frame
        output_frame = ttk.LabelFrame(main_frame, text="Transliteration", padding="10")
        output_frame.pack(fill=tk.X, pady=5)
        
        # Create output text area
        ttk.Label(output_frame, text="Devanagari Text:").pack(anchor=tk.W)
        self.output_text = scrolledtext.ScrolledText(output_frame, height=5)
        self.output_text.pack(fill=tk.X)
        
        # Create confidence frame
        confidence_frame = ttk.LabelFrame(main_frame, text="Confidence", padding="10")
        confidence_frame.pack(fill=tk.X, pady=5)
        
        # Create confidence indicator
        ttk.Label(confidence_frame, text="Average Confidence:").pack(side=tk.LEFT)
        self.confidence_label = ttk.Label(confidence_frame, text="N/A")
        self.confidence_label.pack(side=tk.LEFT, padx=5)
        
        # Create confidence threshold slider
        ttk.Label(confidence_frame, text="Threshold:").pack(side=tk.LEFT, padx=(20, 0))
        self.threshold_var = tk.DoubleVar(value=0.8)
        threshold_slider = ttk.Scale(confidence_frame, from_=0.5, to=1.0, 
                                    variable=self.threshold_var, orient=tk.HORIZONTAL, length=200)
        threshold_slider.pack(side=tk.LEFT, padx=5)
        threshold_slider.bind("<ButtonRelease-1>", self.on_threshold_change)
        
        # Create correction frame
        correction_frame = ttk.LabelFrame(main_frame, text="Correction", padding="10")
        correction_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # Create correction text area
        ttk.Label(correction_frame, text="Corrected Text:").pack(anchor=tk.W)
        self.correction_text = scrolledtext.ScrolledText(correction_frame, height=5)
        self.correction_text.pack(fill=tk.X)
        
        # Create submit button
        ttk.Button(correction_frame, text="Submit Correction", 
                  command=self.on_submit_correction).pack(pady=5)
        
        # Create status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        return root
    
    def on_transliterate(self):
        """
        Handle transliteration button click.
        """
        # Get input text
        brahmi_text = self.input_text.get("1.0", tk.END).strip()
        
        if not brahmi_text:
            self.status_var.set("Error: No input text")
            return
        
        # Transliterate text
        try:
            # Check if using multi-variant transliterator
            if isinstance(self.transliterator, MultiVariantTransliterator):
                result = self.transliterator.transliterate(brahmi_text)
                devanagari_text = result['transliterated_text']
                variant = result.get('detected_variant', 'Unknown')
                self.status_var.set(f"Detected variant: {variant}")
            else:
                devanagari_text = self.transliterator.transliterate(brahmi_text)
            
            # Calculate confidence scores
            confidence_scores = self.confidence_scorer.calculate_character_confidence(
                brahmi_text, devanagari_text)
            
            # Format with confidence
            formatted_result = self.confidence_scorer.format_with_confidence(
                devanagari_text, confidence_scores, self.threshold_var.get())
            
            # Update output text
            self.output_text.delete("1.0", tk.END)
            self.output_text.insert("1.0", formatted_result['text'])
            
            # Highlight low confidence characters
            for idx in formatted_result['low_confidence_indices']:
                if idx < len(devanagari_text):
                    start_pos = f"1.{idx}"
                    end_pos = f"1.{idx + 1}"
                    self.output_text.tag_add("low_confidence", start_pos, end_pos)
                    self.output_text.tag_configure("low_confidence", background="yellow")
            
            # Update confidence label
            avg_confidence = formatted_result['average_confidence']
            self.confidence_label.config(text=f"{avg_confidence:.2f}")
            
            # Copy to correction text
            self.correction_text.delete("1.0", tk.END)
            self.correction_text.insert("1.0", formatted_result['text'])
            
            self.status_var.set("Transliteration complete")
            
        except Exception as e:
            self.status_var.set(f"Error: {str(e)}")
    
    def on_threshold_change(self, event):
        """
        Handle confidence threshold change.
        """
        # Re-highlight low confidence characters
        devanagari_text = self.output_text.get("1.0", tk.END).strip()
        
        if not devanagari_text:
            return
        
        # Get confidence scores (from previous calculation)
        brahmi_text = self.input_text.get("1.0", tk.END).strip()
        confidence_scores = self.confidence_scorer.calculate_character_confidence(
            brahmi_text, devanagari_text)
        
        # Format with new threshold
        formatted_result = self.confidence_scorer.format_with_confidence(
            devanagari_text, confidence_scores, self.threshold_var.get())
        
        # Clear previous highlighting
        self.output_text.tag_remove("low_confidence", "1.0", tk.END)
        
        # Highlight low confidence characters
        for idx in formatted_result['low_confidence_indices']:
            if idx < len(devanagari_text):
                start_pos = f"1.{idx}"
                end_pos = f"1.{idx + 1}"
                self.output_text.tag_add("low_confidence", start_pos, end_pos)
                self.output_text.tag_configure("low_confidence", background="yellow")
    
    def on_submit_correction(self):
        """
        Handle correction submission.
        """
        # Get original and corrected texts
        brahmi_text = self.input_text.get("1.0", tk.END).strip()
        original_text = self.output_text.get("1.0", tk.END).strip()
        corrected_text = self.correction_text.get("1.0", tk.END).strip()
        
        if not brahmi_text or not original_text or not corrected_text:
            self.status_var.set("Error: Missing text")
            return
        
        # Record correction
        correction = {
            'brahmi_text': brahmi_text,
            'original_text': original_text,
            'corrected_text': corrected_text,
            'timestamp': pd.Timestamp.now()
        }
        
        self.correction_history.append(correction)
        
        # Send to feedback handler if available
        if self.feedback_handler:
            self.feedback_handler.process_correction(correction)
            
        self.status_var.set("Correction submitted")
    
    def run(self):
        """
        Run the user interface.
        """
        root = self.create_gui()
        root.mainloop()
    
    def save_correction_history(self, file_path):
        """
        Save correction history to a file.
        
        Args:
            file_path (str): Path to save the correction history
        """
        # Convert to DataFrame
        df = pd.DataFrame(self.correction_history)
        
        # Save to file
        df.to_csv(file_path, index=False)
    
    def load_correction_history(self, file_path):
        """
        Load correction history from a file.
        
        Args:
            file_path (str): Path to load the correction history from
        """
        # Load from file
        df = pd.read_csv(file_path)
        
        # Convert to list of dictionaries
        self.correction_history = df.to_dict('records')


class CommandLineInterface:
    """
    Command-line interface for Brahmi to Devanagari transliteration.
    """
    
    def __init__(self, transliterator, confidence_scorer, feedback_handler=None):
        """
        Initialize the command-line interface.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
            confidence_scorer (ConfidenceScorer): Confidence scorer
            feedback_handler: Handler for user feedback (for self-improvement)
        """
        self.transliterator = transliterator
        self.confidence_scorer = confidence_scorer
        self.feedback_handler = feedback_handler
        self.correction_history = []
    
    def run(self):
        """
        Run the command-line interface.
        """
        print("Brahmi to Devanagari Transliteration")
        print("===================================")
        print("Type 'exit' to quit")
        
        while True:
            # Get input text
            brahmi_text = input("\nEnter Brahmi text: ")
            
            if brahmi_text.lower() == 'exit':
                break
            
            if not brahmi_text:
                print("Error: No input text")
                continue
            
            # Transliterate text
            try:
                # Check if using multi-variant transliterator
                if isinstance(self.transliterator, MultiVariantTransliterator):
                    result = self.transliterator.transliterate(brahmi_text)
                    devanagari_text = result['transliterated_text']
                    variant = result.get('detected_variant', 'Unknown')
                    print(f"Detected variant: {variant}")
                else:
                    devanagari_text = self.transliterator.transliterate(brahmi_text)
                
                # Calculate confidence scores
                confidence_scores = self.confidence_scorer.calculate_character_confidence(
                    brahmi_text, devanagari_text)
                
                # Format with confidence
                formatted_result = self.confidence_scorer.format_with_confidence(
                    devanagari_text, confidence_scores)
                
                # Print transliteration
                print(f"\nTransliteration: {formatted_result['text']}")
                print(f"Average confidence: {formatted_result['average_confidence']:.2f}")
                
                # Print low confidence characters
                if formatted_result['low_confidence_indices']:
                    low_conf_chars = [devanagari_text[i] for i in formatted_result['low_confidence_indices']]
                    print(f"Low confidence characters: {', '.join(low_conf_chars)}")
                
                # Ask for correction
                corrected_text = input("\nEnter correction (press Enter to accept): ")
                
                if corrected_text:
                    # Record correction
                    correction = {
                        'brahmi_text': brahmi_text,
                        'original_text': devanagari_text,
                        'corrected_text': corrected_text,
                        'timestamp': pd.Timestamp.now()
                    }
                    
                    self.correction_history.append(correction)
                    
                    # Send to feedback handler if available
                    if self.feedback_handler:
                        self.feedback_handler.process_correction(correction)
                        
                    print("Correction submitted")
                
            except Exception as e:
                print(f"Error: {str(e)}")
    
    def save_correction_history(self, file_path):
        """
        Save correction history to a file.
        
        Args:
            file_path (str): Path to save the correction history
        """
        # Convert to DataFrame
        df = pd.DataFrame(self.correction_history)
        
        # Save to file
        df.to_csv(file_path, index=False)
    
    def load_correction_history(self, file_path):
        """
        Load correction history from a file.
        
        Args:
            file_path (str): Path to load the correction history from
        """
        # Load from file
        df = pd.read_csv(file_path)
        
        # Convert to list of dictionaries
        self.correction_history = df.to_dict('records')


class WebInterface:
    """
    Web interface for Brahmi to Devanagari transliteration.
    """
    
    def __init__(self, transliterator, confidence_scorer, feedback_handler=None):
        """
        Initialize the web interface.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
            confidence_scorer (ConfidenceScorer): Confidence scorer
            feedback_handler: Handler for user feedback (for self-improvement)
        """
        self.transliterator = transliterator
        self.confidence_scorer = confidence_scorer
        self.feedback_handler = feedback_handler
        self.correction_history = []
        
        # Import Flask
        try:
            from flask import Flask, request, jsonify, render_template
            self.flask = Flask(__name__)
            self.request = request
            self.jsonify = jsonify
            self.render_template = render_template
        except ImportError:
            print("Flask is not installed. Please install it with 'pip install flask'.")
    
    def setup_routes(self):
        """
        Set up Flask routes.
        """
        @self.flask.route('/')
        def index():
            return self.render_template('index.html')
        
        @self.flask.route('/transliterate', methods=['POST'])
        def transliterate():
            data = self.request.json
            brahmi_text = data.get('text', '')
            
            if not brahmi_text:
                return self.jsonify({'error': 'No input text'})
            
            try:
                # Check if using multi-variant transliterator
                if isinstance(self.transliterator, MultiVariantTransliterator):
                    result = self.transliterator.transliterate(brahmi_text)
                    devanagari_text = result['transliterated_text']
                    variant = result.get('detected_variant', 'Unknown')
                    variant_info = {'detected_variant': variant}
                else:
                    devanagari_text = self.transliterator.transliterate(brahmi_text)
                    variant_info = {}
                
                # Calculate confidence scores
                confidence_scores = self.confidence_scorer.calculate_character_confidence(
                    brahmi_text, devanagari_text)
                
                # Format with confidence
                formatted_result = self.confidence_scorer.format_with_confidence(
                    devanagari_text, confidence_scores)
                
                # Combine results
                response = {
                    'text': formatted_result['text'],
                    'confidence_scores': formatted_result['confidence_scores'],
                    'low_confidence_indices': formatted_result['low_confidence_indices'],
                    'average_confidence': formatted_result['average_confidence'],
                    **variant_info
                }
                
                return self.jsonify(response)
                
            except Exception as e:
                return self.jsonify({'error': str(e)})
        
        @self.flask.route('/submit_correction', methods=['POST'])
        def submit_correction():
            data = self.request.json
            brahmi_text = data.get('brahmi_text', '')
            original_text = data.get('original_text', '')
            corrected_text = data.get('corrected_text', '')
            
            if not brahmi_text or not original_text or not corrected_text:
                return self.jsonify({'error': 'Missing text'})
            
            # Record correction
            correction = {
                'brahmi_text': brahmi_text,
                'original_text': original_text,
                'corrected_text': corrected_text,
                'timestamp': pd.Timestamp.now()
            }
            
            self.correction_history.append(correction)
            
            # Send to feedback handler if available
            if self.feedback_handler:
                self.feedback_handler.process_correction(correction)
                
            return self.jsonify({'status': 'success'})
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """
        Run the web interface.
        
        Args:
            host (str): Host to run the server on
            port (int): Port to run the server on
            debug (bool): Whether to run in debug mode
        """
        self.setup_routes()
        self.flask.run(host=host, port=port, debug=debug)
    
    def save_correction_history(self, file_path):
        """
        Save correction history to a file.
        
        Args:
            file_path (str): Path to save the correction history
        """
        # Convert to DataFrame
        df = pd.DataFrame(self.correction_history)
        
        # Save to file
        df.to_csv(file_path, index=False)
    
    def load_correction_history(self, file_path):
        """
        Load correction history from a file.
        
        Args:
            file_path (str): Path to load the correction history from
        """
        # Load from file
        df = pd.read_csv(file_path)
        
        # Convert to list of dictionaries
        self.correction_history = df.to_dict('records')


def create_html_templates(templates_dir):
    """
    Create HTML templates for the web interface.
    
    Args:
        templates_dir (str): Directory to save the templates
    """
    # Create templates directory
    os.makedirs(templates_dir, exist_ok=True)
    
    # Create index.html
    index_html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Brahmi to Devanagari Transliteration</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .container {
                display: flex;
                flex-direction: column;
                gap: 20px;
            }
            .input-section, .output-section, .correction-section {
                border: 1px solid #ddd;
                padding: 15px;
                border-radius: 5px;
            }
            textarea {
                width: 100%;
                min-height: 100px;
                margin-top: 10px;
                padding: 10px;
                border-radius: 5px;
                border: 1px solid #ccc;
            }
            button {
                background-color: #4CAF50;
                color: white;
                padding: 10px 15px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                margin-top: 10px;
            }
            button:hover {
                background-color: #45a049;
            }
            .confidence-bar {
                height: 20px;
                background-color: #f1f1f1;
                border-radius: 5px;
                margin-top: 10px;
            }
            .confidence-fill {
                height: 100%;
                background-color: #4CAF50;
                border-radius: 5px;
                transition: width 0.3s;
            }
            .character {
                display: inline-block;
                padding: 5px;
                margin: 2px;
                border-radius: 3px;
            }
            .low-confidence {
                background-color: #ffeb3b;
            }
            .status {
                margin-top: 20px;
                padding: 10px;
                border-radius: 5px;
                text-align: center;
            }
            .success {
                background-color: #d4edda;
                color: #155724;
            }
            .error {
                background-color: #f8d7da;
                color: #721c24;
            }
        </style>
    </head>
    <body>
        <h1>Brahmi to Devanagari Transliteration</h1>
        
        <div class="container">
            <div class="input-section">
                <h2>Input</h2>
                <textarea id="brahmiText" placeholder="Enter Brahmi text here..."></textarea>
                <button id="transliterateBtn">Transliterate</button>
            </div>
            
            <div class="output-section">
                <h2>Transliteration</h2>
                <div id="variantInfo"></div>
                <div id="devanagariText"></div>
                <div class="confidence-section">
                    <h3>Confidence</h3>
                    <div class="confidence-bar">
                        <div id="confidenceFill" class="confidence-fill"></div>
                    </div>
                    <p id="confidenceValue"></p>
                </div>
            </div>
            
            <div class="correction-section">
                <h2>Correction</h2>
                <textarea id="correctionText" placeholder="Enter corrected text here..."></textarea>
                <button id="submitCorrectionBtn">Submit Correction</button>
            </div>
            
            <div id="status" class="status" style="display: none;"></div>
        </div>
        
        <script>
            document.addEventListener('DOMContentLoaded', function() {
                const transliterateBtn = document.getElementById('transliterateBtn');
                const submitCorrectionBtn = document.getElementById('submitCorrectionBtn');
                const brahmiText = document.getElementById('brahmiText');
                const devanagariText = document.getElementById('devanagariText');
                const correctionText = document.getElementById('correctionText');
                const confidenceFill = document.getElementById('confidenceFill');
                const confidenceValue = document.getElementById('confidenceValue');
                const variantInfo = document.getElementById('variantInfo');
                const status = document.getElementById('status');
                
                let originalText = '';
                
                transliterateBtn.addEventListener('click', async function() {
                    const text = brahmiText.value.trim();
                    
                    if (!text) {
                        showStatus('Please enter Brahmi text', 'error');
                        return;
                    }
                    
                    try {
                        const response = await fetch('/transliterate', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({ text })
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            showStatus(data.error, 'error');
                            return;
                        }
                        
                        // Store original text
                        originalText = data.text;
                        
                        // Display transliteration with confidence highlighting
                        devanagariText.innerHTML = '';
                        for (let i = 0; i < data.text.length; i++) {
                            const char = document.createElement('span');
                            char.textContent = data.text[i];
                            char.classList.add('character');
                            
                            if (data.low_confidence_indices.includes(i)) {
                                char.classList.add('low-confidence');
                                char.title = `Confidence: ${data.confidence_scores[i].toFixed(2)}`;
                            }
                            
                            devanagariText.appendChild(char);
                        }
                        
                        // Update confidence bar
                        const confidencePercent = data.average_confidence * 100;
                        confidenceFill.style.width = `${confidencePercent}%`;
                        confidenceValue.textContent = `Average Confidence: ${confidencePercent.toFixed(2)}%`;
                        
                        // Display variant info if available
                        if (data.detected_variant) {
                            variantInfo.textContent = `Detected Variant: ${data.detected_variant}`;
                            variantInfo.style.display = 'block';
                        } else {
                            variantInfo.style.display = 'none';
                        }
                        
                        // Pre-fill correction text
                        correctionText.value = data.text;
                        
                        showStatus('Transliteration complete', 'success');
                    } catch (error) {
                        showStatus('Error: ' + error.message, 'error');
                    }
                });
                
                submitCorrectionBtn.addEventListener('click', async function() {
                    const brahmi = brahmiText.value.trim();
                    const correction = correctionText.value.trim();
                    
                    if (!brahmi || !originalText || !correction) {
                        showStatus('Missing text for correction', 'error');
                        return;
                    }
                    
                    try {
                        const response = await fetch('/submit_correction', {
                            method: 'POST',
                            headers: {
                                'Content-Type': 'application/json'
                            },
                            body: JSON.stringify({
                                brahmi_text: brahmi,
                                original_text: originalText,
                                corrected_text: correction
                            })
                        });
                        
                        const data = await response.json();
                        
                        if (data.error) {
                            showStatus(data.error, 'error');
                            return;
                        }
                        
                        showStatus('Correction submitted successfully', 'success');
                    } catch (error) {
                        showStatus('Error: ' + error.message, 'error');
                    }
                });
                
                function showStatus(message, type) {
                    status.textContent = message;
                    status.className = 'status ' + type;
                    status.style.display = 'block';
                    
                    setTimeout(() => {
                        status.style.display = 'none';
                    }, 5000);
                }
            });
        </script>
    </body>
    </html>
    """
    
    with open(os.path.join(templates_dir, 'index.html'), 'w', encoding='utf-8') as f:
        f.write(index_html)


if __name__ == "__main__":
    # Example usage
    from src.transliteration_model import BrahmiToDevanagariTransliterator
    
    # Create a transliterator
    transliterator = BrahmiToDevanagariTransliterator()
    
    # Create a confidence scorer
    confidence_scorer = ConfidenceScorer(transliterator)
    
    # Create a command-line interface
    cli = CommandLineInterface(transliterator, confidence_scorer)
    
    # Run the interface
    cli.run()
