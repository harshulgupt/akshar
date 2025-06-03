"""
Explainable AI tools for attention visualization in Brahmi to Devanagari transliteration
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import json
import pandas as pd
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import base64
import unicodedata
from src.transliteration_model import BrahmiToDevanagariTransliterator
from src.variant_support import MultiVariantTransliterator

class AttentionVisualizer:
    """
    Visualizer for attention weights in the transliteration model.
    """
    
    def __init__(self, transliterator):
        """
        Initialize the attention visualizer.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
        """
        self.transliterator = transliterator
        
        # Set up fonts and colors
        self.cmap = cm.get_cmap('viridis')
        self.font_path = None  # Will be set later
        
        # Try to find a suitable font for Devanagari
        self._find_suitable_font()
    
    def _find_suitable_font(self):
        """
        Find a suitable font for Devanagari text.
        """
        # Common font paths for different operating systems
        font_paths = [
            '/usr/share/fonts/truetype/noto/NotoSansDevanagari-Regular.ttf',
            '/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf',
            '/usr/share/fonts/TTF/DejaVuSans.ttf',
            '/usr/share/fonts/dejavu/DejaVuSans.ttf',
            '/Library/Fonts/Arial Unicode.ttf',
            'C:\\Windows\\Fonts\\arial.ttf'
        ]
        
        for path in font_paths:
            if os.path.exists(path):
                self.font_path = path
                break
    
    def _get_attention_weights(self, brahmi_text, devanagari_text):
        """
        Get attention weights for a given Brahmi-Devanagari pair.
        
        Args:
            brahmi_text (str): Brahmi text
            devanagari_text (str): Devanagari text
            
        Returns:
            numpy.ndarray: Attention weights matrix
        """
        # This is a placeholder for actual attention weight extraction
        # In a real implementation, this would extract weights from the model
        
        # For demonstration, generate random attention weights
        # In a real implementation, these would come from the model's attention mechanism
        brahmi_length = len(brahmi_text)
        devanagari_length = len(devanagari_text)
        
        # Create a random attention matrix with a diagonal bias to simulate attention
        attention_weights = np.random.random((devanagari_length, brahmi_length)) * 0.3
        
        # Add diagonal bias to simulate attention focus
        for i in range(min(devanagari_length, brahmi_length)):
            attention_weights[i, i] = 0.7 + 0.3 * np.random.random()
        
        # Normalize each row
        for i in range(devanagari_length):
            row_sum = attention_weights[i].sum()
            if row_sum > 0:
                attention_weights[i] /= row_sum
        
        return attention_weights
    
    def plot_attention_heatmap(self, brahmi_text, devanagari_text, 
                              attention_weights=None, save_path=None):
        """
        Plot attention heatmap for a given Brahmi-Devanagari pair.
        
        Args:
            brahmi_text (str): Brahmi text
            devanagari_text (str): Devanagari text
            attention_weights (numpy.ndarray, optional): Attention weights matrix
            save_path (str, optional): Path to save the plot
            
        Returns:
            matplotlib.figure.Figure: Heatmap figure
        """
        if attention_weights is None:
            attention_weights = self._get_attention_weights(brahmi_text, devanagari_text)
        
        # Create figure
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        ax = sns.heatmap(attention_weights, cmap='viridis', 
                        xticklabels=list(brahmi_text),
                        yticklabels=list(devanagari_text),
                        cbar_kws={'label': 'Attention Weight'})
        
        # Set labels
        plt.xlabel('Brahmi Characters')
        plt.ylabel('Devanagari Characters')
        plt.title('Attention Weights')
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_attention_visualization(self, brahmi_text, devanagari_text, 
                                      attention_weights=None, save_path=None):
        """
        Create a visualization of attention weights with character highlighting.
        
        Args:
            brahmi_text (str): Brahmi text
            devanagari_text (str): Devanagari text
            attention_weights (numpy.ndarray, optional): Attention weights matrix
            save_path (str, optional): Path to save the visualization
            
        Returns:
            PIL.Image.Image: Visualization image
        """
        if attention_weights is None:
            attention_weights = self._get_attention_weights(brahmi_text, devanagari_text)
        
        # Create image
        width = 800
        height = 400
        image = Image.new('RGB', (width, height), color='white')
        draw = ImageDraw.Draw(image)
        
        # Set up font
        try:
            if self.font_path:
                font_large = ImageFont.truetype(self.font_path, 36)
                font_small = ImageFont.truetype(self.font_path, 24)
            else:
                # Fallback to default font
                font_large = ImageFont.load_default()
                font_small = ImageFont.load_default()
        except Exception:
            # Fallback to default font
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # Draw title
        draw.text((width//2, 20), 'Attention Visualization', 
                 fill='black', font=font_large, anchor='mt')
        
        # Draw Brahmi text
        brahmi_y = 100
        brahmi_chars = list(brahmi_text)
        brahmi_width = width // (len(brahmi_chars) + 1)
        
        for i, char in enumerate(brahmi_chars):
            x = (i + 1) * brahmi_width
            draw.text((x, brahmi_y), char, fill='black', font=font_large, anchor='mt')
        
        # Draw Devanagari text
        devanagari_y = 300
        devanagari_chars = list(devanagari_text)
        devanagari_width = width // (len(devanagari_chars) + 1)
        
        for i, char in enumerate(devanagari_chars):
            x = (i + 1) * devanagari_width
            draw.text((x, devanagari_y), char, fill='black', font=font_large, anchor='mt')
        
        # Draw attention lines
        for i, d_char in enumerate(devanagari_chars):
            d_x = (i + 1) * devanagari_width
            
            for j, b_char in enumerate(brahmi_chars):
                b_x = (j + 1) * brahmi_width
                
                # Get attention weight
                if i < attention_weights.shape[0] and j < attention_weights.shape[1]:
                    weight = attention_weights[i, j]
                else:
                    weight = 0
                
                # Skip low weights for clarity
                if weight < 0.1:
                    continue
                
                # Calculate color based on weight
                color_rgba = self.cmap(weight)
                color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
                
                # Calculate line width based on weight
                line_width = int(1 + weight * 5)
                
                # Draw line
                draw.line((b_x, brahmi_y + 40, d_x, devanagari_y - 10), 
                         fill=color_rgb, width=line_width)
        
        # Save image if path is provided
        if save_path:
            image.save(save_path)
        
        return image
    
    def create_interactive_html(self, brahmi_text, devanagari_text, 
                               attention_weights=None, save_path=None):
        """
        Create an interactive HTML visualization of attention weights.
        
        Args:
            brahmi_text (str): Brahmi text
            devanagari_text (str): Devanagari text
            attention_weights (numpy.ndarray, optional): Attention weights matrix
            save_path (str, optional): Path to save the HTML file
            
        Returns:
            str: HTML content
        """
        if attention_weights is None:
            attention_weights = self._get_attention_weights(brahmi_text, devanagari_text)
        
        # Create HTML content
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Attention Visualization</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    text-align: center;
                }
                .container {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    margin-top: 50px;
                }
                .text-row {
                    display: flex;
                    justify-content: center;
                    margin: 20px 0;
                }
                .char-box {
                    display: inline-block;
                    width: 50px;
                    height: 50px;
                    line-height: 50px;
                    text-align: center;
                    margin: 0 5px;
                    font-size: 24px;
                    border: 1px solid #ccc;
                    border-radius: 5px;
                    position: relative;
                }
                .attention-line {
                    position: absolute;
                    pointer-events: none;
                    z-index: -1;
                }
                .attention-value {
                    position: absolute;
                    font-size: 12px;
                    background-color: white;
                    padding: 2px 5px;
                    border-radius: 3px;
                    border: 1px solid #ccc;
                    display: none;
                }
                .heatmap {
                    margin-top: 30px;
                }
                table.attention-table {
                    border-collapse: collapse;
                    margin-top: 30px;
                }
                table.attention-table td {
                    width: 50px;
                    height: 50px;
                    text-align: center;
                    border: 1px solid #ddd;
                }
                table.attention-table th {
                    padding: 10px;
                    border: 1px solid #ddd;
                    background-color: #f2f2f2;
                }
            </style>
        </head>
        <body>
            <h1>Attention Visualization</h1>
            
            <div class="container">
                <div class="text-row brahmi-row">
        """
        
        # Add Brahmi characters
        for i, char in enumerate(brahmi_text):
            html += f'<div class="char-box brahmi-char" data-index="{i}">{char}</div>'
        
        html += """
                </div>
                
                <div id="attention-lines"></div>
                
                <div class="text-row devanagari-row">
        """
        
        # Add Devanagari characters
        for i, char in enumerate(devanagari_text):
            html += f'<div class="char-box devanagari-char" data-index="{i}">{char}</div>'
        
        html += """
                </div>
                
                <div id="attention-values"></div>
                
                <h2>Attention Heatmap</h2>
                <table class="attention-table">
                    <tr>
                        <th></th>
        """
        
        # Add Brahmi headers
        for char in brahmi_text:
            html += f'<th>{char}</th>'
        
        html += """
                    </tr>
        """
        
        # Add rows with Devanagari characters and attention weights
        for i, d_char in enumerate(devanagari_text):
            html += f'<tr><th>{d_char}</th>'
            
            for j, b_char in enumerate(brahmi_text):
                # Get attention weight
                if i < attention_weights.shape[0] and j < attention_weights.shape[1]:
                    weight = attention_weights[i, j]
                else:
                    weight = 0
                
                # Calculate background color based on weight
                color_rgba = self.cmap(weight)
                color_rgb = tuple(int(c * 255) for c in color_rgba[:3])
                bg_color = f'rgb{color_rgb}'
                
                # Calculate text color (white for dark backgrounds, black for light)
                text_color = 'white' if sum(color_rgb) < 384 else 'black'
                
                html += f'<td style="background-color: {bg_color}; color: {text_color};">{weight:.2f}</td>'
            
            html += '</tr>'
        
        html += """
                </table>
            </div>
            
            <script>
                document.addEventListener('DOMContentLoaded', function() {
                    const brahmiChars = document.querySelectorAll('.brahmi-char');
                    const devanagariChars = document.querySelectorAll('.devanagari-char');
                    const attentionLines = document.getElementById('attention-lines');
                    const attentionValues = document.getElementById('attention-values');
                    
                    // Attention weights
                    const attentionWeights = 
        """
        
        # Add attention weights as JavaScript array
        html += json.dumps(attention_weights.tolist())
        
        html += """
                    ;
                    
                    // Function to draw attention lines
                    function drawAttentionLines() {
                        attentionLines.innerHTML = '';
                        attentionValues.innerHTML = '';
                        
                        const brahmiRow = document.querySelector('.brahmi-row');
                        const devanagariRow = document.querySelector('.devanagari-row');
                        const brahmiRect = brahmiRow.getBoundingClientRect();
                        const devanagariRect = devanagariRow.getBoundingClientRect();
                        
                        for (let i = 0; i < devanagariChars.length; i++) {
                            const dChar = devanagariChars[i];
                            const dRect = dChar.getBoundingClientRect();
                            const dCenterX = dRect.left + dRect.width / 2;
                            const dCenterY = dRect.top + dRect.height / 2;
                            
                            for (let j = 0; j < brahmiChars.length; j++) {
                                const bChar = brahmiChars[j];
                                const bRect = bChar.getBoundingClientRect();
                                const bCenterX = bRect.left + bRect.width / 2;
                                const bCenterY = bRect.top + bRect.height / 2;
                                
                                // Get attention weight
                                const weight = i < attentionWeights.length && j < attentionWeights[i].length
                                    ? attentionWeights[i][j]
                                    : 0;
                                
                                // Skip low weights for clarity
                                if (weight < 0.1) continue;
                                
                                // Calculate color based on weight
                                const r = Math.floor(weight * 255);
                                const g = Math.floor(100 + weight * 155);
                                const b = Math.floor(255 - weight * 255);
                                const color = `rgba(${r}, ${g}, ${b}, ${weight})`;
                                
                                // Calculate line width based on weight
                                const lineWidth = 1 + weight * 5;
                                
                                // Create line
                                const line = document.createElement('div');
                                line.className = 'attention-line';
                                
                                // Calculate line position and dimensions
                                const dx = dCenterX - bCenterX;
                                const dy = dCenterY - bCenterY;
                                const length = Math.sqrt(dx * dx + dy * dy);
                                const angle = Math.atan2(dy, dx) * 180 / Math.PI;
                                
                                line.style.width = `${length}px`;
                                line.style.height = `${lineWidth}px`;
                                line.style.backgroundColor = color;
                                line.style.position = 'absolute';
                                line.style.left = `${bCenterX}px`;
                                line.style.top = `${bCenterY - lineWidth / 2}px`;
                                line.style.transform = `rotate(${angle}deg)`;
                                line.style.transformOrigin = '0 50%';
                                
                                attentionLines.appendChild(line);
                                
                                // Create attention value
                                const value = document.createElement('div');
                                value.className = 'attention-value';
                                value.textContent = weight.toFixed(2);
                                value.style.left = `${bCenterX + dx / 2}px`;
                                value.style.top = `${bCenterY + dy / 2}px`;
                                
                                attentionValues.appendChild(value);
                            }
                        }
                    }
                    
                    // Draw initial attention lines
                    drawAttentionLines();
                    
                    // Update on window resize
                    window.addEventListener('resize', drawAttentionLines);
                    
                    // Show attention values on hover
                    brahmiChars.forEach(char => {
                        char.addEventListener('mouseenter', function() {
                            const index = parseInt(this.dataset.index);
                            
                            // Highlight this character
                            this.style.backgroundColor = '#e0f7fa';
                            
                            // Show relevant attention values
                            const values = document.querySelectorAll('.attention-value');
                            values.forEach((value, i) => {
                                if (i % brahmiChars.length === index) {
                                    value.style.display = 'block';
                                }
                            });
                        });
                        
                        char.addEventListener('mouseleave', function() {
                            // Reset highlight
                            this.style.backgroundColor = '';
                            
                            // Hide attention values
                            const values = document.querySelectorAll('.attention-value');
                            values.forEach(value => {
                                value.style.display = 'none';
                            });
                        });
                    });
                    
                    devanagariChars.forEach(char => {
                        char.addEventListener('mouseenter', function() {
                            const index = parseInt(this.dataset.index);
                            
                            // Highlight this character
                            this.style.backgroundColor = '#e0f7fa';
                            
                            // Show relevant attention values
                            const values = document.querySelectorAll('.attention-value');
                            values.forEach((value, i) => {
                                if (Math.floor(i / brahmiChars.length) === index) {
                                    value.style.display = 'block';
                                }
                            });
                        });
                        
                        char.addEventListener('mouseleave', function() {
                            // Reset highlight
                            this.style.backgroundColor = '';
                            
                            // Hide attention values
                            const values = document.querySelectorAll('.attention-value');
                            values.forEach(value => {
                                value.style.display = 'none';
                            });
                        });
                    });
                });
            </script>
        </body>
        </html>
        """
        
        # Save HTML if path is provided
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(html)
        
        return html
    
    def visualize_character_attention(self, brahmi_text, devanagari_text, 
                                     char_index, attention_weights=None, save_path=None):
        """
        Visualize attention for a specific character.
        
        Args:
            brahmi_text (str): Brahmi text
            devanagari_text (str): Devanagari text
            char_index (int): Index of the Devanagari character to visualize
            attention_weights (numpy.ndarray, optional): Attention weights matrix
            save_path (str, optional): Path to save the visualization
            
        Returns:
            matplotlib.figure.Figure: Visualization figure
        """
        if attention_weights is None:
            attention_weights = self._get_attention_weights(brahmi_text, devanagari_text)
        
        # Check if char_index is valid
        if char_index < 0 or char_index >= len(devanagari_text):
            raise ValueError(f"Character index {char_index} is out of range")
        
        # Get attention weights for the specified character
        if char_index < attention_weights.shape[0]:
            char_weights = attention_weights[char_index]
        else:
            char_weights = np.zeros(len(brahmi_text))
        
        # Create figure
        plt.figure(figsize=(12, 6))
        
        # Plot attention weights
        plt.subplot(2, 1, 1)
        bars = plt.bar(range(len(brahmi_text)), char_weights)
        
        # Color bars based on weight
        for i, bar in enumerate(bars):
            bar.set_color(self.cmap(char_weights[i]))
        
        # Set labels
        plt.xlabel('Brahmi Characters')
        plt.ylabel('Attention Weight')
        plt.title(f'Attention Weights for Devanagari Character: {devanagari_text[char_index]}')
        plt.xticks(range(len(brahmi_text)), list(brahmi_text))
        
        # Plot text with highlighting
        plt.subplot(2, 1, 2)
        
        # Create text representation
        brahmi_chars = list(brahmi_text)
        positions = np.arange(len(brahmi_chars))
        
        # Plot characters
        for i, (char, pos) in enumerate(zip(brahmi_chars, positions)):
            # Calculate color based on attention weight
            weight = char_weights[i] if i < len(char_weights) else 0
            color = self.cmap(weight)
            
            # Plot character with color
            plt.text(pos, 0, char, fontsize=20, ha='center', va='center',
                    color='black', bbox=dict(facecolor=color, alpha=0.7))
        
        # Set plot limits and remove axes
        plt.xlim(-0.5, len(brahmi_chars) - 0.5)
        plt.ylim(-0.5, 0.5)
        plt.axis('off')
        
        # Adjust layout
        plt.tight_layout()
        
        # Save figure if path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return plt.gcf()
    
    def create_attention_gif(self, brahmi_text, devanagari_text, 
                            attention_weights=None, save_path=None, duration=100):
        """
        Create an animated GIF showing attention for each character.
        
        Args:
            brahmi_text (str): Brahmi text
            devanagari_text (str): Devanagari text
            attention_weights (numpy.ndarray, optional): Attention weights matrix
            save_path (str, optional): Path to save the GIF
            duration (int): Duration of each frame in milliseconds
            
        Returns:
            list: List of PIL.Image.Image frames
        """
        if attention_weights is None:
            attention_weights = self._get_attention_weights(brahmi_text, devanagari_text)
        
        # Create frames for each character
        frames = []
        
        for i in range(len(devanagari_text)):
            # Create visualization for this character
            plt.figure(figsize=(10, 6))
            
            # Get attention weights for this character
            if i < attention_weights.shape[0]:
                char_weights = attention_weights[i]
            else:
                char_weights = np.zeros(len(brahmi_text))
            
            # Plot Brahmi text with highlighting
            brahmi_chars = list(brahmi_text)
            positions = np.arange(len(brahmi_chars))
            
            # Plot characters
            for j, (char, pos) in enumerate(zip(brahmi_chars, positions)):
                # Calculate color based on attention weight
                weight = char_weights[j] if j < len(char_weights) else 0
                color = self.cmap(weight)
                
                # Plot character with color
                plt.text(pos, 0.5, char, fontsize=20, ha='center', va='center',
                        color='black', bbox=dict(facecolor=color, alpha=0.7))
            
            # Plot Devanagari character
            plt.text(len(brahmi_chars) / 2, -0.5, devanagari_text[i], fontsize=24, 
                    ha='center', va='center', color='black', 
                    bbox=dict(facecolor='lightblue', alpha=0.7))
            
            # Add arrow
            plt.arrow(len(brahmi_chars) / 2, 0.2, 0, -0.5, head_width=0.2, 
                     head_length=0.1, fc='black', ec='black')
            
            # Set plot limits and remove axes
            plt.xlim(-0.5, len(brahmi_chars) - 0.5)
            plt.ylim(-1, 1)
            plt.axis('off')
            plt.title(f'Transliterating Character {i+1} of {len(devanagari_text)}')
            
            # Save figure to buffer
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()
            
            # Create image from buffer
            buf.seek(0)
            img = Image.open(buf)
            frames.append(img.copy())
            buf.close()
        
        # Save GIF if path is provided
        if save_path and frames:
            frames[0].save(
                save_path, 
                format='GIF', 
                append_images=frames[1:], 
                save_all=True, 
                duration=duration, 
                loop=0
            )
        
        return frames


class ModelExplainer:
    """
    Explainer for the transliteration model's decision-making process.
    """
    
    def __init__(self, transliterator, attention_visualizer):
        """
        Initialize the model explainer.
        
        Args:
            transliterator: Transliterator model (single or multi-variant)
            attention_visualizer (AttentionVisualizer): Attention visualizer
        """
        self.transliterator = transliterator
        self.attention_visualizer = attention_visualizer
    
    def explain_transliteration(self, brahmi_text, output_dir=None):
        """
        Explain the transliteration process for a given Brahmi text.
        
        Args:
            brahmi_text (str): Brahmi text to transliterate
            output_dir (str, optional): Directory to save explanation files
            
        Returns:
            dict: Explanation data
        """
        # Create output directory if specified
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        # Transliterate text
        if isinstance(self.transliterator, MultiVariantTransliterator):
            result = self.transliterator.transliterate(brahmi_text)
            devanagari_text = result['transliterated_text']
            variant = result.get('detected_variant', 'Unknown')
            variant_info = {'detected_variant': variant}
        else:
            devanagari_text = self.transliterator.transliterate(brahmi_text)
            variant_info = {}
        
        # Get attention weights
        attention_weights = self.attention_visualizer._get_attention_weights(
            brahmi_text, devanagari_text)
        
        # Create explanation data
        explanation = {
            'brahmi_text': brahmi_text,
            'devanagari_text': devanagari_text,
            'attention_weights': attention_weights.tolist(),
            **variant_info
        }
        
        # Create visualizations if output directory is specified
        if output_dir:
            # Create heatmap
            heatmap_path = os.path.join(output_dir, 'attention_heatmap.png')
            self.attention_visualizer.plot_attention_heatmap(
                brahmi_text, devanagari_text, attention_weights, heatmap_path)
            explanation['heatmap_path'] = heatmap_path
            
            # Create character visualization
            vis_path = os.path.join(output_dir, 'attention_visualization.png')
            self.attention_visualizer.create_attention_visualization(
                brahmi_text, devanagari_text, attention_weights, vis_path)
            explanation['visualization_path'] = vis_path
            
            # Create interactive HTML
            html_path = os.path.join(output_dir, 'attention_interactive.html')
            self.attention_visualizer.create_interactive_html(
                brahmi_text, devanagari_text, attention_weights, html_path)
            explanation['interactive_html_path'] = html_path
            
            # Create animated GIF
            gif_path = os.path.join(output_dir, 'attention_animation.gif')
            self.attention_visualizer.create_attention_gif(
                brahmi_text, devanagari_text, attention_weights, gif_path)
            explanation['animation_path'] = gif_path
            
            # Save explanation data
            json_path = os.path.join(output_dir, 'explanation.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                # Convert numpy arrays to lists for JSON serialization
                json.dump(explanation, f, ensure_ascii=False, indent=4)
        
        return explanation
    
    def create_explanation_report(self, brahmi_text, output_path):
        """
        Create a comprehensive explanation report for a given Brahmi text.
        
        Args:
            brahmi_text (str): Brahmi text to transliterate
            output_path (str): Path to save the report
            
        Returns:
            str: Path to the report
        """
        # Create temporary directory for explanation files
        temp_dir = os.path.join(os.path.dirname(output_path), 'temp_explanation')
        os.makedirs(temp_dir, exist_ok=True)
        
        # Generate explanation
        explanation = self.explain_transliteration(brahmi_text, temp_dir)
        
        # Create report HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>Transliteration Explanation Report</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    margin: 20px;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                h1, h2, h3 {{
                    color: #2c3e50;
                }}
                .section {{
                    margin-bottom: 30px;
                    padding: 20px;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .text-display {{
                    font-size: 24px;
                    margin: 20px 0;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-radius: 5px;
                }}
                .visualization {{
                    text-align: center;
                    margin: 20px 0;
                }}
                .visualization img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: center;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .character {{
                    display: inline-block;
                    padding: 5px 10px;
                    margin: 5px;
                    border-radius: 3px;
                    background-color: #e0f7fa;
                }}
                .footer {{
                    margin-top: 50px;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <h1>Transliteration Explanation Report</h1>
            
            <div class="section">
                <h2>Input and Output</h2>
                <p><strong>Brahmi Text:</strong></p>
                <div class="text-display">
                    {''.join([f'<span class="character">{c}</span>' for c in explanation['brahmi_text']])}
                </div>
                
                <p><strong>Devanagari Transliteration:</strong></p>
                <div class="text-display">
                    {''.join([f'<span class="character">{c}</span>' for c in explanation['devanagari_text']])}
                </div>
        """
        
        # Add variant information if available
        if 'detected_variant' in explanation:
            html += f"""
                <p><strong>Detected Brahmi Variant:</strong> {explanation['detected_variant']}</p>
            """
        
        html += """
            </div>
            
            <div class="section">
                <h2>Attention Heatmap</h2>
                <p>This heatmap shows how much attention each Devanagari character (rows) pays to each Brahmi character (columns) during transliteration.</p>
                <div class="visualization">
                    <img src="attention_heatmap.png" alt="Attention Heatmap">
                </div>
            </div>
            
            <div class="section">
                <h2>Attention Visualization</h2>
                <p>This visualization shows the connections between Brahmi and Devanagari characters, with line thickness indicating attention strength.</p>
                <div class="visualization">
                    <img src="attention_visualization.png" alt="Attention Visualization">
                </div>
            </div>
            
            <div class="section">
                <h2>Character-by-Character Animation</h2>
                <p>This animation shows the transliteration process character by character.</p>
                <div class="visualization">
                    <img src="attention_animation.gif" alt="Attention Animation">
                </div>
            </div>
            
            <div class="section">
                <h2>Attention Weights Table</h2>
                <p>This table shows the numerical attention weights between each pair of characters.</p>
                <table>
                    <tr>
                        <th></th>
        """
        
        # Add Brahmi headers
        for char in explanation['brahmi_text']:
            html += f'<th>{char}</th>'
        
        html += """
                    </tr>
        """
        
        # Add rows with Devanagari characters and attention weights
        attention_weights = np.array(explanation['attention_weights'])
        for i, d_char in enumerate(explanation['devanagari_text']):
            html += f'<tr><th>{d_char}</th>'
            
            for j, b_char in enumerate(explanation['brahmi_text']):
                # Get attention weight
                if i < attention_weights.shape[0] and j < attention_weights.shape[1]:
                    weight = attention_weights[i, j]
                else:
                    weight = 0
                
                html += f'<td>{weight:.2f}</td>'
            
            html += '</tr>'
        
        html += """
                </table>
            </div>
            
            <div class="section">
                <h2>Interactive Visualization</h2>
                <p>For an interactive visualization of the attention weights, please open the <a href="attention_interactive.html" target="_blank">interactive HTML file</a>.</p>
            </div>
            
            <div class="footer">
                <p>Generated by Brahmi to Devanagari Transliteration System</p>
            </div>
        </body>
        </html>
        """
        
        # Save report
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        return output_path


if __name__ == "__main__":
    # Example usage
    from src.transliteration_model import BrahmiToDevanagariTransliterator
    
    # Create a transliterator
    transliterator = BrahmiToDevanagariTransliterator()
    
    # Create an attention visualizer
    visualizer = AttentionVisualizer(transliterator)
    
    # Create a model explainer
    explainer = ModelExplainer(transliterator, visualizer)
    
    # Example Brahmi text
    brahmi_text = '𑀓𑀸𑀯𑀺'
    
    # Create explanation
    explanation = explainer.explain_transliteration(brahmi_text, 'explanation_output')
    
    # Create explanation report
    report_path = explainer.create_explanation_report(brahmi_text, 'explanation_report.html')
    
    print(f"Explanation report saved to: {report_path}")
