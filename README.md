# AKSHAR
Advanced Knowledge System for Historical Akṣara Recognition

AKSHAR is an advanced deep learning-based framework designed for the transliteration of ancient Brahmi script into modern Devanagari Unicode text. This project aims to assist researchers, historians, and linguists in deciphering, preserving, and studying early South Asian inscriptions by providing an automated, end-to-end transliteration pipeline.
> **Note**: In Sanskrit, Akṣara (अक्षर) refers to a written syllable or character — making it an apt name for a system focused on script recognition and transliteration.
---

## 🔍 Overview

Brahmi is one of the oldest scripts in South Asia and the ancestor of many modern scripts, including Devanagari. Transliteration from Brahmi to Devanagari is challenging due to:

- **Scriptual variations** over time and geography.
- **Inscription degradation** in historical artifacts.
- **Limited annotated data** for supervised learning.
- **Complex character formations** and conjuncts.

AKSHAR addresses these challenges through an end-to-end pipeline integrating:

- A custom **CNN-based OCR module** for character recognition  
- A **Transformer-based transliteration engine**  
- Support for multiple **historical Brahmi variants**  
- A **confidence scoring system** with optional human-in-the-loop correction  
- A **self-improving feedback loop** using active learning  
- Integration of **explainable AI** to visualize attention weights  

---

## Architecture

The AKSHAR pipeline comprises the following modules:

1. **Image Preprocessing**: Normalization and segmentation of raw inscription images.
2. **OCR Component**: MobileNetV2-based classifier for recognizing individual Brahmi glyphs (170+ classes).
3. **Transliteration Engine**: Transformer model with contextual attention mechanisms.
4. **Variant Handler**: Domain adaptation for Ashokan, Gupta, and other Brahmi forms.
5. **Explainability Toolkit**: Attention heatmaps for interpretability.
6. **Feedback Loop**: Few-shot fine-tuning from user corrections.

---

## Dataset (Preliminary)

> The project uses a combination of public datasets, epigraphic corpora, and synthetically generated inscription images. Detailed dataset statistics and sources will be disclosed after paper publication.

- Brahmi characters: ~170 classes
- Synthetic sentence pairs: ~45,000
- Real historical corpus: Includes segments from Corpus Inscriptionum Indicarum and epigraphic glossaries

---

## Getting Started

> Full code, pretrained models, and setup instructions will be released post-publication.

For now, this repository contains:
- Model architecture definitions
- Inference scripts
- Sample outputs
- Placeholder data loaders for Brahmi image datasets

---

## 📈 Performance (Preliminary)

- **OCR Accuracy**: ~95.94% on character-level recognition  
- **Transliteration Accuracy**: ~97.96% character-level, ~93.87% end-to-end  
- **Multi-variant Support**: Adaptable to regional/temporal Brahmi variants

---

## 🛠️ Tech Stack

- Python 3.9
- PyTorch
- OpenCV
- TensorFlow (for visualization utilities)
- SentencePiece tokenizer
- Tesseract (for initial OCR benchmarking)

---

## Intended Use

This tool is designed for:

- Epigraphists working with South Asian inscriptions
- Researchers in computational linguistics and digital humanities
- Developers creating educational tools for ancient languages

---

## ⚖️ License

This repository is released under the MIT License. See [`LICENSE`](./LICENSE) for details.
