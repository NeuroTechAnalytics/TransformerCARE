# TransformerCARE: Speech Analysis Pipeline

## Overview

TransformerCARE is a speech processing pipeline designed to detect cognitive impairment through speech analysis. The pipeline leverages advanced speech transformer models to generate rich representations of speech data. To handle long speech inputs and bypass transformer constraints, we implemented segmentation and aggregation approaches, specifically Voting and Embed-based (averaging embeddings of speech segments). Additionally, we employed audio augmentation techniques, particularly Frequency Masking to augment speech waveforms and enhance model performance.

![Pipeline Overview](https://github.com/NeuroTechAnalytics/TransformerCARE/blob/master/imgs/pipeline.jpg)

*Figure 1: Overview of the TransformerCARE speech processing pipeline.*

## Dataset Description

We measured the performance of TransformerCARE using the DementiaBank speech corpus, which includes recordings from 237 subjects who participated in a picture description task. The subjects comprised 122 cognitively impaired and 115 cognitively normal individuals. The dataset was split into training and testing sets with the following characteristics:

#### Training Data

| Attributes                        | Case Group     | Control Group    |
|-----------------------------------|----------------|------------------|
| Participants                      | 87             | 79               |
| Gender (F/M)                      | 58 / 29        | 52 / 27          |
| Age (mean ± std)                  | 69.72 ± 6.80   | 67.00 ± 6.15     |
| MMSE score (mean ± std)           | 17.44 ± 5.33   | 28.99 ± 1.15     |

#### Testing Data

| Attributes                        | Case Group     | Control Group    |
|-----------------------------------|----------------|------------------|
| Participants                      | 35             | 36               |
| Gender (F/M)                      | 21 / 14        | 23 / 13          |
| Age (mean ± std)                  | 68.51 ± 7.12   | 66.11 ± 6.53     |
| MMSE score (mean ± std)           | 18.86 ± 5.80   | 28.91 ± 1.25     |

## Results

Performance of Transformer Models with Different Aggregation approaches

| Model           | Aggregation Method | F1-Score | AUC    |
|-----------------|--------------------|----------|--------|
| HuBERT      | Embed-Based        | **79.31**    | **81.80**  |
| HuBERT      | Voting             | 78.83    | 81.84  |
| WavLM       | Embed-Based        | 75.19    | 79.88  |
| WavLM       | Voting             | 71.90    | 78.37  |
| Wav2vec 2.0 | Embed-Based        | 72.92    | 76.49  |
| Wav2vec 2.0 | Voting             | 72.72    | 78.09  |
| DistilHuBERT| Embed-Based        | 72.71    | 75.25  |
| DistilHuBERT| Voting             | 69.29    | 73.12  |


Performance of the HuBERT model (with Embed-based approach) using different data augmentation techniques on the dataset.

| Augmentation Technique                | F1-Score | AUC    |
|---------------------------------------|----------|--------|
| None (Original Dataset)           | 79.31    | 81.80  |
| Time Shifting                     | 81.65    | 82.75  |
| Frequency Masking                 | **84.63**    | **86.11**  |


## Usage

To utilize the TransformerCARE pipeline:

1. **Clone the Repository**
   ```bash
   git clone https://github.com/NeuroTechAnalytics/TransformerCARE.git

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   
3. **Run Training**
   ```bash
   python main.py

4. **Adjust Configuration**
   - You can change the transformer model type, speech segment size, or adjust other hyperparameters using `config.py`.


## Contact

For questions or inquiries, please contact h.azadmaleki@gmail.com



