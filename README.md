# Legal Clause Classification Using GPT-2 and Model Quantization

## Project Overview

This project focuses on fine-tuning a GPT-2-based language model for legal clause classification and evaluating the effect of post-training quantization on model performance.

The main objective is to reduce model size and improve inference efficiency while maintaining reasonable classification accuracy. This is particularly useful for legal aid workers who need to process large numbers of legal documents efficiently.

## Objectives

- Fine-tune a GPT-2-based model for legal clause classification.
- Apply Dynamic Post-Training Quantization (Dynamic PTQ).
- Apply Static Post-Training Quantization (Static PTQ).
- Compare the original and quantized models.
- Evaluate model size, inference time, and classification accuracy.
- Present the comparison through graphs and a performance dashboard.

## Dataset

**Dataset:** LEDGAR (Legal Entity and Governance Aspects Dataset), available through the LexGLUE benchmark.

The experiment uses six frequently occurring legal clause categories selected from the training dataset.

The dataset is prepared into:
- 500 training samples
- 500 testing samples

The text is tokenized using the GPT-2 tokenizer with a maximum sequence length of 128 tokens.

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- Hugging Face Datasets
- NumPy
- Pandas
- Matplotlib

## Project Workflow

### 1. Dataset Preparation
- Load the LEDGAR dataset.
- Select the six most frequent clause categories.
- Prepare training and testing samples.
- Tokenize the legal text.

### 2. Model Fine-Tuning
- Load the pretrained DistilGPT-2 model with a classification head.
- Fine-tune the model using the selected legal clause samples.
- Evaluate the original model on the test dataset.

### 3. Dynamic Post-Training Quantization
- Convert supported transformer projection layers into Linear layers.
- Apply dynamic INT8 quantization to supported Linear layers.
- Evaluate the quantized model.

### 4. Static Post-Training Quantization
- Apply static quantization to the classification head.
- Calibrate the model using representative training samples.
- Convert the classification head to quantized operations.
- Evaluate its classification accuracy.

### 5. Performance Evaluation
The following metrics are compared:

- **Model Size:** Serialized model state size in MB.
- **Inference Time:** Average time taken to classify one sample in milliseconds.
- **Accuracy:** Percentage of correctly classified legal clauses.

### 6. Dashboard
The project generates
