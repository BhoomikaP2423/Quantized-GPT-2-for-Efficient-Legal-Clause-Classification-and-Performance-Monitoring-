# Legal Clause Classification Using GPT-2 and Model Quantization

## 1. Project Overview

This project focuses on fine-tuning a GPT-2-based language model for legal clause classification and applying post-training quantization techniques to reduce model size and evaluate inference efficiency.

The project compares the original model with dynamically and statically quantized versions based on model size, inference time, and classification accuracy.

The implementation is performed using Python in Google Colab.

## 2. Objectives

- Fine-tune a pretrained GPT-2 model for legal clause classification.
- Apply Dynamic Post-Training Quantization (PTQ).
- Apply Static Post-Training Quantization (PTQ).
- Compare original and quantized model performance.
- Measure model size and inference time.
- Visualize the comparison using a performance dashboard.

## 3. Dataset

**Dataset:** LEDGAR from the LexGLUE benchmark.

LEDGAR contains legal provisions and their corresponding clause labels.

For this experiment:
- Six frequently occurring clause categories are selected.
- 500 samples are used for training.
- 500 samples are used for testing.
- Text is tokenized using the pretrained GPT-2 tokenizer.
- Maximum sequence length: 128 tokens.

## 4. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Programming language |
| PyTorch | Model training and quantization |
| Hugging Face Transformers | Pretrained GPT-2 model |
| Hugging Face Datasets | Dataset loading |
| NumPy | Numerical operations |
| Pandas | Results analysis |
| Matplotlib | Dashboard visualizations |
| Google Colab | Development environment |

## 5. Methodology

### Step 1: Dataset Preparation
The LEDGAR dataset is loaded and filtered to select six frequently occurring legal clause categories. Training and testing samples are prepared and tokenized.

### Step 2: GPT-2 Fine-Tuning
A pretrained DistilGPT-2 model with a sequence classification head is fine-tuned on the selected legal clause dataset.

### Step 3: Dynamic Post-Training Quantization
Dynamic INT8 quantization is applied to supported Linear layers. This reduces the precision of model weights and performs activation quantization dynamically during inference.

### Step 4: Static Post-Training Quantization
Static quantization is applied to the classification head. Calibration samples are used to estimate activation ranges before converting the supported layers into quantized operations.

### Step 5: Performance Evaluation
The three model versions are evaluated using the same test dataset.

The evaluation considers:
- Model size in MB.
- Average inference time per sample.
- Classification accuracy.

## 6. Performance
