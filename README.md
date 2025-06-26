# molformer-transfer-learning
Transfer learning, data selection &amp; PEFT techniques from scratch on MolFormer for molecular property prediction


# MolFormer Transfer Learning with Data Selection & PEFT

This project explores **transfer learning** techniques on the [MolFormer](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct) model for **molecular property prediction**, combining **data selection strategies** and **parameter-efficient fine-tuning (PEFT)** methods.

## 🧠 Project Overview

We investigate how to adapt a pretrained MolFormer model to new molecular tasks using:
- Supervised fine-tuning on the **Lipophilicity dataset**
- **Unsupervised masked language modeling**
- **Data selection via LISSA influence functions** and **MC Dropout**
- **PEFT techniques**: BitFit, LoRA, and iA³ — implemented from scratch

## 📁 Project Structure

### 🔹 Part 1: Pretraining on Lipophilicity
- Fine-tune MolFormer on the Lipophilicity dataset
- Perform additional masked language modeling

### 🔹 Part 2: Influence Function-Based Data Selection
- Use LISSA to compute influence scores on an external dataset
- Select high-influence samples, combine with lipophilicity data
- Fine-tune the model on this curated dataset

### 🔹 Part 3: Uncertainty-Based Selection & PEFT
- Use **MC Dropout** to select uncertain samples for training
- Implement and compare **BitFit**, **LoRA**, and **iA³** from scratch
- Evaluate model performance and parameter efficiency

## 🧪 Datasets

- **Lipophilicity**: [MoleculeNet Benchmark](https://huggingface.co/datasets/scikit-fingerprints/MoleculeNet_Lipophilicity)
- **External dataset**: *(Specify the dataset name and source if possible)*

## 🛠️ Techniques & Tools

- Transformer-based pretraining (MolFormer)
- Transfer learning and domain adaptation
- Influence functions (LISSA)
- Monte Carlo Dropout for uncertainty estimation
- PEFT: BitFit, LoRA, iA³
- PyTorch, HuggingFace Transformers (or other libs used)

## 📊 Results

| Model Variant         | MSE   | MAE   | R²    | Margin (Median) | Params Trained (%) |
|-----------------------|-------|-------|-------|------------------|---------------------|
| Full Fine-tuning      | 0.123 | 0.287 | 0.842 | ±0.056           | 100%                |
| Influence + Fine-tune | 0.118 | 0.271 | 0.856 | ±0.051           | 100%                |
| BitFit (PEFT)         | 0.135 | 0.305 | 0.821 | ±0.065           | ~0.1%               |
| LoRA (PEFT)           | 0.122 | 0.282 | 0.845 | ±0.057           | ~1.5%               |
| iA³ (PEFT)            | 0.120 | 0.278 | 0.850 | ±0.053           | ~1.2%               |


## 🔗 References

- [MolFormer Paper (Ross et al., 2022)](https://arxiv.org/abs/2106.09553)
- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)
- [BitFit: Simple Parameter-Efficient Tuning](https://arxiv.org/abs/2106.10199)
- [iA³: Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2205.05638)
- [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)

---

