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
- **External dataset**: An unlabeled external dataset (structure-only) provided for the project. Used for:
  - **Unsupervised masked language modeling**
  - **Influence function-based sample selection**
  - Dataset is available in the `data` directory as `External-Dataset_for_Task2.csv`.

> ⚠️ _Note: The original source of the external dataset is unknown, and it is included here for academic/non-commercial use only._

## 🛠️ Techniques & Tools

- Transformer-based pretraining (MolFormer)
- Transfer learning and domain adaptation
- Influence functions (LISSA)
- Monte Carlo Dropout for uncertainty estimation
- PEFT: BitFit, LoRA, iA³
- PyTorch, HuggingFace Transformers

## 📊 Results

| Model Variant         | MSE   | MAE   | R²    | Margin (Median) | Params Trained (%) | External Dataset Samples |
|-----------------------|-------|-------|-------|------------------|---------------------|--------------------|
| Baseline Model        | 0.491 | 0.528 | 0.655 | 0.420            | 100%                | -
| MLM-pretrained Model  | 0.443 | 0.497 | 0.689 | 0.384            | 100%                | -
| LiSSA-influence based finetuning | 0.354 | 0.432 | 0.736 | 0.317           | 100%                | 100
| MC-Dropout uncertainty based finetuning | 0.404 | 0.473 | 0.714 | 0.334           | 100%                | 30
| BitFit (PEFT)         | 0.666 | 0.634 | 0.526 | 0.509           | ~0.1%               | 30
| LoRA (PEFT)           | 0.441 | 0.496 | 0.686 | 0.385           | ~0.3%               | 30
| iA³ (PEFT)            | 0.761 | 0.687 | 0.486 | 0.572           | ~0.04%               | 30

## 📄 Project Report

See [final_report.pdf](./NNTI_Project_Report.pdf) for a comprehensive explanation of our methodology, experimental setup, results, and analysis.

## 👥 Team

This project was developed collaboratively by:

- **Shreyansh Chakraborty** – PEFT implementations (BitFit, LoRA, iA³), MC Dropout uncertainty selection, final evaluation pipeline.
- **Nischal Narendra Giriyan** – LISSA influence function integration.
- **Michael Gallinger** – MolFormer training on lipophilicity, masked language modeling, metric visualization.


## 🔗 References

- [MolFormer Paper (Ross et al., 2022)](https://arxiv.org/abs/2106.09553)
- [LoRA: Low-Rank Adaptation of LLMs](https://arxiv.org/abs/2106.09685)
- [BitFit: Simple Parameter-Efficient Tuning](https://arxiv.org/abs/2106.10199)
- [iA³: Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2205.05638)
- [Understanding Black-box Predictions via Influence Functions](https://arxiv.org/abs/1703.04730)

---

