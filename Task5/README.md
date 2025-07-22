# Auto Tagging Support Tickets Using LLM

## Overview

This project aims to automatically assign the most appropriate tags to IT support tickets using Large Language Models (LLMs). The task demonstrates three approaches for automated tagging:

- **Zero-shot classification** using a pre-trained model (`facebook/bart-large-mnli`)
- **Few-shot learning** via in-context examples (prompt-based)
- **Fine-tuned transformer model** (`distilbert-base-uncased`) using HuggingFace's `Trainer`

The final output for each ticket is the **Top 3 most probable tags**.


## Objectives

- Automatically tag support tickets into categories
- Use prompt engineering or fine-tuning with an LLM
- Compare zero-shot vs fine-tuned performance
- Apply few-shot learning techniques to improve accuracy
- Output top 3 most probable tags per ticket

## Dataset

A sample dataset of 10 support tickets was manually created. Each ticket contains:
- `ticket_id`: Unique identifier
- `text`: Free-text description of the issue
- `actual_tags`: List of relevant tags (multi-label)

## How to run:
1. Open notebook.ipynb in Jupyter Notebook, JupyterLab, or VS Code with Jupyter extension.
2. Run all the cells in order.
The notebook performs:
- Zero-shot predictions
- Few-shot predictions
- Fine-tuning on sample dataset
- Model evaluation and comparison

## Approaches Used

### 1. Zero-Shot Classification
- Utilizes `facebook/bart-large-mnli`
- No training required
- Assigns top 3 tags directly from model predictions

### 2. Few-Shot Learning (Prompt-Based)
- Embeds 3 example ticket-tag pairs into the prompt
- Simulates few-shot learning for improved context
- Uses the same zero-shot model but guided via examples

### 3. Fine-Tuning a Transformer
- Uses `distilbert-base-uncased` from HuggingFace
- Trained on simplified (single-label) version of the dataset
- Evaluation performed on a test split using accuracy metrics

## Evaluation Metrics

All three methods are evaluated using the following metrics:

- **Top-1 Accuracy**: Whether the first predicted tag matches the actual tag
- **Top-3 Accuracy**: Whether any of the top 3 predicted tags match the actual tags

### Example Entry:
```json
{
  "ticket_id": 1,
  "text": "My internet is not working. I can't connect to any websites.",
  "actual_tags": ["Internet", "Connectivity"]
}
