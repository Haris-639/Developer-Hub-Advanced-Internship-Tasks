# Multimodal House Price Prediction (Images + Tabular Data)

This project presents a deep learning approach that combines **image data** (house exterior images) and **tabular data** (e.g., number of bedrooms, bathrooms, square footage) to predict house prices. This multimodal method helps improve prediction accuracy by learning both visual and structured signals from real estate data.

## Dataset

We use the **SoCal Housing Dataset** from Kaggle:

- `socal2.csv`: Tabular data containing columns like `image_id`, `bed`, `bath`, `sqft`, `price`, etc.
- `socal_pics/`: Folder containing over 15,000 images named as `0.jpg`, `1.jpg`, ..., `n.jpg`.

### Key Features Used:
- **Image Input**: Raw images resized to 224x224
- **Tabular Input**: `bed`, `bath`, `sqft`, `latitude`, `longitude`
- **Target Variable**: `price` (continuous)

## Task

We aim to solve a **regression problem** where the model learns to predict the price of a house using both:
- Visual appearance (image of the house)
- Numerical details (features from the CSV)

## Project Structure

The notebook includes the following steps:

1. **Dataset Loading**
   - Load images and CSV
   - Merge image paths with corresponding tabular rows

2. **Preprocessing**
   - Resize and normalize images (EfficientNetB0-compatible)
   - Scale tabular data using `StandardScaler`

3. **Model Building**
   - Image branch: `EfficientNetB0` (pretrained on ImageNet)
   - Tabular branch: Dense layers
   - Combined output: Concatenate + Dense layers for final prediction

4. **Model Training**
   - Loss: Mean Squared Error
   - Optimizer: Adam
   - Epochs: 10

5. **Evaluation**
   - MAE (Mean Absolute Error)
   - RMSE (Root Mean Squared Error)

## Model Architecture

- **EfficientNetB0** (frozen) for image feature extraction
- Dense layers for tabular inputs
- Concatenated layer combining both
- Final dense layer for price regression

## Results

- **MAE**: 212205.77
- **RMSE**: 300856.37


## How to Run

This notebook is built to run on **Kaggle Kernels**.

1. Open in [Kaggle](https://www.kaggle.com/)
2. Attach datasets:
   - `house-prices-and-images-socal`
3. Run all cells

