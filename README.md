# meesho-data-challenge
Submission for the CODS-COMAD Data Challenge, Sponsored by Meesho

# Visual Taxonomy Classification Project

This project implements a multi-attribute classification system using ResNet50 for different product categories. The model is designed to predict multiple attributes for each product image using a shared feature extractor.

## Project Structure

```
visual-taxonomy/
├── category_attributes.parquet
├── train.csv
├── test.csv
├── train_images/
│   └── *.jpg
└── merge.ipynb
```

## Data Description

The dataset consists of:
- Product images in JPG format
- Category attributes stored in a parquet file
- Train and test CSV files containing product information and attribute labels

Available categories:
- Men Tshirts
- Sarees
- Kurtis
- Women Tshirts
- Women Tops & Tunics

## Model Architecture

The model uses a ResNet50 backbone with multiple classification heads:

1. **Feature Extractor**: Pre-trained ResNet50 (excluding final classification layer)
2. **Classification Heads**: Separate fully connected layers for each attribute
3. **Fine-tuning**: Last 3 layers of ResNet50 are fine-tuned while earlier layers remain frozen
