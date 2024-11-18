## Team wdc
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

### Components:

- **Base Model**: ResNet50 pretrained on ImageNet
- **Custom Dataset**: `CategoryDataset` class handling image loading and preprocessing
- **Multi-output Architecture**: Separate classification heads for each attribute
- **Loss Function**: Cross-entropy loss with mask handling for missing attributes

## Data Processing

1. **Image Preprocessing**:
   - Resize to 224x224
   - Convert to RGB
   - Normalize using ImageNet statistics
   ```python
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.ToTensor(),
       transforms.Normalize([0.485, 0.456, 0.406], 
                          [0.229, 0.224, 0.225])
   ])
   ```

2. **Label Processing**:
   - Label encoding for categorical attributes
   - Mask generation for missing attributes
   - Data splitting (90% train, 10% validation)
## Model Training Details

- **Batch Size**: 32
- **Image Size**: 224x224
- **Train/Val Split**: 90/10
- **Optimization**:
  - Optimizer: Adam
  - Loss: Cross-entropy with masking for missing attributes
  - Fine-tuning: Last 3 layers of ResNet50
# Reproducibility

1. **Setup Data**:
   - Place the dataset in the following structure:
     ```
     ./visual-taxonomy/
         category_attributes.parquet
         train.csv
         test.csv
         train_images/
     ```

2. **Select Category**:
   ```python
   category_name = 'Sarees'  # Change to desired category
   category_info = get_category_details(category_name)
   ```

3. **Run Training**:
   - The code will automatically:
     - Load and preprocess data for the selected category
     - Initialize and train the model
     - Generate predictions
     - Save results to a CSV file

4. **Merge Results**:
   - Use `merge.ipynb` to combine results from different categories
  
## Dependencies

- PyTorch
- torchvision
- pandas
- PIL
- sklearn
- matplotlib


