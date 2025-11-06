# Steam Sale Predictor

A machine learning project that predicts whether a video game on Steam will go on sale based on game metadata, pricing, ratings, and other features. This tool can benefit both players (to anticipate discounts) and publishers (to plan sale events).

## Project Overview

The Steam Sale Predictor analyzes game attributes such as:
- Player activity and engagement metrics
- Pricing history and patterns
- Publisher and developer information
- Release timing and game age
- Genre classifications
- User ratings and reviews

By combining data from multiple Kaggle datasets, the model identifies patterns that influence discount likelihood.

## Data Sources

The project uses two Kaggle datasets that are automatically downloaded and merged:

1. **FronkonGames Steam Games Dataset** (`fronkongames/steam-games-dataset`)
   - Comprehensive game metadata
   - Release dates, prices, ratings
   - Developer/publisher information
   - Platform support

2. **NikDavis Steam Store Games** (`nikdavis/steam-store-games`)
   - Additional game information
   - Genre classifications
   - User ratings and reviews
   - Playtime statistics

The datasets are automatically downloaded using the Kaggle API and merged into a hybrid dataset for analysis.

## Project Structure

```
4210-steam-ml/
├── src/
│   ├── parser.py              # Data download, merging, and basic cleaning
│   ├── feature_engineering.py # Feature extraction and engineering
│   ├── preprocess.py          # Data preprocessing for modeling
│   ├── train.py               # Model training script
│   ├── predict.py             # Prediction script
│   └── load_data.py           # Data loading utilities
├── data/
│   ├── kaggle/                # Raw Kaggle datasets
│   │   ├── games.csv
│   │   └── steam.csv
│   └── processed/             # Processed datasets
│       └── steam_kaggle_hybrid.csv
├── models/                     # Saved trained models
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

### Prerequisites

- Python 3.8+
- Kaggle API credentials (for downloading datasets)

### Installation

1. **Clone the repository** (if applicable) or navigate to the project directory

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv myvenv
   # On Windows:
   myvenv\Scripts\activate
   # On Linux/Mac:
   source myvenv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up Kaggle API**:
   - Create a Kaggle account at [kaggle.com](https://www.kaggle.com)
   - Go to Account → API → Create New Token
   - Download `kaggle.json`
   - Place it in `~/.kaggle/kaggle.json` (or `C:\Users\<username>\.kaggle\kaggle.json` on Windows)
   - Install Kaggle CLI: `pip install kaggle`

## Usage

### 1. Data Collection and Processing

Run the parser to download and merge datasets:

```bash
python src/parser.py
```

This script will:
- Check for Kaggle API credentials
- Download both Kaggle datasets
- Extract and merge the CSV files
- Clean and standardize column names
- Handle missing values
- Create a unified hybrid dataset
- Save to `data/processed/steam_kaggle_hybrid.csv`

**Note**: If the CSV files already exist, the download step will be skipped.

### 2. Feature Engineering

The merged dataset includes basic features. For advanced feature engineering, you can extend `feature_engineering.py` to create:
- Genre-based features
- Temporal features (release date breakdowns)
- Interaction features (price × rating, etc.)
- Developer/publisher statistics
- Engagement metrics

### 3. Model Training

Train a model using the processed data:

```bash
python src/train.py
```

This will:
- Load the processed dataset
- Create features for modeling
- Split data into train/test sets
- Train a logistic regression model
- Evaluate performance
- Save the model to `models/logreg_model.pkl`

### 4. Making Predictions

Use the trained model to make predictions:

```bash
python src/predict.py
```

## Key Features

### Data Processing (`parser.py`)

- **Automatic Dataset Download**: Uses Kaggle API to fetch datasets
- **Smart Column Mapping**: Handles different column names across datasets
- **Data Cleaning**: 
  - Price normalization (removes $, commas)
  - Date parsing and standardization
  - Missing value handling
  - Name normalization for merging
- **Merge Strategy**: Outer join to preserve all games from both datasets

### Feature Engineering

The project supports extensive feature engineering including:

- **Numeric Features**: Price, game age, ratings, playtime, etc.
- **Categorical Features**: Developer, Publisher, genres
- **Binary Features**: Platform support, genre flags, sale indicators
- **Temporal Features**: Release year, month, quarter, game age
- **Interaction Features**: Price × Rating, Age × Price
