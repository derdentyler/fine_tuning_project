# MindSpeak Research

## Description

**MindSpeak Research** aims to identify distinctions between speech patterns associated with various mental disorders and natural monologues, including creative, professional, and emotional speech.

The project consists of two key stages:

1. **Developing a relevant language model** capable of classifying speech into predefined categories.
2. **Conducting a comparative analysis** to interpret the linguistic features that the model uses to differentiate between categories.

This repository represents the **first stage** of the project, focusing on model training and classification. The second stage, which involves in-depth analysis and interpretation, is planned for the future.

## Project Structure

```
fine_tuning_project/
│-- src/                # Source code
│-- notebooks/          # Jupyter notebooks for exploratory data analysis
│-- tests/              # Unit tests
│-- config.yaml         # Configuration file
│-- README.md           # Project documentation
```

## Installation

To set up the project locally:

```bash
# Clone the repository
git clone https://github.com/YOUR_REPO.git
cd fine_tuning_project

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On macOS/Linux
venv\Scripts\activate    # On Windows

# Install dependencies
pip install -r requirements.txt
```

## Running the Project

🚀 **Full pipeline: from data collection to fine-tuning**

### 1️⃣ Data Collection

Run the data collection script (replace `source_list.txt` with your file containing links to videos):

```bash
python src/data_scraper.py --config config.yaml
```

💡 The collected data will be stored in `data/raw/`.

### 2️⃣ Data Processing (preprocessing, labeling, saving)

```bash
python src/subtitle_preprocessor.py --config config.yaml
python src/dataset_saver.py --config config.yaml
```

💡 Processed files will be saved in `data/processed/`.

### 3️⃣ Exploratory Data Analysis (EDA, Jupyter Notebook)

```bash
jupyter notebook
```

Open `notebooks/eda.ipynb` to analyze the dataset.

### 4️⃣ Fine-Tuning the Model Locally

```bash
python -m src.fine_tune
```

📌 **Important Notes:**

- Model and dataset paths are specified in `config.yaml`.
- Training results will be stored in `checkpoints/`.

## Data Format

The dataset configuration is stored in `config.yaml` and follows this format:

```yaml
categories:
  category_1:
    - "link_1"
    - "link_2"
  category_2:
    - "link_3"
    - "link_4"
```

⚠️ **Important:** The categories should not be strictly divided into **"disorder"** vs. **"normal speech".**&#x20;

Instead, they should encompass a diverse range of expressive speech styles, such as **"hallucinations", "poetry", "emotional speech", etc.**

## Technologies Used

- Python
- PyTorch
- LLMs

## Future Plans

1. Migrate the project to Poetry
2. Add synthetic data to dataset
3. Fine-tune multiple LLM models
4. Compare fine-tuning results
5. Implement neural network interpretation techniques (e.g., concept activation vectors)

## Contact

For any questions or suggestions, feel free to reach out:
📧 **[alexander.polybinsky@gmail.com](mailto\:alexander.polybinsky@gmail.com)**
