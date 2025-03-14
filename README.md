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

### 1️⃣ Clone the repository
```bash
git clone https://github.com/derdentyler/fine_tuning_project
cd fine_tuning_project
```
### 2️⃣ Install Poetry (if not installed)

On macOS/Linux:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```
On Windows (PowerShell):
```bash
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | python -
```

### 3️⃣ Set up the virtual environment and install dependencies

Poetry automatically creates and manages a virtual environment
```bash
poetry install
```

### 4️⃣ Activate the virtual environment (optional)

Poetry allows running commands without activation, but if needed:
```bash
poetry shell
```

## Running the Project

🚀 **Full pipeline: from data collection to fine-tuning**

### 1️⃣ Data Collection and Preprocessing

Run the data collection script (replace `config.yaml` with your file containing links to videos):

```bash
poetry run python -m src.youtube_dataset_builder.py
```

💡 The collected data will be stored in `data/raw/`.
💡 Processed .json will be saved in `data/processed/`.

### 2️⃣ Exploratory Data Analysis (EDA, Jupyter Notebook)

```bash
poetry run jupyter lab
```

Open `notebooks/eda.ipynb` to analyze the dataset and get handle preprocessing (if necessary).

### 3️⃣ Fine-Tuning the Model Locally

```bash
poetry run python -m src.fine_tune
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

## Unit tests

To use unit and integration tests:

```bash
# all tests
pytest

# only unit
pytest -m unit

# only intergation
pytest -m integration

# last failed test
pytest --lf

# detailed view
pytest -v
```

## Technologies Used

- Python
- PyTorch
- LLMs

## Future Plans

1. Add .env and run.py
2. Add synthetic data creation module (not enough original text data)
3. Fine-tune multiple LLM models
4. Compare fine-tuning results
5. Implement neural network interpretation techniques (e.g., concept activation vectors)

## Contact

For any questions or suggestions, feel free to reach out:
📧 **[alexander.polybinsky@gmail.com](mailto\:alexander.polybinsky@gmail.com)**
