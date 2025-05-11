# Poetry Completion with Language Models

This project explores using pre-trained and fine-tuned language models (LLMs) to complete poetry stanzas where only the first line is provided. The project analyzes how different model parameters and tokenizers affect the quality of the generated poetry, particularly when dealing with different languages (English and Romanian).

## Project Structure

- `poetry_completion.py`: Main script for generating poetry completions using different models and parameters
- `analyze_results.py`: Script for analyzing the results, generating visualizations, and producing a report
- `requirements.txt`: Required dependencies for the project

## Installation

To set up the project, run:

```bash
pip install -r requirements.txt
```

## Usage

You can run the entire workflow using the run_all.py script:

```bash
python run_all.py
```

This will:

1. Prepare the environment (install dependencies)
2. Prepare datasets for training and testing
3. Run the pre-trained model analysis only (skipping fine-tuning by default)
4. Analyze parameter effects
5. Evaluate model performance
6. Open result reports in your browser

### Fine-Tuning Options

Fine-tuning is resource-intensive and time-consuming, so it's disabled by default in the main workflow. You can run the fine-tuning step separately using:

```bash
# Run fine-tuning for all models (might take a long time)
python run_fine_tuning.py

# Run a quicker version with fewer epochs
python run_fine_tuning.py --quick

# Fine-tune only specific models
python run_fine_tuning.py --english-only
python run_fine_tuning.py --romanian-only
python run_fine_tuning.py --nature-only
```

After fine-tuning, you can run the evaluation and analysis without re-training:

```bash
python run_all.py --skip-training
```

### Command-line Options

You can customize the workflow using these options:

```bash
python run_all.py --skip-training     # Skip model training (use existing models)
python run_all.py --skip-environment  # Skip environment preparation
python run_all.py --skip-datasets     # Skip dataset preparation
python run_all.py --skip-parameters   # Skip parameter analysis
python run_all.py --skip-evaluation   # Skip model evaluation
```

### Running Individual Components

You can also run individual components of the workflow:

```bash
# Prepare datasets
python prepare_datasets.py
python test_poetry_samples.py

# Train and generate with models
python poetry_completion.py

# Analyze parameter effects
python visualize_parameters.py

# Evaluate models
python evaluate_poems.py
```

## Project Components

### 1. Dataset Preparation

- `prepare_datasets.py`: Creates poetry datasets for training
- `test_poetry_samples.py`: Prepares test sets with first lines intact

### 2. Poetry Completion

- `poetry_completion.py`: Main script for completing poetry using both pre-trained and fine-tuned models

### 3. Parameter Analysis

- `visualize_parameters.py`: Analyzes the effect of different parameters (temperature, top-k, top-p) on text generation

### 4. Evaluation

- `evaluate_poems.py`: Evaluates the quality of generated poems using metrics like BLEU scores and sentiment analysis

## Tested Models

The project tests and compares the following models:

1. Pre-trained GPT-2 model
2. GPT-2 fine-tuned on English poetry
3. GPT-2 fine-tuned on Romanian poetry
4. GPT-2 fine-tuned on nature-themed poetry (for pastoral style)

## Analysis Questions Addressed

The project addresses the following analysis questions:

1. **Quality differences**: How do pre-trained and fine-tuned models compare in terms of poem quality?

   - This is evaluated using BLEU scores and manual inspection.

2. **English prompts**: How do models perform when the prompt (first line) is in English?

   - Both pre-trained and fine-tuned models are tested with English prompts.

3. **Romanian prompts**: How do models perform when the prompt is in Romanian?

   - Both pre-trained and fine-tuned models are tested with Romanian prompts.

4. **Cross-lingual generation**: What happens when Romanian prompts are used with models trained on English data?

   - Tests are run with Romanian prompts on English-trained models.

5. **Specialization for pastoral poetry**: How can an LLM be personalized for generating nature-focused poetry?
   - A model is fine-tuned on nature-themed poetry and its output compared to other models.

## Results

After running the workflow, you can view detailed results in the generated HTML reports:

- `evaluation_results/report.html`: Contains comparison of different models
- `parameter_results/report.html`: Shows the effect of different parameters on generation

The best-generated poem is saved to `best_poem.txt`.
