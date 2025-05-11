"""
Main script to run the entire poetry completion workflow.
This includes data preparation, model training, generation, and evaluation.
"""

import os
import subprocess
import argparse
import sys

def run_command(command):
    """Run a command and return its success status"""
    print(f"Running: {command}")
    result = subprocess.run(command, shell=True)
    return result.returncode == 0

def prepare_environment():
    """Prepare the environment by installing required packages"""
    print("=== Preparing environment ===")
    if not run_command("pip install -r requirements.txt"):
        print("Failed to install requirements")
        return False
    
    # Download NLTK data for BLEU score calculation
    try:
        import nltk
        nltk.download('punkt')
    except Exception as e:
        print(f"Failed to download NLTK data: {e}")
        print("Warning: BLEU score calculation may fail")
    
    return True

def prepare_datasets():
    """Prepare datasets for training and testing"""
    print("\n=== Preparing datasets ===")
    
    # Prepare test poem samples
    if not run_command("python test_poetry_samples.py"):
        print("Failed to prepare test poem samples")
        return False
    
    # Prepare training datasets
    if not run_command("python prepare_datasets.py"):
        print("Failed to prepare training datasets")
        return False
    
    return True

def train_models(skip_training=False):
    """Train the models or skip training if requested"""
    if skip_training:
        print("\n=== Skipping model training ===")
        return True
    
    print("\n=== Training models ===")
    if not run_command("python poetry_completion.py --skip-fine-tuning"):
        print("Failed to train models")
        return False
    
    return True

def analyze_parameters():
    """Analyze the effect of parameters on poetry generation"""
    print("\n=== Analyzing parameters ===")
    if not run_command("python visualize_parameters.py"):
        print("Failed to analyze parameters")
        return False
    
    return True

def evaluate_models():
    """Evaluate model performance"""
    print("\n=== Evaluating models ===")
    if not run_command("python evaluate_poems.py"):
        print("Failed to evaluate models")
        return False
    
    return True

def open_results():
    """Open results in browser if available"""
    try:
        import webbrowser
        
        # Try to open evaluation report
        if os.path.exists("evaluation_results/report.html"):
            webbrowser.open("evaluation_results/report.html")
        
        # Try to open parameter analysis report
        if os.path.exists("parameter_results/report.html"):
            webbrowser.open("parameter_results/report.html")
        
    except Exception as e:
        print(f"Failed to open results in browser: {e}")
        print("Please open the HTML reports manually:")
        print("- evaluation_results/report.html")
        print("- parameter_results/report.html")

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Run poetry completion workflow")
    parser.add_argument("--skip-training", action="store_true", 
                        help="Skip model training (use existing models)")
    parser.add_argument("--skip-environment", action="store_true",
                        help="Skip environment preparation")
    parser.add_argument("--skip-datasets", action="store_true",
                        help="Skip dataset preparation")
    parser.add_argument("--skip-parameters", action="store_true",
                        help="Skip parameter analysis")
    parser.add_argument("--skip-evaluation", action="store_true",
                        help="Skip model evaluation")
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    
    # Prepare environment
    if not args.skip_environment:
        if not prepare_environment():
            return 1
    
    # Prepare datasets
    if not args.skip_datasets:
        if not prepare_datasets():
            return 1
    
    # Train models
    if not train_models(args.skip_training):
        return 1
    
    # Analyze parameters
    if not args.skip_parameters:
        if not analyze_parameters():
            return 1
    
    # Evaluate models
    if not args.skip_evaluation:
        if not evaluate_models():
            return 1
    
    # Open results
    open_results()
    
    print("\n=== All tasks completed successfully ===")
    print("Check the HTML reports for detailed results:")
    print("- evaluation_results/report.html")
    print("- parameter_results/report.html")
    
    # Check for the best poem
    if os.path.exists("best_poem.txt"):
        print("\nBest generated poem:")
        with open("best_poem.txt", "r", encoding="utf-8") as f:
            print(f.read())
    
    return 0

if __name__ == "__main__":
    sys.exit(main())