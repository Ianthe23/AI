"""
Run the fine-tuning step for poetry completion models.
This is a specialized script to run just the fine-tuning part of the poetry completion task.
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

def main():
    parser = argparse.ArgumentParser(description="Run the fine-tuning step for poetry completion")
    parser.add_argument("--quick", action="store_true", 
                        help="Run a quicker version of fine-tuning (less epochs)")
    parser.add_argument("--english-only", action="store_true",
                        help="Only fine-tune the English model")
    parser.add_argument("--romanian-only", action="store_true",
                        help="Only fine-tune the Romanian model")
    parser.add_argument("--nature-only", action="store_true",
                        help="Only fine-tune the nature poetry model")
    
    args = parser.parse_args()
    
    # Ensure datasets are prepared
    if not os.path.exists("english_poetry.txt") or not os.path.exists("romanian_poetry.txt"):
        print("Preparing datasets...")
        if not run_command("python prepare_datasets.py"):
            print("Failed to prepare datasets")
            return 1
    
    # Fine-tune the English model
    if not args.romanian_only and not args.nature_only:
        print("\n=== Fine-tuning English Poetry Model ===")
        command = "python -c \"from poetry_completion import PoetryGenerator, prepare_poetry_corpus; "
        command += "generator = PoetryGenerator('gpt2'); "
        command += "corpus_file = prepare_poetry_corpus('en', 'english_poetry.txt'); "
        
        if args.quick:
            command += "generator.fine_tune(corpus_file, output_dir='./fine-tuned-en', epochs=1, batch_size=2)\""
        else:
            command += "generator.fine_tune(corpus_file, output_dir='./fine-tuned-en', epochs=2, batch_size=2)\""
        
        if not run_command(command):
            print("Failed to fine-tune English model")
            return 1
    
    # Fine-tune the Romanian model
    if not args.english_only and not args.nature_only:
        print("\n=== Fine-tuning Romanian Poetry Model ===")
        command = "python -c \"from poetry_completion import PoetryGenerator, prepare_poetry_corpus; "
        command += "generator = PoetryGenerator('gpt2'); "
        command += "corpus_file = prepare_poetry_corpus('ro', 'romanian_poetry.txt'); "
        
        if args.quick:
            command += "generator.fine_tune(corpus_file, output_dir='./fine-tuned-ro', epochs=1, batch_size=2)\""
        else:
            command += "generator.fine_tune(corpus_file, output_dir='./fine-tuned-ro', epochs=2, batch_size=2)\""
        
        if not run_command(command):
            print("Failed to fine-tune Romanian model")
            return 1
    
    # Fine-tune the nature model
    if not args.english_only and not args.romanian_only:
        print("\n=== Fine-tuning Nature Poetry Model ===")
        
        # Ensure nature poetry corpus exists
        if not os.path.exists("nature_poetry.txt"):
            print("Preparing nature poetry corpus...")
            command = "python -c \"from prepare_datasets import prepare_nature_poetry_dataset; "
            command += "prepare_nature_poetry_dataset('nature_poetry.txt')\""
            
            if not run_command(command):
                print("Failed to prepare nature poetry corpus")
                return 1
        
        command = "python -c \"from poetry_completion import PoetryGenerator; "
        command += "generator = PoetryGenerator('gpt2'); "
        
        if args.quick:
            command += "generator.fine_tune('nature_poetry.txt', output_dir='./nature-tuned', epochs=1, batch_size=2)\""
        else:
            command += "generator.fine_tune('nature_poetry.txt', output_dir='./nature-tuned', epochs=2, batch_size=2)\""
        
        if not run_command(command):
            print("Failed to fine-tune nature model")
            return 1
    
    print("\n=== Fine-tuning Completed Successfully ===")
    print("You can now run the full evaluation with:")
    print("  python run_all.py --skip-training")
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 