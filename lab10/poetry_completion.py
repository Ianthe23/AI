import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
import matplotlib.pyplot as plt
import numpy as np
import os
from datasets import load_dataset
import random
import argparse

class PoetryGenerator:
    def __init__(self, model_name="gpt2", device=None):
        self.model_name = model_name
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.tokenizer = None
        self.model = None
        self.load_model()
        
    def load_model(self):
        """Load pre-trained model and tokenizer"""
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
        
        # Special handling for GPT2 tokenizer
        if "gpt2" in self.model_name:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
    def fine_tune(self, poetry_file, output_dir="./fine-tuned-model", epochs=1, batch_size=2):
        """Fine-tune the model on poetry corpus"""
        print(f"Fine-tuning model on {poetry_file}")
        
        # Create dataset
        train_dataset = TextDataset(
            tokenizer=self.tokenizer,
            file_path=poetry_file,
            block_size=64  # Reduced block size
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer, 
            mlm=False
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            overwrite_output_dir=True,
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            save_steps=500,  # Save more frequently
            save_total_limit=1,  # Keep only the latest checkpoint
            logging_steps=100,
            warmup_steps=50,
            weight_decay=0.01,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            data_collator=data_collator,
            train_dataset=train_dataset,
        )
        
        # Start training
        trainer.train()
        
        # Save model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        print(f"Model saved to {output_dir}")
        
    def complete_stanza(self, first_line, num_lines=4, max_length=100, temperature=0.7, top_k=50, top_p=0.95, repetition_penalty=1.2):
        """Complete a stanza given the first line"""
        print(f"Generating completion for: {first_line}")
        
        input_ids = self.tokenizer.encode(first_line, return_tensors="pt").to(self.device)
        
        # Generate text
        output = self.model.generate(
            input_ids, 
            max_length=max_length, 
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Split into lines for a stanza
        lines = generated_text.split('\n')
        
        # Return only the requested number of lines
        return '\n'.join(lines[:num_lines])
    
    def analyze_tokenizer_effect(self, first_line, languages=["en", "ro"]):
        """Analyze how tokenization affects generated poetry"""
        tokens = self.tokenizer.tokenize(first_line)
        token_ids = self.tokenizer.encode(first_line)
        
        print(f"Tokenization of '{first_line}':")
        print(f"Number of tokens: {len(tokens)}")
        print(f"Tokens: {tokens}")
        print(f"Token IDs: {token_ids}")
        
        # Get token probabilities for different continuations
        input_ids = self.tokenizer.encode(first_line, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids)
            predictions = outputs.logits[:, -1, :]
            probabilities = torch.softmax(predictions, dim=-1)
        
        # Get top 5 most likely next tokens
        values, indices = torch.topk(probabilities, 5)
        
        print("\nTop 5 predicted next tokens:")
        for i, (index, prob) in enumerate(zip(indices[0], values[0])):
            token = self.tokenizer.decode([index])
            print(f"{i+1}. '{token}' with probability {prob:.4f}")
            
        return {
            "num_tokens": len(tokens),
            "tokens": tokens,
            "token_ids": token_ids
        }
        
    def compare_parameters(self, first_line, params_list):
        """Compare the effect of different generation parameters"""
        results = []
        
        for params in params_list:
            completion = self.complete_stanza(
                first_line,
                num_lines=params.get("num_lines", 4),
                max_length=params.get("max_length", 100),
                temperature=params.get("temperature", 0.7),
                top_k=params.get("top_k", 50),
                top_p=params.get("top_p", 0.95),
                repetition_penalty=params.get("repetition_penalty", 1.2)
            )
            
            results.append({
                "params": params,
                "completion": completion
            })
            
        return results

def prepare_poetry_corpus(language="en", output_file="poetry_corpus.txt"):
    """Download or prepare a poetry corpus for fine-tuning"""
    
    if language == "en":
        try:
            # Use a different poetry dataset that's available on Hugging Face
            dataset = load_dataset("merve/poetry", split="train")
            
            # Write to file
            with open(output_file, 'w', encoding='utf-8') as f:
                sample_size = min(1000, len(dataset))
                indices = random.sample(range(len(dataset)), sample_size)
                
                for idx in indices:
                    poem = dataset[idx]['content']
                    f.write(poem + "\n\n")
        except Exception as e:
            print(f"Error loading poetry dataset: {e}")
            # Fallback to a small collection of famous poems
            famous_poems = [
                "I wandered lonely as a cloud\nThat floats on high o'er vales and hills,\nWhen all at once I saw a crowd,\nA host, of golden daffodils;",
                "Two roads diverged in a yellow wood,\nAnd sorry I could not travel both\nAnd be one traveler, long I stood\nAnd looked down one as far as I could",
                "Because I could not stop for Death –\nHe kindly stopped for me –\nThe Carriage held but just Ourselves –\nAnd Immortality.",
                "The woods are lovely, dark and deep,\nBut I have promises to keep,\nAnd miles to go before I sleep,\nAnd miles to go before I sleep.",
                "I met a traveller from an antique land,\nWho said—\"Two vast and trunkless legs of stone\nStand in the desert. . . . Near them, on the sand,\nHalf sunk a shattered visage lies"
            ]
            
            with open(output_file, 'w', encoding='utf-8') as f:
                for poem in famous_poems:
                    f.write(poem + "\n\n")
                    
            print(f"Fallback English poetry dataset saved to {output_file}")
    
    elif language == "ro":
        # Romanian poetry - we'll use a small sample here
        romanian_poems = [
            "Fiind băiet păduri cutreieram\nȘi mă culcam ades lângă izvor,\nIar brațul drept sub cap eu mi-l puneam\nS-aud cum apa sună-ncetișor.",
            "Pe lângă plopii fără soț\nAdesea am trecut;\nMă cunoșteau vecinii toți,\nTu nu m-ai cunoscut.",
            "Lacul codrilor albastru\nNuferi galbeni îl încarcă;\nTresărind în cercuri albe\nEl cutremură o barcă.",
            "Dormi adânc, copil cu bucle,\nDormi adânc şi lin, uşor!\nTe-aş trezi, dar vai! cu greu\nPoţi dormi în viitor."
        ]
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for poem in romanian_poems:
                f.write(poem + "\n\n")
    
    print(f"Poetry corpus saved to {output_file}")
    return output_file

def main():
    # Example poetry first lines for testing
    romanian_first_lines = [
        "Fiind băiet păduri cutreieram",
        "Pe lângă plopii fără soț",
        "Lacul codrilor albastru",
        "Dormi adânc, copil cu bucle"
    ]
    
    english_first_lines = [
        "I wandered lonely as a cloud",
        "The woods are lovely, dark and deep",
        "Because I could not stop for Death",
        "Two roads diverged in a yellow wood"
    ]
    
    # Task A: Use a pre-trained model
    print("\n=== Task A: Pre-trained Model Analysis ===")
    pretrained_generator = PoetryGenerator("gpt2")
    
    print("\n--- Romanian First Line ---")
    ro_completion = pretrained_generator.complete_stanza(romanian_first_lines[0])
    print(ro_completion)
    
    print("\n--- English First Line ---")
    en_completion = pretrained_generator.complete_stanza(english_first_lines[0])
    print(en_completion)
    
    # Analyze tokenizer effect
    print("\n--- Tokenizer Analysis ---")
    pretrained_generator.analyze_tokenizer_effect(romanian_first_lines[0])
    pretrained_generator.analyze_tokenizer_effect(english_first_lines[0])
    
    # Compare parameters
    print("\n--- Parameter Analysis ---")
    params_to_test = [
        {"temperature": 0.5, "top_k": 50, "top_p": 0.9},
        {"temperature": 1.0, "top_k": 50, "top_p": 0.9},
    ]
    
    results = pretrained_generator.compare_parameters(english_first_lines[0], params_to_test)
    for i, result in enumerate(results):
        print(f"\nParameters: {result['params']}")
        print(f"Completion:\n{result['completion']}")
    
    # Task B: Fine-tune a model (optional - can be skipped with --skip-fine-tuning)
    print("\n=== Task B: Fine-tuned Model Analysis ===")
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-fine-tuning", action="store_true", help="Skip the fine-tuning process")
    args, _ = parser.parse_known_args()
    
    if args.skip_fine_tuning:
        print("Skipping fine-tuning step (use --skip-fine-tuning to enable this)")
        return
    
    # Prepare corpus - just use a small sample for quick demonstration
    print("Preparing small poetry corpora for demonstration...")
    en_corpus_file = prepare_poetry_corpus("en", "english_poetry.txt")
    ro_corpus_file = prepare_poetry_corpus("ro", "romanian_poetry.txt")
    
    # Fine-tune on English poetry with smaller dataset and fewer epochs
    print("Fine-tuning on English poetry (small demo)...")
    fine_tuned_en = PoetryGenerator("gpt2")
    fine_tuned_en.fine_tune(en_corpus_file, output_dir="./fine-tuned-en", epochs=1, batch_size=2)
    
    # Fine-tune on Romanian poetry with smaller dataset and fewer epochs
    print("Fine-tuning on Romanian poetry (small demo)...")
    fine_tuned_ro = PoetryGenerator("gpt2")
    fine_tuned_ro.fine_tune(ro_corpus_file, output_dir="./fine-tuned-ro", epochs=1, batch_size=2)
    
    # Task C: Compare results and answer questions
    print("\n=== Task C: Comparisons and Analysis ===")
    
    # C.1: Quality differences between pre-trained and fine-tuned
    print("\n--- C.1: Quality Comparison ---")
    
    print("\nPre-trained model on English:")
    print(pretrained_generator.complete_stanza(english_first_lines[0]))
    
    print("\nFine-tuned (English corpus) model on English:")
    print(fine_tuned_en.complete_stanza(english_first_lines[0]))
    
    print("\nPre-trained model on Romanian:")
    print(pretrained_generator.complete_stanza(romanian_first_lines[0]))
    
    print("\nFine-tuned (Romanian corpus) model on Romanian:")
    print(fine_tuned_ro.complete_stanza(romanian_first_lines[0]))
    
    # C.4: Romanian prompt with English corpus
    print("\n--- C.4: Romanian Prompt with English-trained Model ---")
    print(fine_tuned_en.complete_stanza(romanian_first_lines[0]))
    
    # C.5: Nature-focused poetry (pastoral style) - simplified for the demo
    print("\n--- C.5: Nature-focused Poetry Generation ---")
    
    # Create a very small nature corpus for quick demonstration
    nature_poems = [
        "The forest whispers ancient tales\nOf times long past and yet to be\nThe leaves dance in the gentle breeze\nAs sunlight filters through the trees",
        "Mountains rise against the sky\nStreams flow clear and sweet nearby\nFlowers bloom in vibrant hue\nNature's canvas ever new"
    ]
    
    with open("nature_poetry.txt", "w", encoding="utf-8") as f:
        for poem in nature_poems:
            f.write(poem + "\n\n")
    
    # Fine-tune on nature poetry (minimal example)
    print("Fine-tuning on nature poetry (small demo)...")
    nature_tuned = PoetryGenerator("gpt2")
    nature_tuned.fine_tune("nature_poetry.txt", output_dir="./nature-tuned", epochs=1, batch_size=1)
    
    # Generate nature-focused poetry
    pastoral_completion = nature_tuned.complete_stanza("The forest whispers ancient tales")
    print(pastoral_completion)
    
    # Save the best poem
    best_poem = pastoral_completion
    with open("best_poem.txt", "w", encoding="utf-8") as f:
        f.write(best_poem)
    
    print("\nBest poem saved to best_poem.txt")

if __name__ == "__main__":
    main() 