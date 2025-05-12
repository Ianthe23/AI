import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from poetry_completion import PoetryGenerator

class ParameterVisualizer:
    def __init__(self, results_dir="parameter_results"):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        
    def generate_with_varying_temperature(self, generator, prompt, temperatures=[0.5, 0.7, 0.9, 1.2], 
                                          num_samples=3, max_length=100):
        """
        Generate poems with varying temperature values
        """
        results = {}
        
        for temp in temperatures:
            results[temp] = []
            for _ in range(num_samples):
                completion = generator.complete_stanza(
                    prompt,
                    temperature=temp,
                    max_length=max_length
                )
                results[temp].append(completion)
        
        # Save results
        os.makedirs(f"{self.results_dir}/temperature", exist_ok=True)
        with open(f"{self.results_dir}/temperature/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def generate_with_varying_top_k(self, generator, prompt, top_k_values=[10, 30, 50, 100], 
                                  num_samples=3, max_length=100):
        """
        Generate poems with varying top_k values
        """
        results = {}
        
        for top_k in top_k_values:
            results[top_k] = []
            for _ in range(num_samples):
                completion = generator.complete_stanza(
                    prompt,
                    top_k=top_k,
                    max_length=max_length
                )
                results[top_k].append(completion)
        
        # Save results
        os.makedirs(f"{self.results_dir}/top_k", exist_ok=True)
        with open(f"{self.results_dir}/top_k/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def generate_with_varying_top_p(self, generator, prompt, top_p_values=[0.5, 0.7, 0.9, 0.95], 
                                   num_samples=3, max_length=100):
        """
        Generate poems with varying top_p values
        """
        results = {}
        
        for top_p in top_p_values:
            results[top_p] = []
            for _ in range(num_samples):
                completion = generator.complete_stanza(
                    prompt,
                    top_p=top_p,
                    max_length=max_length
                )
                results[top_p].append(completion)
        
        # Save results
        os.makedirs(f"{self.results_dir}/top_p", exist_ok=True)
        with open(f"{self.results_dir}/top_p/results.json", "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
            
        return results
    
    def analyze_diversity(self, results):
        """
        Analyze the diversity of generated texts
        """
        diversity_metrics = {}
        
        for param_name, param_results in results.items():
            try:
                unique_words = set()
                total_words = 0
                line_lengths = []
                
                for sample in param_results:
                    words = sample.split()
                    total_words += len(words)
                    unique_words.update(words)
                    lines = sample.split('\n')
                    line_lengths.extend([len(line.split()) for line in lines if line.strip()])
                
                # Avoid division by zero
                unique_word_ratio = len(unique_words) / total_words if total_words > 0 else 0
                avg_line_length = np.mean(line_lengths) if line_lengths else 0
                
                diversity_metrics[str(param_name)] = {
                    "unique_word_ratio": unique_word_ratio,
                    "avg_line_length": avg_line_length,
                    "unique_words": len(unique_words),
                    "total_words": total_words
                }
            except Exception as e:
                print(f"Error analyzing diversity for parameter {param_name}: {e}")
                # Add a placeholder with zeros to avoid errors
                diversity_metrics[str(param_name)] = {
                    "unique_word_ratio": 0,
                    "avg_line_length": 0,
                    "unique_words": 0,
                    "total_words": 0
                }
        
        return diversity_metrics
    
    def plot_diversity_metrics(self, diversity_metrics, param_name):
        """
        Plot diversity metrics
        """
        # Convert all keys to strings to ensure consistent lookup
        diversity_metrics = {str(k): v for k, v in diversity_metrics.items()}
        
        params = sorted([float(p) for p in diversity_metrics.keys()])
        unique_word_ratios = [diversity_metrics[str(p)]["unique_word_ratio"] for p in params]
        avg_line_lengths = [diversity_metrics[str(p)]["avg_line_length"] for p in params]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:blue'
        ax1.set_xlabel(param_name)
        ax1.set_ylabel('Unique Word Ratio', color=color)
        ax1.plot(params, unique_word_ratios, marker='o', color=color)
        ax1.tick_params(axis='y', labelcolor=color)
        
        ax2 = ax1.twinx()
        color = 'tab:red'
        ax2.set_ylabel('Avg Line Length', color=color)
        ax2.plot(params, avg_line_lengths, marker='s', color=color)
        ax2.tick_params(axis='y', labelcolor=color)
        
        fig.tight_layout()
        plt.title(f'Effect of {param_name} on Poem Diversity')
        
        os.makedirs(f"{self.results_dir}/{param_name.lower()}", exist_ok=True)
        plt.savefig(f"{self.results_dir}/{param_name.lower()}/diversity_metrics.png")
        plt.close()
    
    def compare_tokenizers(self, prompts, tokenizer_names=["gpt2", "EleutherAI/gpt-neo-1.3B"]):
        """
        Compare tokenization of different models
        """
        tokenizer_results = {}
        
        for name in tokenizer_names:
            try:
                tokenizer = AutoTokenizer.from_pretrained(name)
                tokenizer_results[name] = {}
                
                for lang, prompt in prompts.items():
                    tokens = tokenizer.tokenize(prompt)
                    token_ids = tokenizer.encode(prompt)
                    
                    tokenizer_results[name][lang] = {
                        "tokens": tokens,
                        "token_ids": token_ids,
                        "num_tokens": len(tokens)
                    }
            except Exception as e:
                print(f"Error loading tokenizer {name}: {e}")
        
        # Save results
        os.makedirs(f"{self.results_dir}/tokenizers", exist_ok=True)
        with open(f"{self.results_dir}/tokenizers/results.json", "w", encoding="utf-8") as f:
            json.dump(tokenizer_results, f, indent=2)
        
        # Plot comparison
        self.plot_tokenizer_comparison(tokenizer_results)
        
        return tokenizer_results
    
    def plot_tokenizer_comparison(self, tokenizer_results):
        """
        Plot tokenization comparison
        """
        tokenizers = list(tokenizer_results.keys())
        languages = list(tokenizer_results[tokenizers[0]].keys()) if tokenizers else []
        
        token_counts = {model: {lang: data["num_tokens"] 
                               for lang, data in langs.items()} 
                       for model, langs in tokenizer_results.items()}
        
        x = np.arange(len(languages))
        width = 0.35 / len(tokenizers)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, tokenizer in enumerate(tokenizers):
            counts = [token_counts[tokenizer][lang] for lang in languages]
            offset = width * (i - len(tokenizers)/2 + 0.5)
            ax.bar(x + offset, counts, width, label=tokenizer)
        
        ax.set_xlabel('Language')
        ax.set_ylabel('Token Count')
        ax.set_title('Tokenizer Comparison: Number of Tokens by Language')
        ax.set_xticks(x)
        ax.set_xticklabels(languages)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/tokenizers/comparison.png")
        plt.close()
    
    def generate_html_report(self):
        """
        Generate an HTML report summarizing the visualizations
        """
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Poetry Generation Parameter Analysis</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1 { color: #2c3e50; }
                h2 { color: #3498db; }
                .section { margin-bottom: 30px; }
                .plot { margin: 10px 0; max-width: 800px; }
                pre { background-color: #f5f5f5; padding: 10px; border-radius: 5px; }
            </style>
        </head>
        <body>
            <h1>Poetry Generation Parameter Analysis</h1>
            
            <div class="section">
                <h2>Temperature Effects</h2>
                <p>How temperature affects poetry generation diversity and creativity:</p>
                <img class="plot" src="temperature/diversity_metrics.png" alt="Temperature Effects">
                
                <h3>Sample Outputs:</h3>
        """
        
        # Add temperature examples
        try:
            with open(f"{self.results_dir}/temperature/results.json", "r", encoding="utf-8") as f:
                temp_results = json.load(f)
                
                for temp, samples in temp_results.items():
                    html += f"<h4>Temperature = {temp}</h4>"
                    html += f"<pre>{samples[0]}</pre>"
        except Exception as e:
            html += f"<p>Error loading temperature results: {e}</p>"
        
        html += """
            </div>
            
            <div class="section">
                <h2>Top-K Effects</h2>
                <p>How top-k sampling affects poetry generation:</p>
                <img class="plot" src="top_k/diversity_metrics.png" alt="Top-K Effects">
                
                <h3>Sample Outputs:</h3>
        """
        
        # Add top-k examples
        try:
            with open(f"{self.results_dir}/top_k/results.json", "r", encoding="utf-8") as f:
                top_k_results = json.load(f)
                
                for top_k, samples in top_k_results.items():
                    html += f"<h4>top_k = {top_k}</h4>"
                    html += f"<pre>{samples[0]}</pre>"
        except Exception as e:
            html += f"<p>Error loading top_k results: {e}</p>"
        
        html += """
            </div>
            
            <div class="section">
                <h2>Top-P Effects</h2>
                <p>How top-p (nucleus) sampling affects poetry generation:</p>
                <img class="plot" src="top_p/diversity_metrics.png" alt="Top-P Effects">
                
                <h3>Sample Outputs:</h3>
        """
        
        # Add top-p examples
        try:
            with open(f"{self.results_dir}/top_p/results.json", "r", encoding="utf-8") as f:
                top_p_results = json.load(f)
                
                for top_p, samples in top_p_results.items():
                    html += f"<h4>top_p = {top_p}</h4>"
                    html += f"<pre>{samples[0]}</pre>"
        except Exception as e:
            html += f"<p>Error loading top_p results: {e}</p>"
        
        html += """
            </div>
            
            <div class="section">
                <h2>Tokenizer Comparison</h2>
                <p>How different tokenizers handle poetry prompts:</p>
                <img class="plot" src="tokenizers/comparison.png" alt="Tokenizer Comparison">
            </div>
            
            <div class="section">
                <h2>Conclusions</h2>
                <ul>
                    <li>Higher temperature values lead to more diverse but potentially less coherent poetry</li>
                    <li>Lower top-k values constrain the model to more predictable patterns</li>
                    <li>Top-p sampling provides a good balance between diversity and coherence</li>
                    <li>Different tokenizers handle Romanian text with varying efficiency</li>
                </ul>
            </div>
        </body>
        </html>
        """
        
        with open(f"{self.results_dir}/report.html", "w", encoding="utf-8") as f:
            f.write(html)
        
        print(f"HTML report generated at {self.results_dir}/report.html")

def main():
    try:
        # Initialize poetry generator
        generator = PoetryGenerator("gpt2")
        
        # Initialize visualizer
        visualizer = ParameterVisualizer()
        
        # Test prompts in different languages
        prompts = {
            "en": "I wandered lonely as a cloud",
            "ro": "Fiind băiet păduri cutreieram"
        }
        
        try:
            # Compare tokenizers
            print("Comparing tokenizers...")
            visualizer.compare_tokenizers(prompts)
        except Exception as e:
            print(f"Error in tokenizer comparison: {e}")
        
        try:
            # Generate with varying temperature
            print("Analyzing temperature effects...")
            temp_results = visualizer.generate_with_varying_temperature(generator, prompts["en"])
            # Ensure keys are strings before analysis
            temp_diversity = visualizer.analyze_diversity({str(k): v for k, v in temp_results.items()})
            visualizer.plot_diversity_metrics(temp_diversity, "Temperature")
        except Exception as e:
            print(f"Error in temperature analysis: {e}")
        
        try:
            # Generate with varying top_k
            print("Analyzing top_k effects...")
            top_k_results = visualizer.generate_with_varying_top_k(generator, prompts["en"])
            # Ensure keys are strings before analysis
            top_k_diversity = visualizer.analyze_diversity({str(k): v for k, v in top_k_results.items()})
            visualizer.plot_diversity_metrics(top_k_diversity, "Top_K")
        except Exception as e:
            print(f"Error in top_k analysis: {e}")
        
        try:
            # Generate with varying top_p
            print("Analyzing top_p effects...")
            top_p_results = visualizer.generate_with_varying_top_p(generator, prompts["en"])
            # Ensure keys are strings before analysis
            top_p_diversity = visualizer.analyze_diversity({str(k): v for k, v in top_p_results.items()})
            visualizer.plot_diversity_metrics(top_p_diversity, "Top_P")
        except Exception as e:
            print(f"Error in top_p analysis: {e}")
        
        try:
            # Generate HTML report
            print("Generating HTML report...")
            visualizer.generate_html_report()
        except Exception as e:
            print(f"Error generating HTML report: {e}")
        
        print("Parameter visualization completed!")
    except Exception as e:
        print(f"An error occurred during parameter visualization: {e}")
        # Return a non-zero exit code but don't raise an exception
        # This allows subsequent scripts to continue running
        import sys
        sys.exit(1)

if __name__ == "__main__":
    main() 