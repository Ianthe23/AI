from datasets import load_dataset
import os
import random

def prepare_english_poetry_dataset(output_file="english_poetry.txt", max_poems=1000):
    """
    Download and prepare an English poetry dataset from Gutenberg.
    
    Args:
        output_file: Path to save the prepared dataset
        max_poems: Maximum number of poems to include
    """
    print(f"Preparing English poetry dataset...")
    
    try:
        # Try to load from Gutenberg poetry dataset
        dataset = load_dataset("cjhutto/gutenberg_poetry", split="train")
        
        # Write poems to file
        with open(output_file, 'w', encoding='utf-8') as f:
            # Take a sample of poems to avoid too large dataset
            sample_size = min(max_poems, len(dataset))
            indices = random.sample(range(len(dataset)), sample_size)
            
            for idx in indices:
                poem = dataset[idx]['text']
                f.write(poem + "\n\n")
                
        print(f"English poetry dataset saved to {output_file}")
        
    except Exception as e:
        print(f"Error loading Gutenberg dataset: {e}")
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
    
    return output_file

def prepare_romanian_poetry_dataset(output_file="romanian_poetry.txt"):
    """
    Prepare a Romanian poetry dataset.
    
    Args:
        output_file: Path to save the prepared dataset
    """
    print(f"Preparing Romanian poetry dataset...")
    
    # Collection of Romanian poems (excerpts from Eminescu, Coșbuc, etc.)
    romanian_poems = [
        "Fiind băiet păduri cutreieram\nȘi mă culcam ades lângă izvor,\nIar brațul drept sub cap eu mi-l puneam\nS-aud cum apa sună-ncetișor.",
        
        "Pe lângă plopii fără soț\nAdesea am trecut;\nMă cunoșteau vecinii toți,\nTu nu m-ai cunoscut.",
        
        "Lacul codrilor albastru\nNuferi galbeni îl încarcă;\nTresărind în cercuri albe\nEl cutremură o barcă.",
        
        "Dormi adânc, copil cu bucle,\nDormi adânc şi lin, uşor!\nTe-aş trezi, dar vai! cu greu\nPoţi dormi în viitor.",
        
        "O, rămâi, rămâi la mine,\nTe iubesc atât de mult!\nAle tale doruri toate\nNumai eu știu să le-ascult;",
        
        "Sara pe deal buciumul sună cu jale,\nTurmele-l urc, stele le scapără-n cale,\nApele plâng, clar izvorând în fântâne;\nSub un salcâm, dragă, m-aștepți tu pe mine.",
        
        "Cobori în jos, luceafăr blând,\nAlunecând pe-o rază,\nPătrunde-n casă și în gând\nȘi viața-mi luminează!",
        
        "Codrule cu râuri line,\nVreme trece, vreme vine,\nTu din tânăr precum ești\nTot mereu întinerești.",
        
        "Somnoroase păsărele\nPe la cuiburi se adună,\nSe ascund în rămurele -\nNoapte bună!",
        
        "Ce te legeni, codrule,\nFără ploaie, fără vânt,\nCu crengile la pământ?\n- De ce nu m-aș legăna,\nDacă trece vremea mea!"
    ]
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for poem in romanian_poems:
            f.write(poem + "\n\n")
    
    print(f"Romanian poetry dataset saved to {output_file}")
    return output_file

def prepare_nature_poetry_dataset(output_file="nature_poetry.txt"):
    """
    Prepare a dataset focused on nature-themed poetry (pastoral style).
    
    Args:
        output_file: Path to save the prepared dataset
    """
    print(f"Preparing nature-themed poetry dataset...")
    
    nature_poems = [
        "The trees sway gently in the breeze\nSunlight filters through the leaves\nA chorus of birds fills the air\nNature's beauty beyond compare",
        
        "Mountains rise against the sky\nStreams flow clear and sweet nearby\nFlowers bloom in vibrant hue\nNature's canvas ever new",
        
        "The lake reflects the azure sky\nAs clouds drift slowly passing by\nReeds sway gently at the shore\nPeace abounds forevermore",
        
        "Autumn leaves of red and gold\nNature's story being told\nWinds whisper through ancient trees\nCarrying memories on the breeze",
        
        "The forest whispers ancient tales\nOf times long past and yet to be\nThe leaves dance in the gentle breeze\nAs sunlight filters through the trees",
        
        "The meadow blooms with wildflowers bright\nSwaying gently in the light\nBirds soar high on graceful wing\nAs nature's chorus starts to sing",
        
        "The river flows with gentle grace\nCarving paths through rock and space\nReflecting skies of deepest blue\nNature's miracles ever true",
        
        "The mountains stand in silent might\nTheir peaks aglow with morning light\nValleys nestled far below\nWhere crystal waters gently flow",
        
        "The rain falls soft upon the ground\nLife-giving water all around\nThe plants drink deep the precious dew\nAs nature starts the day anew",
        
        "The sunset paints the evening sky\nWith colors that enchant the eye\nAs day gives way to starlit night\nNature's wonders a pure delight"
    ]
    
    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        for poem in nature_poems:
            f.write(poem + "\n\n")
    
    print(f"Nature-themed poetry dataset saved to {output_file}")
    return output_file

if __name__ == "__main__":
    # Prepare datasets
    prepare_english_poetry_dataset()
    prepare_romanian_poetry_dataset()
    prepare_nature_poetry_dataset()
    
    print("All datasets prepared successfully!") 