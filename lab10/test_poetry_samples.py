"""
Sample poetry for testing poetry completion models.
This file provides examples with first lines intact but subsequent lines "damaged"
to simulate the scenario described in the assignment.
"""

# Romanian poems with first line intact
ROMANIAN_POEMS = [
    {
        "title": "Fiind băiet...",
        "author": "Mihai Eminescu",
        "text": """Fiind băiet păduri cutreieram
Și mă culcam ades lângă izvor,
Iar brațul drept sub cap eu mi-l puneam
S-aud cum apa sună-ncetișor.""",
        "first_line_only": "Fiind băiet păduri cutreieram"
    },
    {
        "title": "Pe lângă plopii fără soț",
        "author": "Mihai Eminescu",
        "text": """Pe lângă plopii fără soț
Adesea am trecut;
Mă cunoșteau vecinii toți,
Tu nu m-ai cunoscut.""",
        "first_line_only": "Pe lângă plopii fără soț"
    },
    {
        "title": "Lacul",
        "author": "Mihai Eminescu",
        "text": """Lacul codrilor albastru
Nuferi galbeni îl încarcă;
Tresărind în cercuri albe
El cutremură o barcă.""",
        "first_line_only": "Lacul codrilor albastru"
    },
    {
        "title": "Dormi adânc",
        "author": "George Coșbuc",
        "text": """Dormi adânc, copil cu bucle,
Dormi adânc şi lin, uşor!
Te-aş trezi, dar vai! cu greu
Poţi dormi în viitor.""",
        "first_line_only": "Dormi adânc, copil cu bucle"
    },
    {
        "title": "O, rămâi",
        "author": "Mihai Eminescu",
        "text": """O, rămâi, rămâi la mine,
Te iubesc atât de mult!
Ale tale doruri toate
Numai eu știu să le-ascult;""",
        "first_line_only": "O, rămâi, rămâi la mine"
    }
]

# English poems with first line intact
ENGLISH_POEMS = [
    {
        "title": "I Wandered Lonely as a Cloud",
        "author": "William Wordsworth",
        "text": """I wandered lonely as a cloud
That floats on high o'er vales and hills,
When all at once I saw a crowd,
A host, of golden daffodils;""",
        "first_line_only": "I wandered lonely as a cloud"
    },
    {
        "title": "Stopping by Woods on a Snowy Evening",
        "author": "Robert Frost",
        "text": """The woods are lovely, dark and deep,
But I have promises to keep,
And miles to go before I sleep,
And miles to go before I sleep.""",
        "first_line_only": "The woods are lovely, dark and deep"
    },
    {
        "title": "Because I could not stop for Death",
        "author": "Emily Dickinson",
        "text": """Because I could not stop for Death –
He kindly stopped for me –
The Carriage held but just Ourselves –
And Immortality.""",
        "first_line_only": "Because I could not stop for Death"
    },
    {
        "title": "The Road Not Taken",
        "author": "Robert Frost",
        "text": """Two roads diverged in a yellow wood,
And sorry I could not travel both
And be one traveler, long I stood
And looked down one as far as I could""",
        "first_line_only": "Two roads diverged in a yellow wood"
    },
    {
        "title": "Hope is the thing with feathers",
        "author": "Emily Dickinson",
        "text": """Hope is the thing with feathers
That perches in the soul,
And sings the tune without the words,
And never stops at all,""",
        "first_line_only": "Hope is the thing with feathers"
    }
]

# Nature-themed poems with first line intact
NATURE_POEMS = [
    {
        "title": "The Forest Path",
        "author": "Anonymous",
        "text": """The forest whispers ancient tales
Of times long past and yet to be
The leaves dance in the gentle breeze
As sunlight filters through the trees""",
        "first_line_only": "The forest whispers ancient tales"
    },
    {
        "title": "Mountain Symphony",
        "author": "Anonymous",
        "text": """Mountains rise against the sky
Streams flow clear and sweet nearby
Flowers bloom in vibrant hue
Nature's canvas ever new""",
        "first_line_only": "Mountains rise against the sky"
    },
    {
        "title": "Lake Reflection",
        "author": "Anonymous",
        "text": """The lake reflects the azure sky
As clouds drift slowly passing by
Reeds sway gently at the shore
Peace abounds forevermore""",
        "first_line_only": "The lake reflects the azure sky"
    },
    {
        "title": "Autumn Glory",
        "author": "Anonymous",
        "text": """Autumn leaves of red and gold
Nature's story being told
Winds whisper through ancient trees
Carrying memories on the breeze""",
        "first_line_only": "Autumn leaves of red and gold"
    },
    {
        "title": "Spring Awakening",
        "author": "Anonymous",
        "text": """The meadow blooms with wildflowers bright
Swaying gently in the light
Birds soar high on graceful wing
As nature's chorus starts to sing""",
        "first_line_only": "The meadow blooms with wildflowers bright"
    }
]

def get_all_first_lines():
    """
    Get all first lines of poems for testing poem completion
    
    Returns:
        dict: Dictionary with language keys and lists of first lines
    """
    return {
        "romanian": [poem["first_line_only"] for poem in ROMANIAN_POEMS],
        "english": [poem["first_line_only"] for poem in ENGLISH_POEMS],
        "nature": [poem["first_line_only"] for poem in NATURE_POEMS]
    }

def get_original_poems():
    """
    Get all original complete poems for evaluation
    
    Returns:
        dict: Dictionary with language keys and lists of complete poems
    """
    return {
        "romanian": ROMANIAN_POEMS,
        "english": ENGLISH_POEMS,
        "nature": NATURE_POEMS
    }

def save_test_sets():
    """
    Save the test sets to files for use in training and evaluation
    """
    # Save Romanian first lines
    with open("romanian_first_lines.txt", "w", encoding="utf-8") as f:
        for poem in ROMANIAN_POEMS:
            f.write(f"{poem['first_line_only']}\n")
    
    # Save English first lines
    with open("english_first_lines.txt", "w", encoding="utf-8") as f:
        for poem in ENGLISH_POEMS:
            f.write(f"{poem['first_line_only']}\n")
    
    # Save Nature first lines
    with open("nature_first_lines.txt", "w", encoding="utf-8") as f:
        for poem in NATURE_POEMS:
            f.write(f"{poem['first_line_only']}\n")
    
    # Save complete poems for reference
    with open("original_poems.txt", "w", encoding="utf-8") as f:
        f.write("=== ROMANIAN POEMS ===\n\n")
        for poem in ROMANIAN_POEMS:
            f.write(f"Title: {poem['title']}\n")
            f.write(f"Author: {poem['author']}\n")
            f.write(f"{poem['text']}\n\n")
        
        f.write("=== ENGLISH POEMS ===\n\n")
        for poem in ENGLISH_POEMS:
            f.write(f"Title: {poem['title']}\n")
            f.write(f"Author: {poem['author']}\n")
            f.write(f"{poem['text']}\n\n")
        
        f.write("=== NATURE POEMS ===\n\n")
        for poem in NATURE_POEMS:
            f.write(f"Title: {poem['title']}\n")
            f.write(f"Author: {poem['author']}\n")
            f.write(f"{poem['text']}\n\n")
    
    print("Test sets saved to files:")
    print("- romanian_first_lines.txt")
    print("- english_first_lines.txt")
    print("- nature_first_lines.txt")
    print("- original_poems.txt")

if __name__ == "__main__":
    save_test_sets()