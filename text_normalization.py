import re

def normalize_atc_text(text: str) -> str:
    text = text.lower().strip()
    
    # Dictionary for numbers
    num_map = {
        "zero": "0", "one": "1", "two": "2", "three": "3",
        "four": "4", "five": "5", "six": "6", "seven": "7",
        "eight": "8", "nine": "9", "niner": "9"
    }
    
    # Replace spelled-out numbers with digits
    for word, digit in num_map.items():
        text = re.sub(rf'\b{word}\b', digit, text)
        
    # Standardize specific ATC phrases
    text = re.sub(r'\bflight level\s*(\d)\s*(\d)\s*(\d)\b', r'FL\1\2\3', text)
    text = re.sub(r'\brunway\s*(\d)\s*(\d)\s*(left|right|center)\b', r'RWY \1\2 \3', text)
    
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text.upper() # خروجی معمولاً در ATC با حروف بزرگ است