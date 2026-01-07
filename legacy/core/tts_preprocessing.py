"""
Text preprocessing for TTS to make responses sound more natural.
Expands abbreviations, formats numbers, and normalizes text for speech synthesis.
"""
import re
from typing import Optional


# Common abbreviations that should be expanded for natural speech
ABBREVIATIONS = {
    # Temperature
    r'\b(\d+)\s*°?\s*F\b': r'\1 degrees Fahrenheit',
    r'\b(\d+)\s*°?\s*C\b': r'\1 degrees Celsius',
    r'\b(\d+)\s*degrees?\s*F\b': r'\1 degrees',  # Simplify "degrees F" to just "degrees"
    
    # Units
    r'\bmph\b': 'miles per hour',
    r'\bkph\b': 'kilometers per hour',
    r'\bkm/h\b': 'kilometers per hour',
    r'\bft\b': 'feet',
    r'\bin\b(?=\s*\d|\s*$)': 'inches',  # Only when followed by number or end
    r'\blbs?\b': 'pounds',
    r'\boz\b': 'ounces',
    r'\bkg\b': 'kilograms',
    r'\bmg\b': 'milligrams',
    r'\bml\b': 'milliliters',
    r'\bL\b': 'liters',
    
    # Time
    r'\bhr\b': 'hour',
    r'\bhrs\b': 'hours',
    r'\bmin\b': 'minute',
    r'\bmins\b': 'minutes',
    r'\bsec\b': 'second',
    r'\bsecs\b': 'seconds',
    r'\bam\b': 'A M',
    r'\bpm\b': 'P M',
    r'\bAM\b': 'A M',
    r'\bPM\b': 'P M',
    
    # Common
    r'\bvs\.?\b': 'versus',
    r'\betc\.?\b': 'etcetera',
    r'\be\.g\.': 'for example',
    r'\bi\.e\.': 'that is',
    r'\bDr\.': 'Doctor',
    r'\bMr\.': 'Mister',
    r'\bMrs\.': 'Missus',
    r'\bMs\.': 'Miss',
    r'\bSt\.': 'Street',
    r'\bAve\.': 'Avenue',
    r'\bBlvd\.': 'Boulevard',
    r'\bRd\.': 'Road',
    
    # Tech
    r'\bGB\b': 'gigabytes',
    r'\bMB\b': 'megabytes',
    r'\bKB\b': 'kilobytes',
    r'\bGHz\b': 'gigahertz',
    r'\bMHz\b': 'megahertz',
    
    # Weather-specific
    r'\bNNE\b': 'north northeast',
    r'\bENE\b': 'east northeast',
    r'\bESE\b': 'east southeast',
    r'\bSSE\b': 'south southeast',
    r'\bSSW\b': 'south southwest',
    r'\bWSW\b': 'west southwest',
    r'\bWNW\b': 'west northwest',
    r'\bNNW\b': 'north northwest',
    r'\bNE\b': 'northeast',
    r'\bNW\b': 'northwest',
    r'\bSE\b': 'southeast',
    r'\bSW\b': 'southwest',
    r'\bN\b(?=\s+wind|\s+at)': 'north',
    r'\bS\b(?=\s+wind|\s+at)': 'south',
    r'\bE\b(?=\s+wind|\s+at)': 'east',
    r'\bW\b(?=\s+wind|\s+at)': 'west',
}

# Compile patterns for efficiency
COMPILED_ABBREVIATIONS = [(re.compile(pattern, re.IGNORECASE), replacement) 
                          for pattern, replacement in ABBREVIATIONS.items()]


def expand_abbreviations(text: str) -> str:
    """Expand common abbreviations for more natural speech."""
    for pattern, replacement in COMPILED_ABBREVIATIONS:
        text = pattern.sub(replacement, text)
    return text


def format_numbers(text: str) -> str:
    """Format numbers for more natural speech."""
    # Convert decimal numbers with many places to simpler form
    # e.g., "3.14159" -> "3.14"
    def round_decimal(match):
        num = float(match.group(0))
        if num == int(num):
            return str(int(num))
        return f"{num:.1f}"
    
    text = re.sub(r'\d+\.\d{3,}', round_decimal, text)
    
    # Add "percent" after standalone % 
    text = re.sub(r'(\d+(?:\.\d+)?)\s*%', r'\1 percent', text)
    
    # Format currency with large numbers naturally
    # e.g., "$1,234,567" -> "$1.2 million"
    def simplify_currency(match):
        symbol = match.group(1)
        num_str = match.group(2).replace(',', '')
        # Handle decimal part if present
        decimal_part = match.group(3) if match.group(3) else ""
        num = float(num_str + decimal_part) if decimal_part else int(num_str)
        
        if num >= 1_000_000_000:
            return f"{symbol}{num / 1_000_000_000:.1f} billion".replace('.0 ', ' ')
        elif num >= 1_000_000:
            return f"{symbol}{num / 1_000_000:.1f} million".replace('.0 ', ' ')
        elif num >= 10_000:
            return f"{symbol}{num / 1_000:.0f} thousand"
        return match.group(0)
    
    text = re.sub(r'(\$)(\d{1,3}(?:,\d{3})+)(\.\d+)?', simplify_currency, text)
    
    # Format large numbers with commas spoken naturally (non-currency)
    def simplify_large_number(match):
        num_str = match.group(0).replace(',', '')
        num = int(num_str)
        if num >= 1_000_000_000:
            return f"{num / 1_000_000_000:.1f} billion".replace('.0 ', ' ')
        elif num >= 1_000_000:
            return f"{num / 1_000_000:.1f} million".replace('.0 ', ' ')
        elif num >= 10_000:
            return f"{num / 1_000:.0f} thousand"
        return num_str
    
    text = re.sub(r'(?<!\$)\b\d{1,3}(?:,\d{3})+\b', simplify_large_number, text)
    
    return text


def clean_for_speech(text: str) -> str:
    """Remove or replace characters that don't speak well."""
    # Remove markdown formatting
    text = re.sub(r'\*\*([^*]+)\*\*', r'\1', text)  # Bold
    text = re.sub(r'\*([^*]+)\*', r'\1', text)      # Italic
    text = re.sub(r'`([^`]+)`', r'\1', text)        # Code
    
    # Remove URLs (they sound terrible when spoken)
    text = re.sub(r'https?://\S+', '', text)
    
    # Remove bullet points and list markers
    text = re.sub(r'^[\s]*[-•*]\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'^[\s]*\d+\.\s*', '', text, flags=re.MULTILINE)
    
    # Replace em-dashes and en-dashes with commas for natural pauses
    text = re.sub(r'[—–]', ', ', text)
    
    # Remove parenthetical asides that are too technical
    # e.g., "(UTC-5)" or "(28211)"
    text = re.sub(r'\([A-Z]{2,}-?\d*\)', '', text)
    text = re.sub(r'\(\d{5}\)', '', text)
    
    # Clean up extra whitespace
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


def preprocess_for_tts(text: str) -> str:
    """
    Main preprocessing function for TTS.
    Applies all transformations to make text sound natural when spoken.
    
    Args:
        text: Raw text from LLM response
        
    Returns:
        Preprocessed text optimized for speech synthesis
    """
    if not text:
        return text
    
    # Apply transformations in order
    text = clean_for_speech(text)
    text = expand_abbreviations(text)
    text = format_numbers(text)
    
    # Final cleanup
    text = re.sub(r'\s+', ' ', text)
    text = text.strip()
    
    return text


# Quick test
if __name__ == "__main__":
    test_cases = [
        "The temperature is 58°F with winds at 10 mph from the NE.",
        "It's currently 72 degrees F with 45% humidity.",
        "The high will be 85°F and the low will be 62°F.",
        "Stock price is $1,234,567.89 up 3.5%.",
        "The meeting is at 3:30 PM in the conference room.",
        "Dr. Smith lives on 123 Main St. in Charlotte, NC (28211).",
    ]
    
    for test in test_cases:
        print(f"Original: {test}")
        print(f"Processed: {preprocess_for_tts(test)}")
        print()
