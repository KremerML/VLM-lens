import re
from sae.custom_clevr_lite.src.clevr_lite_config import CLEVRLiteConfig


def extract_answer(generated_text: str, question_type: str) -> str:
    """
    Extract the actual answer from the model's generated text using question-type-specific regex patterns.
    
    Expected formats:
    - query_shape_unambiguous: "The {color} object is a {shape}." -> extract {shape}
    - query_color_unambiguous: "The {shape} is {color}." -> extract {color}
    - query_color_negation: "The object that is NOT {color_neg} is {color}." -> extract {color}
    """
    
    text = generated_text.strip().lower()
    
    # Allowed vocab from the data generator
    COLORS = set(CLEVRLiteConfig.COLORS)   # ['red','blue','green','yellow','purple','black']
    SHAPES = set(CLEVRLiteConfig.SHAPES)   # ['square','circle','triangle']
    
    # A few common natural synonyms the VLM might produce
    synonym_map = {
        # shapes
        "round": "circle",
        "ball": "circle",
        "spherical": "circle",
        "triangular": "triangle",
        "pyramid": "triangle",     # sometimes VLMs say "pyramid" for 2D triangle-like
        "box": "square",
        "cubic": "square",
        "block": "square",
        
        # colors
        "violet": "purple",
        "magenta": "purple",       # occasionally appears for saturated purple
        "gold": "yellow",
        "golden": "yellow",
        "navy": "blue",
        "azure": "blue",
        "lime": "green",
    }
    
    def normalize_answer(word: str) -> str:
        """Normalize answer using synonym map and validate against vocab."""
        word = word.lower().strip()
        
        # Try direct match first
        if question_type == "query_shape_unambiguous" and word in SHAPES:
            return word
        elif question_type in ["query_color_unambiguous", "query_color_negation"] and word in COLORS:
            return word
        
        # Try synonym mapping
        mapped = synonym_map.get(word)
        if mapped:
            if question_type == "query_shape_unambiguous" and mapped in SHAPES:
                return mapped
            elif question_type in ["query_color_unambiguous", "query_color_negation"] and mapped in COLORS:
                return mapped
        
        return ""
    
    # Apply question-type-specific regex patterns
    if question_type == "query_shape_unambiguous":
        # Pattern: "The {color} object is a {shape}."
        # We want to extract {shape}
        match = re.search(r"the\s+\w+\s+object\s+is\s+a(?:n)?\s+(\w+)", text)
        if match:
            result = normalize_answer(match.group(1))
            if result:
                return result
    
    elif question_type == "query_color_unambiguous":
        # Pattern: "The {shape} is {color}."
        # We want to extract {color}
        match = re.search(r"the\s+\w+\s+is\s+(\w+)", text)
        if match:
            result = normalize_answer(match.group(1))
            if result:
                return result
    
    elif question_type == "query_color_negation":
        # Pattern: "The object that is NOT {color_neg} is {color}."
        # We want to extract {color} (the final one)
        match = re.search(r"the\s+object\s+that\s+is\s+not\s+\w+\s+is\s+(\w+)", text)
        if match:
            result = normalize_answer(match.group(1))
            if result:
                return result
    
    # If nothing valid found, return empty string to mark as incorrect downstream
    return ""