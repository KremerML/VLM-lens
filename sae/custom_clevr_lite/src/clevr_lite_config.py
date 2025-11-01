

class CLEVRLiteConfig:
    """Dataset generation configuration"""
    # Colors and shapes
    COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan']
    SHAPES = ['cube', 'sphere', 'triangle']
    SIZES = ['small', 'large']
    
    # Color RGB values (bright, saturated)
    COLOR_RGB = {
        'red': (220, 20, 60),
        'blue': (30, 144, 255),
        'green': (50, 205, 50),
        'yellow': (255, 215, 0),
        'purple': (147, 112, 219),
        'cyan': (0, 206, 209),
    }
    
    # Image settings
    IMG_SIZE = 224
    CANVAS_SIZE = 1.0  # normalized coordinates
    
    # Object settings
    MIN_OBJECTS = 2
    MAX_OBJECTS = 4
    SMALL_SIZE = 20
    LARGE_SIZE = 35
    MIN_DISTANCE = 0.25  # minimum distance between object centers
    
    # Question templates
    TEMPLATES = {
    # ONLY use these for circuit discovery
    'query_color_unambiguous': [
        "What color is the {shape}?",  # only when exactly 1 of that shape
    ],
    'query_shape_unambiguous': [
        "What shape is the {color} object?",  # only when exactly 1 of that color
    ],
    'query_color_negation': [
        "What color is the object that is NOT {distractor_color}?",
    ],
}