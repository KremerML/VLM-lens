

class CLEVRLiteConfig:
    """Dataset generation configuration"""
    # Colors and shapes
    COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan']
    SHAPES = ['cube', 'sphere', 'cylinder']
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
    MIN_OBJECTS = 3
    MAX_OBJECTS = 5
    SMALL_SIZE = 20
    LARGE_SIZE = 35
    MIN_DISTANCE = 0.15  # minimum distance between object centers
    
    # Question templates
    TEMPLATES = {
        'count': [
            "How many {color} objects are there?",
            "How many {shape}s are there?",
            "How many objects are there?",
        ],
        'exist': [
            "Is there a {color} {shape}?",
            "Are there any {color} objects?",
        ],
        'query_color': [
            "What color is the {shape} on the left?",
            "What is the color of the large {shape}?",
        ],
        'query_shape': [
            "What shape is the {color} object?",
            "What is the shape of the leftmost object?",
        ],
        'compare_distance': [
            "Which {color} object is closest to the {color2} {shape2}?",
            "Is the {color} {shape} closer to the {color2} object than the {color3} {shape3}?",
        ],
    }