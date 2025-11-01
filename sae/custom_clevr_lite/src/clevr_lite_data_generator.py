"""
CLEVR-Lite Dataset Generator for VLM Circuit Analysis

Generates controlled synthetic scenes with:
- Compositional splits (held-out colorxshape combinations)
- Near-tie relational queries for stress-testing comparators
- Optional texture variation
- Fully reproducible with seed control
"""

import numpy as np
import torch
from PIL import Image, ImageDraw
from pathlib import Path
from typing import List
import json
from dataclasses import asdict
from tqdm import tqdm
import random
from src.clevr_lite_config import CLEVRLiteConfig
from src.scene_qa_datamodel import Object, Scene, Question

class CLEVRLiteGenerator:
    """Generate CLEVR-Lite dataset with compositional splits"""
    
    def __init__(
        self,
        output_dir: str,
        num_train: int = 100,
        num_val: int = 20,
        held_out_ratio: float = 0.5,
        near_tie_ratio: float = 0.3,
        texture_ratio: float = 0.0,  # start without textures
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'train' / 'images').mkdir(parents=True, exist_ok=True)
        (self.output_dir / 'val' / 'images').mkdir(parents=True, exist_ok=True)
        
        self.num_train = num_train
        self.num_val = num_val
        self.held_out_ratio = held_out_ratio
        self.near_tie_ratio = near_tie_ratio
        self.texture_ratio = texture_ratio
        self.seed = seed
        
        self.config = CLEVRLiteConfig()
        self.rng = np.random.RandomState(seed)
        random.seed(seed)
        
        # Determine held-out combinations (50% of color×shape pairs)
        self._setup_compositional_split()
        
    def _setup_compositional_split(self):
        """Create held-out colorxshape combinations for compositional generalization"""
        all_combos = [
            (c, s) for c in self.config.COLORS for s in self.config.SHAPES
        ]
        n_held_out = int(len(all_combos) * self.held_out_ratio)
        
        # Deterministic shuffle
        self.rng.shuffle(all_combos)
        
        self.train_combos = set(all_combos[n_held_out:])
        self.held_out_combos = set(all_combos[:n_held_out])
        
        print(f"Training combinations: {len(self.train_combos)}")
        print(f"Held-out combinations: {len(self.held_out_combos)}")
        print(f"Examples held-out: {list(self.held_out_combos)[:3]}")
        
    def _generate_scene(self, allow_held_out: bool = False, force_unique_attribute: bool = True) -> Scene:
        """
        Generate a single scene with 3-5 objects.
        
        Args:
            allow_held_out: If True, allow held-out color x shape combinations
            force_unique_attribute: If True, ensure at least one object has a unique shape or color
                                    (enables unambiguous attribute queries)
        """
        n_objects = self.rng.randint(self.config.MIN_OBJECTS, self.config.MAX_OBJECTS + 1)
        
        objects = []
        positions = []
        
        # Track used attributes for uniqueness constraint
        used_shapes = []
        used_colors = []
        
        for obj_id in range(n_objects):
            # Sample color and shape (respecting compositional split)
            max_attempts = 100
            for _ in range(max_attempts):
                color = self.rng.choice(self.config.COLORS)
                shape = self.rng.choice(self.config.SHAPES)
                
                is_held_out = (color, shape) in self.held_out_combos
                if allow_held_out or not is_held_out:
                    break
            else:
                # Fallback: use any training combo
                color, shape = random.choice(list(self.train_combos))
            
            size = self.rng.choice(self.config.SIZES)
            
            # Sample position with minimum distance constraint
            for _ in range(100):
                x = self.rng.uniform(0.1, 0.9)
                y = self.rng.uniform(0.1, 0.9)
                
                # Check distance to existing objects
                too_close = False
                for px, py in positions:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < self.config.MIN_DISTANCE:
                        too_close = True
                        break
                
                if not too_close:
                    positions.append((x, y))
                    break
            else:
                # Fallback: use random position
                x = self.rng.uniform(0.1, 0.9)
                y = self.rng.uniform(0.1, 0.9)
                positions.append((x, y))
            
            # Convert to pixel coordinates
            px = int(x * self.config.IMG_SIZE)
            py = int(y * self.config.IMG_SIZE)
            obj_size = self.config.LARGE_SIZE if size == 'large' else self.config.SMALL_SIZE
            
            obj = Object(
                id=obj_id,
                shape=shape,
                color=color,
                size=size,
                position=(x, y),
                pixel_box=(px - obj_size, py - obj_size, px + obj_size, py + obj_size)
            )
            objects.append(obj)
            used_shapes.append(shape)
            used_colors.append(color)
        
        # Post-process: Ensure at least one unique attribute for unambiguous questions
        if force_unique_attribute and n_objects >= 2:
            shape_counts = {s: used_shapes.count(s) for s in set(used_shapes)}
            color_counts = {c: used_colors.count(c) for c in set(used_colors)}
            
            has_unique_shape = any(count == 1 for count in shape_counts.values())
            has_unique_color = any(count == 1 for count in color_counts.values())
            
            # If no unique attributes, force one by modifying the last object
            if not has_unique_shape and not has_unique_color:
                # Strategy: Change last object's attribute to create uniqueness
                last_obj = objects[-1]
                
                # Try to make shape unique
                available_shapes = [s for s in self.config.SHAPES if s not in used_shapes[:-1]]
                if available_shapes:
                    new_shape = self.rng.choice(available_shapes)
                    # Check if this combo is allowed
                    is_held_out = (last_obj.color, new_shape) in self.held_out_combos
                    if allow_held_out or not is_held_out:
                        last_obj.shape = new_shape
                    else:
                        # Try to make color unique instead
                        available_colors = [c for c in self.config.COLORS if c not in used_colors[:-1]]
                        if available_colors:
                            new_color = self.rng.choice(available_colors)
                            is_held_out = (new_color, last_obj.shape) in self.held_out_combos
                            if allow_held_out or not is_held_out:
                                last_obj.color = new_color
                else:
                    # Try to make color unique
                    available_colors = [c for c in self.config.COLORS if c not in used_colors[:-1]]
                    if available_colors:
                        new_color = self.rng.choice(available_colors)
                        is_held_out = (new_color, last_obj.shape) in self.held_out_combos
                        if allow_held_out or not is_held_out:
                            last_obj.color = new_color
        
        return Scene(objects=objects, image_path='', scene_id=-1)
    
    def _render_scene(self, scene: Scene, output_path: str):
        """Render scene to image (simple 2D shapes)"""
        img = Image.new('RGB', (self.config.IMG_SIZE, self.config.IMG_SIZE), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)
        
        for obj in scene.objects:
            x1, y1, x2, y2 = obj.pixel_box
            color = self.config.COLOR_RGB[obj.color]
            
            if obj.shape == 'cube':
                draw.rectangle([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
            elif obj.shape == 'sphere':
                draw.ellipse([x1, y1, x2, y2], fill=color, outline=(0, 0, 0), width=2)
            elif obj.shape == 'triangle':
                # Draw as triangle
                draw.polygon([x1, y2, x2, y2, (x1 + x2) // 2, y1], fill=color, outline=(0, 0, 0), width=2)
        
        img.save(output_path)
    
    def _generate_questions(self, scene: Scene, scene_id: int) -> List[Question]:
        """Generate ONLY unambiguous attribute queries for circuit discovery"""
        questions = []
        
        # Count objects by shape and color
        shape_counts = {s: sum(1 for obj in scene.objects if obj.shape == s) 
                        for s in self.config.SHAPES}
        color_counts = {c: sum(1 for obj in scene.objects if obj.color == c) 
                        for c in self.config.COLORS}
        
        # Query color: "What color is the {shape}?" - only when exactly 1 of that shape
        for shape, count in shape_counts.items():
            if count == 1:
                target_obj = next(obj for obj in scene.objects if obj.shape == shape)
                question_text = f"What color is the {shape}?"
                
                questions.append(Question(
                    scene_id=scene_id,
                    question=question_text,
                    answer=target_obj.color,
                    question_type='query_color_unambiguous',
                    image_path=scene.image_path,
                    scene_objects=[asdict(obj) for obj in scene.objects],
                    template_id=0,
                    is_held_out_combo=(target_obj.color, target_obj.shape) in self.held_out_combos,
                ))
        
        # Query shape: "What shape is the {color} object?" - only when exactly 1 of that color
        for color, count in color_counts.items():
            if count == 1:
                target_obj = next(obj for obj in scene.objects if obj.color == color)
                question_text = f"What shape is the {color} object?"
                
                questions.append(Question(
                    scene_id=scene_id,
                    question=question_text,
                    answer=target_obj.shape,
                    question_type='query_shape_unambiguous',
                    image_path=scene.image_path,
                    scene_objects=[asdict(obj) for obj in scene.objects],
                    template_id=1,
                    is_held_out_combo=(target_obj.color, target_obj.shape) in self.held_out_combos,
                ))
        
        # Negation queries: "What color is the object that is NOT {distractor_color}?"
        # Only generate if there are exactly 2 objects (unambiguous without spatial reasoning)
        if len(scene.objects) == 2:
            obj1, obj2 = scene.objects
            
            # Case 1: Ask about obj1, negate obj2's color
            if obj1.color != obj2.color:  # Only if colors differ
                question_text = f"What color is the object that is NOT {obj2.color}?"
                questions.append(Question(
                    scene_id=scene_id,
                    question=question_text,
                    answer=obj1.color,
                    question_type='query_color_negation',
                    image_path=scene.image_path,
                    scene_objects=[asdict(obj) for obj in scene.objects],
                    template_id=2,
                    is_held_out_combo=(obj1.color, obj1.shape) in self.held_out_combos,
                ))
            
            # Case 2: Ask about obj2, negate obj1's color
            if obj1.color != obj2.color:
                question_text = f"What color is the object that is NOT {obj1.color}?"
                questions.append(Question(
                    scene_id=scene_id,
                    question=question_text,
                    answer=obj2.color,
                    question_type='query_color_negation',
                    image_path=scene.image_path,
                    scene_objects=[asdict(obj) for obj in scene.objects],
                    template_id=2,
                    is_held_out_combo=(obj2.color, obj2.shape) in self.held_out_combos,
                ))
        
        return questions
    
    def generate_dataset(self):
        """Generate complete dataset"""
        print(f"Generating {self.num_train} training + {self.num_val} validation samples...")
        
        all_data = {'train': [], 'val': []}
        
        for split, num_samples in [('train', self.num_train), ('val', self.num_val)]:
            print(f"\nGenerating {split} split...")
            
            for scene_id in tqdm(range(num_samples)):
                # Validation includes held-out combinations
                allow_held_out = (split == 'val')
                scene = self._generate_scene(allow_held_out=allow_held_out)
                
                # Render image
                img_filename = f"{scene_id:08d}.png"
                img_path = self.output_dir / split / 'images' / img_filename
                scene.image_path = str(img_path.relative_to(self.output_dir))
                scene.scene_id = scene_id
                
                self._render_scene(scene, str(img_path))
                
                # Generate questions
                questions = self._generate_questions(scene, scene_id)
                all_data[split].extend([asdict(q) for q in questions])
        
        # Save metadata
        for split in ['train', 'val']:
            output_file = self.output_dir / f"{split}_questions.json"
            with open(output_file, 'w') as f:
                json.dump(all_data[split], f, indent=2)
            print(f"Saved {len(all_data[split])} questions to {output_file}")
        
        # Save config
        config_data = {
            'num_train': self.num_train,
            'num_val': self.num_val,
            'held_out_ratio': self.held_out_ratio,
            'train_combos': list(self.train_combos),
            'held_out_combos': list(self.held_out_combos),
            'seed': self.seed,
        }
        with open(self.output_dir / 'config.json', 'w') as f:
            json.dump(config_data, f, indent=2)
        
        print(f"\n✓ Dataset generation complete!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Train samples: {len(all_data['train'])}")
        print(f"  Val samples: {len(all_data['val'])}")