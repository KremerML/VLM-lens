import json
from typing import List, Optional

class CLEVRVocabulary:
    """Minimal vocabulary for CLEVR-Lite questions and answers"""
    
    # Special tokens
    PAD_TOKEN = '[PAD]'
    QCLS_TOKEN = '[QCLS]'
    IMG_TOKEN = '[IMG]'
    SEP_TOKEN = '[SEP]'
    
    # Core vocabulary (deterministic)
    BASE_VOCAB = [
        PAD_TOKEN, QCLS_TOKEN, IMG_TOKEN, SEP_TOKEN,
        # Colors
        'red', 'blue', 'green', 'yellow', 'purple', 'cyan',
        # Shapes
        'cube', 'sphere', 'cylinder', 'cubes', 'spheres', 'cylinders',
        # Sizes
        'small', 'large',
        # Spatial/relational
        'left', 'right', 'top', 'bottom', 'closest', 'farthest',
        'leftmost', 'rightmost', 'topmost', 'bottommost',
        # Question words
        'what', 'how', 'many', 'is', 'are', 'there', 'the', 'a', 'an',
        'color', 'shape', 'size', 'of', 'to', 'than', 'on', 'in',
        'which', 'object', 'objects',
        # Answers (numbers + yes/no)
        'yes', 'no', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10',
        # Punctuation
        '?', '.', ',',
    ]
    
    def __init__(self, questions_file: Optional[str] = None):
        """
        Initialize vocabulary.
        If questions_file provided, extends vocab with any new words found.
        """
        self.token2id = {token: idx for idx, token in enumerate(self.BASE_VOCAB)}
        self.id2token = {idx: token for token, idx in self.token2id.items()}
        
        # Special token IDs
        self.pad_id = self.token2id[self.PAD_TOKEN]
        self.qcls_id = self.token2id[self.QCLS_TOKEN]
        self.img_id = self.token2id[self.IMG_TOKEN]
        self.sep_id = self.token2id[self.SEP_TOKEN]
        
        # Extend vocabulary from questions if provided
        if questions_file:
            self._extend_vocab_from_file(questions_file)
    
    def _extend_vocab_from_file(self, questions_file: str):
        """Add any new tokens found in questions"""
        with open(questions_file, 'r') as f:
            data = json.load(f)
        
        new_tokens = set()
        for item in data:
            # Tokenize question and answer
            q_tokens = self._simple_tokenize(item['question'])
            a_tokens = self._simple_tokenize(item['answer'])
            
            for token in q_tokens + a_tokens:
                if token not in self.token2id:
                    new_tokens.add(token)
        
        # Add new tokens
        for token in sorted(new_tokens):  # sorted for determinism
            idx = len(self.token2id)
            self.token2id[token] = idx
            self.id2token[idx] = token
        
        if new_tokens:
            print(f"Extended vocabulary with {len(new_tokens)} new tokens: {sorted(new_tokens)[:10]}...")
    
    @staticmethod
    def _simple_tokenize(text: str) -> List[str]:
        """Simple whitespace + punctuation tokenizer"""
        # Add spaces around punctuation
        for punct in ['?', '.', ',', '!']:
            text = text.replace(punct, f' {punct} ')
        
        # Split and lowercase
        tokens = text.lower().split()
        return [t for t in tokens if t]  # remove empty strings
    
    def encode(self, text: str) -> List[int]:
        """Convert text to token IDs"""
        tokens = self._simple_tokenize(text)
        return [self.token2id.get(token, self.pad_id) for token in tokens]
    
    def decode(self, ids: List[int]) -> str:
        """Convert token IDs back to text"""
        tokens = [self.id2token.get(idx, self.PAD_TOKEN) for idx in ids]
        # Remove padding and special tokens
        tokens = [t for t in tokens if t not in [self.PAD_TOKEN, self.QCLS_TOKEN, self.IMG_TOKEN, self.SEP_TOKEN]]
        return ' '.join(tokens)
    
    def __len__(self):
        return len(self.token2id)