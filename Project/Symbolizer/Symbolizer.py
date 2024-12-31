from collections import defaultdict, Counter
import re
import numpy as np
from typing import Dict, Tuple, Generator
from multiprocessing import Pool, cpu_count
import json
import mmap
import os
from pathlib import Path
from tqdm import tqdm
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class Symbolizer:
    def __init__(self, 
                 max_vocab_size: int = 50000,
                 min_freq: int = 2,
                 max_subword_length: int = 20,
                 n_workers: int = None,
                 file_extensions: set = {'.txt', '.md', '.py', '.java', '.cpp', '.h', '.c', '.js', '.html', '.css'}):
        """
        Initialize the symbolizer with configurable parameters
        
        Args:
            max_vocab_size: Maximum number of symbols to keep
            min_freq: Minimum frequency for a symbol to be included
            max_subword_length: Maximum length of subwords to consider
            n_workers: Number of processes for parallel processing
            file_extensions: Set of file extensions to process
        """
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.max_subword_length = max_subword_length
        self.n_workers = n_workers or max(1, cpu_count() - 1)
        self.file_extensions = file_extensions
        
        # Special tokens
        self.UNKNOWN_TOKEN = "<UNK>"
        self.END_OF_LINE_TOKEN = "<EOL>"
        self.PAD_TOKEN = "<PAD>"
        self.FILE_START_TOKEN = "<FILE_START>"
        self.FILE_END_TOKEN = "<FILE_END>"
        
        # Initialize vocabularies
        self.vocabulary: Dict[str, int] = {}
        self.reverse_vocab: Dict[int, str] = {}
        self.frequencies: Dict[str, int] = defaultdict(int)
        
        # Compile regex patterns for better performance
        self._word_pattern = re.compile(r'\S+')
        self._repeating_char_pattern = re.compile(r'(.)\1+')
        
        # Initialize trie for faster token matching
        self.trie = {}
        
        self._initialize_special_tokens()

    def _initialize_special_tokens(self):
        """Initialize special tokens in vocabulary"""
        special_tokens = [
            self.PAD_TOKEN, 
            self.UNKNOWN_TOKEN, 
            self.END_OF_LINE_TOKEN,
            self.FILE_START_TOKEN,
            self.FILE_END_TOKEN
        ]
        for idx, token in enumerate(special_tokens):
            self.vocabulary[token] = idx
            self.reverse_vocab[idx] = token
            self.frequencies[token] = float('inf')

    def _build_trie(self):
        """Build trie structure for efficient token matching"""
        self.trie = {}
        for token in self.vocabulary:
            current = self.trie
            for char in token:
                if char not in current:
                    current[char] = {}
                current = current[char]
            current['$'] = token  # Mark end of token

    def _find_longest_token(self, text: str, start: int, dp_cache: Dict[int, Tuple[int, str]]) -> Tuple[int, str]:
        """
        Find longest matching token starting at given position using dynamic programming
        
        Args:
            text: Input text
            start: Starting position
            dp_cache: Dynamic programming cache
            
        Returns:
            Tuple of (token length, token)
        """
        if start in dp_cache:
            return dp_cache[start]

        if start >= len(text):
            return (0, self.UNKNOWN_TOKEN)

        current = self.trie
        longest_match = (1, self.UNKNOWN_TOKEN)
        
        for i in range(start, min(start + self.max_subword_length, len(text))):
            char = text[i]
            if char not in current:
                break
            current = current[char]
            if '$' in current:  # Found a complete token
                token = current['$']
                longest_match = (len(token), token)

        dp_cache[start] = longest_match
        return longest_match

    def _find_text_files(self, root_dir: str) -> Generator[Path, None, None]:
        """
        Recursively find all text files in directory
        
        Args:
            root_dir: Root directory to start search
            
        Yields:
            Path objects for each text file found
        """
        root_path = Path(root_dir)
        for path in root_path.rglob('*'):
            if path.is_file() and path.suffix.lower() in self.file_extensions:
                yield path

    @staticmethod
    def _is_binary_file(file_path: str, block_size: int = 512) -> bool:
        """Check if file is binary by reading first block"""
        with open(file_path, 'rb') as f:
            block = f.read(block_size)
            return b'\x00' in block

    @staticmethod
    def _count_chunk_symbols(args) -> Tuple[Counter, int]:
        """Process a chunk of text and count symbols"""
        chunk, max_length, file_path = args
        try:
            local_counter = Counter()
            chunk_size = len(chunk)
            
            # Count single characters
            local_counter.update(chunk)
            
            # Count repeating characters
            for match in re.finditer(r'(.)\1+', chunk):
                local_counter[match.group(0)] += 1
                
            # Count subwords with DP
            words = re.findall(r'\S+', chunk)
            for word in words:
                n = len(word)
                dp = {}  # Cache for subword counts
                for length in range(2, min(n + 1, max_length + 1)):
                    for i in range(n - length + 1):
                        if i in dp and dp[i][0] >= length:
                            continue
                        subword = word[i:i+length]
                        local_counter[subword] += 1
                        dp[i] = (length, subword)
                        
            return local_counter, chunk_size
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return Counter(), 0

    def _process_file(self, file_path: str, chunk_size: int = 1024*1024) -> Tuple[Counter, int]:
        """Process a single file and return symbol counts and file size"""
        try:
            if self._is_binary_file(file_path):
                logger.warning(f"Skipping binary file: {file_path}")
                return Counter(), 0

            file_size = os.path.getsize(file_path)
            chunks = []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
                    for i in range(0, file_size, chunk_size):
                        mm.seek(i)
                        chunk = mm.read(chunk_size).decode('utf-8', errors='ignore')
                        chunks.append((chunk, self.max_subword_length, file_path))
                    mm.close()
                except Exception as e:
                    logger.warning(f"Memory mapping failed for {file_path}, falling back to normal reading")
                    f.seek(0)
                    content = f.read()
                    chunks = [(content, self.max_subword_length, file_path)]

            with Pool(self.n_workers) as pool:
                results = pool.map(self._count_chunk_symbols, chunks)
                
            total_counter = sum((counter for counter, _ in results), Counter())
            total_size = sum(size for _, size in results)
            
            return total_counter, total_size
            
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {str(e)}")
            return Counter(), 0

    def build_vocabulary_from_directory(self, root_dir: str):
        """
        Build vocabulary from all text files in directory and subdirectories
        
        Args:
            root_dir: Root directory containing text files
        """
        logger.info(f"Scanning directory: {root_dir}")
        files = list(self._find_text_files(root_dir))
        total_files = len(files)
        logger.info(f"Found {total_files} files to process")

        total_counter = Counter()
        total_size = 0
        processed_files = 0
        
        with tqdm(total=total_files, desc="Processing files") as pbar:
            for file_path in files:
                counter, size = self._process_file(str(file_path))
                if size > 0:
                    total_counter.update(counter)
                    total_size += size
                    processed_files += 1
                pbar.update(1)
        
        logger.info(f"Successfully processed {processed_files}/{total_files} files")
        logger.info(f"Processed {total_size/1024/1024:.2f} MB of text")
        
        # Filter and sort symbols by frequency
        filtered_symbols = {
            symbol: count for symbol, count in total_counter.items()
            if count >= self.min_freq
        }
        
        sorted_symbols = sorted(
            filtered_symbols.items(),
            key=lambda x: (-x[1], len(x[0]), x[0])
        )[:self.max_vocab_size - len(self.vocabulary)]
        
        # Update vocabulary and frequencies
        current_idx = len(self.vocabulary)
        for symbol, count in sorted_symbols:
            self.vocabulary[symbol] = current_idx
            self.reverse_vocab[current_idx] = symbol
            self.frequencies[symbol] = count
            current_idx += 1
            
        # Build trie for efficient token matching
        self._build_trie()
            
        logger.info(f"Final vocabulary size: {len(self.vocabulary)}")

    def encode_file(self, file_path: str) -> np.ndarray:
        """
        Encode an entire file
        
        Args:
            file_path: Path to the file to encode
            
        Returns:
            numpy array of indices
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Add file start/end tokens
            content = f"{self.FILE_START_TOKEN}{content}{self.FILE_END_TOKEN}"
            return self.encode(content)
            
        except Exception as e:
            logger.error(f"Error encoding file {file_path}: {str(e)}")
            return np.array([], dtype=np.int32)

    def encode(self, text: str) -> np.ndarray:
        """Encode text to sequence of indices using dynamic programming"""
        tokens = []
        dp_cache = {}  # Cache for dynamic programming
        i = 0
        text_len = len(text)
        
        while i < text_len:
            length, token = self._find_longest_token(text, i, dp_cache)
            tokens.append(self.vocabulary[token])
            i += length
            
        return np.array(tokens, dtype=np.int32)

    def decode(self, indices: np.ndarray) -> str:
        """Decode sequence of indices back to text"""
        text = []
        for idx in indices:
            symbol = self.reverse_vocab.get(idx, self.UNKNOWN_TOKEN)
            if symbol == self.END_OF_LINE_TOKEN:
                text.append('\n')
            elif symbol not in {self.UNKNOWN_TOKEN, self.PAD_TOKEN, 
                              self.FILE_START_TOKEN, self.FILE_END_TOKEN}:
                text.append(symbol)
        return ''.join(text)

    def save_vocabulary(self, file_path: str):
        """Save vocabulary to file"""
        
        # Convert infinity values to a large number for JSON serialization
        serializable_frequencies = {}
        for k, v in self.frequencies.items():
            if v == float('inf'):
                serializable_frequencies[k] = 1e308  # Close to max float value
            else:
                serializable_frequencies[k] = v
                
        data = {
            'vocabulary': self.vocabulary,
            'frequencies': serializable_frequencies,
            'config': {
                'max_vocab_size': self.max_vocab_size,
                'min_freq': self.min_freq,
                'max_subword_length': self.max_subword_length,
                'file_extensions': list(self.file_extensions)  # Convert set to list for JSON
            }
        }
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def load_vocabulary(self, file_path: str):
        """Load vocabulary from file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.vocabulary = data['vocabulary']
        
        # Convert the large number back to infinity for special tokens
        self.frequencies = defaultdict(int)
        for k, v in data['frequencies'].items():
            if v >= 1e308:  # If it's our placeholder for infinity
                self.frequencies[k] = float('inf')
            else:
                self.frequencies[k] = v
                
        self.reverse_vocab = {int(v): k for k, v in self.vocabulary.items()}
        
        config = data['config']
        self.max_vocab_size = config['max_vocab_size']
        self.min_freq = config['min_freq']
        self.max_subword_length = config['max_subword_length']
        self.file_extensions = set(config.get('file_extensions', 
            {'.txt', '.md', '.py', '.java', '.cpp', '.h', '.c', '.js', '.html', '.css'}))
        
        # Rebuild trie after loading vocabulary
        self._build_trie()

if __name__ == "__main__":
    symbolizer = Symbolizer(
        max_vocab_size=50000,
        min_freq=2,
        max_subword_length=20,
        file_extensions={'.txt', '.md', '.py'}
    )
    
    # Build vocabulary from directory
    symbolizer.build_vocabulary_from_directory("./data")
    
    # Save vocabulary
    symbolizer.save_vocabulary("vocabulary.json")
    
    # Later, load vocabulary
    new_symbolizer = Symbolizer()
    new_symbolizer.load_vocabulary("vocabulary.json")
    
    # Test encoding/decoding
    test_text = "Hello, world!"
    encoded = new_symbolizer.encode(test_text)
    decoded = new_symbolizer.decode(encoded)
    print(f"Original: {test_text}")
    print(f"Encoded: ", encoded)
    print(f"Decoded: ", decoded)