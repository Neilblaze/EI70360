# Symbolizer

A robust and efficient text tokenization library that builds vocabularies from text data using subword tokenization. The Symbolizer is particularly useful for natural language processing tasks, code analysis, and text preprocessing pipelines.

## Features

- Subword tokenization with configurable parameters
- Parallel processing for improved performance
- Memory-mapped file handling for efficient large file processing
- Support for multiple file types
- Special token handling (PAD, UNK, EOL, etc.)
- Vocabulary persistence (save/load)
- Progress tracking and detailed logging
- Binary file detection and skipping

## Installation

### Prerequisites
- Python 3.6+
- NumPy
- tqdm

### Dependencies
```bash
pip install numpy tqdm
```

## Usage

### Basic Usage

```python
from symbolizer import Symbolizer

# Initialize with default parameters
symbolizer = Symbolizer()

# Or customize the parameters
symbolizer = Symbolizer(
    max_vocab_size=50000,
    min_freq=2,
    max_subword_length=20,
    file_extensions={'.txt', '.md', '.py'}
)

# Build vocabulary from a directory
symbolizer.build_vocabulary_from_directory("./data")

# Save the vocabulary for later use
symbolizer.save_vocabulary("vocabulary.json")

# Load a previously saved vocabulary
symbolizer.load_vocabulary("vocabulary.json")

# Encode text
text = "Hello, world!"
encoded = symbolizer.encode(text)

# Decode back to text
decoded = symbolizer.decode(encoded)
```

### Configuration Parameters

- `max_vocab_size` (default: 50000): Maximum number of symbols in vocabulary
- `min_freq` (default: 2): Minimum frequency for a symbol to be included
- `max_subword_length` (default: 20): Maximum length of subwords to consider
- `n_workers` (default: CPU count - 1): Number of parallel workers
- `file_extensions` (default: {'.txt', '.md', '.py', '.java', '.cpp', '.h', '.c', '.js', '.html', '.css'}): File types to process

### Special Tokens

The Symbolizer includes several special tokens:
- `<UNK>`: Unknown token
- `<EOL>`: End of line
- `<PAD>`: Padding token
- `<FILE_START>`: File start marker
- `<FILE_END>`: File end marker

## How It Works

1. **Vocabulary Building**
   - Recursively scans directories for text files
   - Processes files in parallel using memory mapping
   - Counts symbol frequencies (characters, repeating patterns, subwords)
   - Filters symbols based on minimum frequency
   - Sorts by frequency and length
   - Maintains special tokens

2. **Encoding**
   - Matches longest possible symbols from vocabulary
   - Falls back to unknown token for unrecognized sequences
   - Adds file markers when encoding complete files

3. **Decoding**
   - Converts indices back to symbols
   - Handles special tokens appropriately
   - Reconstructs original text

## Performance Considerations

- Uses memory mapping for efficient file reading
- Implements parallel processing for faster vocabulary building
- Includes binary file detection to skip non-text files
- Caches frequently used operations
- Provides progress bars and logging for long operations

## Error Handling

- Gracefully handles binary files
- Provides detailed logging for debugging
- Falls back to standard file reading if memory mapping fails
- Includes comprehensive error reporting

## Note on Memory Usage

When processing large directories, be aware that:
- Memory mapping helps manage large files efficiently
- Parallel processing can increase memory usage
- Vocabulary size affects memory footprint
- Consider adjusting `max_vocab_size` and `n_workers` based on available system resources