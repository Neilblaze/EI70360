import numpy as np

def learned_positional_encoding(token_embeddings: np.ndarray, position_embedding_table: np.ndarray, start_pos: int = 0) -> np.ndarray:
    """
    Apply learned positional embeddings to token embeddings.
    
    Args:
        token_embeddings: (batch_size, seq_len, d_model) array of token embeddings
        position_embedding_table: (max_seq_len, d_model) learned positional embedding lookup table
        start_pos: Starting position index (default 0)
    
    Returns:
        Array of shape (batch_size, seq_len, d_model) with positional information applied
    """
    batch_size, seq_len, d_model = token_embeddings.shape
    positions = position_embedding_table[start_pos:start_pos + seq_len]
    return token_embeddings + positions
