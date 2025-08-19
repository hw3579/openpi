"""
Same-round chunked transmission module for Flower federated learning.

Handles large model parameter transmission within a single FL round using 
a request-response pattern.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

class SyncChunkManager:
    """Manages synchronous chunked transmission within a single FL round."""
    
    def __init__(self, chunk_size_limit: int = None):
        """
        Initialize chunk manager.
        
        Args:
            chunk_size_limit: Maximum size per chunk in bytes. Default 1.8GB.
        """
        self.chunk_size_limit = chunk_size_limit or int(18 * 1024 * 1024 * 1024)  # 1.8GB
        
    def needs_chunking(self, parameters: List[np.ndarray]) -> bool:
        """Check if parameters need chunking."""
        total_bytes = sum(arr.nbytes for arr in parameters)
        return total_bytes > self.chunk_size_limit
    
    def split_into_chunks(self, parameters: List[np.ndarray]) -> List[List[np.ndarray]]:
        """
        Split parameters into chunks that fit within size limit.
        
        Args:
            parameters: List of numpy arrays to chunk
            
        Returns:
            List of chunks, where each chunk is a list of parameters
        """
        chunks = []
        current_chunk = []
        current_chunk_bytes = 0
        
        for arr in parameters:
            param_bytes = arr.nbytes
            
            # If adding this parameter would exceed the limit, start a new chunk
            if current_chunk_bytes + param_bytes > self.chunk_size_limit and current_chunk:
                chunks.append(current_chunk)
                current_chunk = [arr]
                current_chunk_bytes = param_bytes
            else:
                current_chunk.append(arr)
                current_chunk_bytes += param_bytes
        
        # Add the last chunk if it has any parameters
        if current_chunk:
            chunks.append(current_chunk)
        
        total_bytes = sum(arr.nbytes for chunk in chunks for arr in chunk)
        print(f"[SyncChunkManager] Split {total_bytes/1024/1024:.1f}MB into {len(chunks)} chunks")
        
        for i, chunk in enumerate(chunks):
            chunk_bytes = sum(arr.nbytes for arr in chunk)
            print(f"  Chunk {i+1}: {chunk_bytes/1024/1024:.1f}MB, {len(chunk)} parameters")
        
        return chunks
    
    def merge_chunks(self, chunks: List[List[np.ndarray]]) -> List[np.ndarray]:
        """
        Merge chunks back into a single parameter list.
        
        Args:
            chunks: List of chunks to merge
            
        Returns:
            Merged parameter list
        """
        merged = []
        for chunk in chunks:
            merged.extend(chunk)
        
        total_bytes = sum(arr.nbytes for arr in merged)
        print(f"[SyncChunkManager] Merged {len(chunks)} chunks into {total_bytes/1024/1024:.1f}MB, {len(merged)} parameters")
        
        return merged


def test_sync_chunking():
    """Test the synchronous chunking functionality."""
    # Create sample parameters
    params = [
        np.random.rand(1000).astype(np.float32),  # Small param
        np.random.rand(500, 1000, 1000).astype(np.float32),  # ~2GB param
        np.random.rand(300, 1000, 1000).astype(np.float32),  # ~1.2GB param
        np.random.rand(400, 1000, 1000).astype(np.float32),  # ~1.6GB param
        np.random.rand(100, 1000, 1000).astype(np.float32),  # ~400MB param
    ]
    
    chunk_manager = SyncChunkManager()
    
    print("Testing synchronous chunking:")
    total_size = sum(p.nbytes for p in params)
    print(f"Total size: {total_size/1024/1024:.1f}MB")
    
    if chunk_manager.needs_chunking(params):
        print("Chunking needed!")
        
        # Split into chunks
        chunks = chunk_manager.split_into_chunks(params)
        
        # Merge back
        merged = chunk_manager.merge_chunks(chunks)
        
        # Verify
        original_bytes = sum(p.nbytes for p in params)
        merged_bytes = sum(p.nbytes for p in merged)
        
        print(f"Verification: Original={original_bytes}, Merged={merged_bytes}, Match={original_bytes == merged_bytes}")
        print(f"Original params: {len(params)}, Merged params: {len(merged)}")
        
    else:
        print("No chunking needed")


if __name__ == "__main__":
    test_sync_chunking()
