"""
Fixed-size chunked transmission module for Flower federated learning.

Handles large model parameter transmission by splitting into fixed-size chunks
with an index file for reassembly.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import io

logger = logging.getLogger(__name__)

class FixedSizeChunkManager:
    """Manages fixed-size chunked transmission with index metadata."""
    
    def __init__(self, chunk_size_mb: int = 1800):
        """
        Initialize fixed-size chunk manager.
        
        Args:
            chunk_size_mb: Maximum size per chunk in MB. Default 1800MB for efficiency.
        """
        self.chunk_size_bytes = chunk_size_mb * 1024 * 1024
        print(f"[FixedSizeChunkManager] Initialized with chunk size: {chunk_size_mb}MB")  
        
    def needs_chunking(self, parameters: List[np.ndarray]) -> bool:
        """Check if parameters need chunking."""
        total_bytes = sum(arr.nbytes for arr in parameters)
        return total_bytes > self.chunk_size_bytes
    
    def create_chunks_with_index(self, parameters: List[np.ndarray]) -> Tuple[List[np.ndarray], Dict]:
        """
        Split parameters into fixed-size chunks with an index for reassembly.
        
        Args:
            parameters: List of numpy arrays to chunk
            
        Returns:
            Tuple of (chunk_list, index_metadata)
        """
        # Create index metadata
        index = {
            "total_parameters": len(parameters),
            "parameter_info": [],
            "chunk_info": [],
            "total_size_bytes": 0
        }
        
        # Record original parameter information
        for i, arr in enumerate(parameters):
            index["parameter_info"].append({
                "param_id": i,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "size_bytes": int(arr.nbytes)
            })
            index["total_size_bytes"] += arr.nbytes
        
        # Serialize all parameters into a single byte stream
        buffer = io.BytesIO()
        param_offsets = []
        
        for arr in parameters:
            offset = buffer.tell()
            param_offsets.append(offset)
            np.save(buffer, arr)
        
        # Get the complete byte stream
        buffer.seek(0)
        data_bytes = buffer.getvalue()
        total_bytes = len(data_bytes)
        
        print(f"[FixedSizeChunkManager] Serialized {len(parameters)} parameters into {total_bytes/1024/1024:.1f}MB byte stream")
        
        # Split into fixed-size chunks
        chunks = []
        chunk_start = 0
        chunk_id = 0
        
        while chunk_start < total_bytes:
            chunk_end = min(chunk_start + self.chunk_size_bytes, total_bytes)
            chunk_data = data_bytes[chunk_start:chunk_end]
            
            # Convert chunk to numpy array for Flower compatibility
            chunk_array = np.frombuffer(chunk_data, dtype=np.uint8)
            chunks.append(chunk_array)
            
            chunk_info = {
                "chunk_id": chunk_id,
                "start_byte": chunk_start,
                "end_byte": chunk_end,
                "size_bytes": len(chunk_data),
                "size_mb": len(chunk_data) / 1024 / 1024
            }
            index["chunk_info"].append(chunk_info)
            
            print(f"  Chunk {chunk_id}: {chunk_info['size_mb']:.1f}MB (bytes {chunk_start}-{chunk_end})")
            
            chunk_start = chunk_end
            chunk_id += 1
        
        print(f"[FixedSizeChunkManager] Created {len(chunks)} fixed-size chunks")
        
        return chunks, index
    
    def reassemble_from_chunks(self, chunks: List[np.ndarray], index: Dict) -> List[np.ndarray]:
        """
        Reassemble parameters from fixed-size chunks using index metadata.
        
        Args:
            chunks: List of chunk arrays
            index: Index metadata
            
        Returns:
            Reassembled parameter list
        """
        print(f"[FixedSizeChunkManager] Reassembling {len(chunks)} chunks into {index['total_parameters']} parameters")
        
        # Validate chunks
        if len(chunks) != len(index["chunk_info"]):
            raise ValueError(f"Chunk count mismatch: got {len(chunks)}, expected {len(index['chunk_info'])}")
        
        # Reconstruct byte stream
        data_bytes = b""
        for i, chunk in enumerate(chunks):
            chunk_info = index["chunk_info"][i]
            expected_size = chunk_info["size_bytes"]
            
            # Convert chunk back to bytes
            chunk_bytes = chunk.tobytes()
            
            if len(chunk_bytes) != expected_size:
                print(f"Warning: Chunk {i} size mismatch: got {len(chunk_bytes)}, expected {expected_size}")
            
            data_bytes += chunk_bytes
        
        print(f"[FixedSizeChunkManager] Reconstructed {len(data_bytes)/1024/1024:.1f}MB byte stream")
        
        # Deserialize parameters
        buffer = io.BytesIO(data_bytes)
        parameters = []
        
        for param_info in index["parameter_info"]:
            try:
                arr = np.load(buffer)
                
                # Validate shape and dtype
                expected_shape = tuple(param_info["shape"])
                expected_dtype = param_info["dtype"]
                
                if arr.shape != expected_shape:
                    print(f"Warning: Parameter {param_info['param_id']} shape mismatch: "
                          f"got {arr.shape}, expected {expected_shape}")
                
                if str(arr.dtype) != expected_dtype:
                    print(f"Warning: Parameter {param_info['param_id']} dtype mismatch: "
                          f"got {arr.dtype}, expected {expected_dtype}")
                
                parameters.append(arr)
                
            except Exception as e:
                raise ValueError(f"Failed to deserialize parameter {param_info['param_id']}: {e}")
        
        # Validate total
        reassembled_bytes = sum(arr.nbytes for arr in parameters)
        expected_bytes = index["total_size_bytes"]
        
        print(f"[FixedSizeChunkManager] Reassembled {len(parameters)} parameters, "
              f"total size: {reassembled_bytes/1024/1024:.1f}MB")
        
        if reassembled_bytes != expected_bytes:
            print(f"Warning: Total size mismatch: got {reassembled_bytes}, expected {expected_bytes}")
        
        return parameters


def test_fixed_size_chunking():
    """Test the fixed-size chunking functionality."""
    # Create sample parameters
    params = [
        np.random.rand(1000).astype(np.float32),  # Small param
        np.random.rand(500, 1000, 1000).astype(np.float32),  # ~2GB param
        np.random.rand(300, 1000, 1000).astype(np.float32),  # ~1.2GB param
        np.random.rand(100, 1000, 1000).astype(np.float32),  # ~400MB param
    ]
    
    chunk_manager = FixedSizeChunkManager()
    
    print("Testing fixed-size chunking:")
    total_size = sum(p.nbytes for p in params)
    print(f"Total size: {total_size/1024/1024:.1f}MB")
    
    if chunk_manager.needs_chunking(params):
        print("Chunking needed!")
        
        # Create chunks with index
        chunks, index = chunk_manager.create_chunks_with_index(params)
        
        print(f"\nIndex metadata:")
        print(f"  Total parameters: {index['total_parameters']}")
        print(f"  Total chunks: {len(index['chunk_info'])}")
        print(f"  Total size: {index['total_size_bytes']/1024/1024:.1f}MB")
        
        # Reassemble
        reassembled = chunk_manager.reassemble_from_chunks(chunks, index)
        
        # Verify
        print(f"\nVerification:")
        print(f"  Original params: {len(params)}")
        print(f"  Reassembled params: {len(reassembled)}")
        
        # Check each parameter
        all_match = True
        for i, (orig, reasm) in enumerate(zip(params, reassembled)):
            if not np.array_equal(orig, reasm):
                print(f"  Parameter {i}: MISMATCH")
                all_match = False
            else:
                print(f"  Parameter {i}: OK")
        
        print(f"  Overall verification: {'PASS' if all_match else 'FAIL'}")
        
    else:
        print("No chunking needed")


if __name__ == "__main__":
    test_fixed_size_chunking()
