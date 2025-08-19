"""
Chunked transmission module for Flower federated learning.

Handles large model parameter transmission by splitting into chunks within size limits.
"""
from __future__ import annotations

import logging
from typing import List, Dict, Optional, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

class ChunkManager:
    """Manages chunked transmission of large parameter arrays."""
    
    def __init__(self, chunk_size_limit: int = None):
        """
        Initialize chunk manager.
        
        Args:
            chunk_size_limit: Maximum size per chunk in bytes. Default 1.8GB.
        """
        self.chunk_size_limit = chunk_size_limit or int(1.8 * 1024 * 1024 * 1024)  # 1.8GB
        
    def calculate_chunks(self, parameters: List[np.ndarray]) -> int:
        """
        Calculate how many chunks are needed for the given parameters.
        
        Args:
            parameters: List of numpy arrays to chunk
            
        Returns:
            Number of chunks needed
        """
        total_bytes = 0
        current_chunk_bytes = 0
        chunks = 1
        
        for i, arr in enumerate(parameters):
            param_bytes = arr.nbytes
            
            if current_chunk_bytes + param_bytes > self.chunk_size_limit and current_chunk_bytes > 0:
                logger.debug(f"Chunk {chunks}: {current_chunk_bytes/1024/1024:.1f}MB (params 0-{i-1})")
                chunks += 1
                current_chunk_bytes = param_bytes
            else:
                current_chunk_bytes += param_bytes
            
            total_bytes += param_bytes
        
        logger.debug(f"Final chunk {chunks}: {current_chunk_bytes/1024/1024:.1f}MB")
        logger.info(f"Chunking calculation: {total_bytes/1024/1024:.1f}MB total, {chunks} chunks needed")
        return chunks
    
    def get_chunk(self, parameters: List[np.ndarray], chunk_id: int) -> Tuple[List[np.ndarray], bool]:
        """
        Extract a specific chunk from parameters.
        
        Args:
            parameters: List of numpy arrays
            chunk_id: Which chunk to extract (0-based)
            
        Returns:
            Tuple of (chunk_parameters, is_final_chunk)
        """
        current_chunk = 0
        current_chunk_bytes = 0
        chunk_start_idx = 0
        
        for i, arr in enumerate(parameters):
            param_bytes = arr.nbytes
            
            if current_chunk_bytes + param_bytes > self.chunk_size_limit and current_chunk_bytes > 0:
                # End of current chunk
                if current_chunk == chunk_id:
                    chunk_arrays = parameters[chunk_start_idx:i]
                    total_chunks = self.calculate_chunks(parameters)
                    is_final = (chunk_id == total_chunks - 1)
                    
                    chunk_bytes = sum(arr.nbytes for arr in chunk_arrays)
                    logger.info(f"Extracted chunk {chunk_id+1}/{total_chunks}: {chunk_bytes/1024/1024:.1f}MB, {len(chunk_arrays)} parameters")
                    
                    return chunk_arrays, is_final
                
                current_chunk += 1
                chunk_start_idx = i
                current_chunk_bytes = param_bytes
            else:
                current_chunk_bytes += param_bytes
        
        # Handle last chunk
        if current_chunk == chunk_id:
            chunk_arrays = parameters[chunk_start_idx:]
            total_chunks = self.calculate_chunks(parameters)
            is_final = (chunk_id == total_chunks - 1)
            
            chunk_bytes = sum(arr.nbytes for arr in chunk_arrays)
            logger.info(f"Extracted final chunk {chunk_id+1}/{total_chunks}: {chunk_bytes/1024/1024:.1f}MB, {len(chunk_arrays)} parameters")
            
            return chunk_arrays, is_final
        
        # Chunk not found
        total_chunks = self.calculate_chunks(parameters)
        raise ValueError(f"Chunk {chunk_id} not found (total chunks: {total_chunks})")
    
    def needs_chunking(self, parameters: List[np.ndarray]) -> bool:
        """
        Check if parameters need chunking.
        
        Args:
            parameters: List of numpy arrays
            
        Returns:
            True if chunking is needed
        """
        total_bytes = sum(arr.nbytes for arr in parameters)
        return total_bytes > self.chunk_size_limit


class ChunkReceiver:
    """Manages receiving and assembling chunks on the server side."""
    
    def __init__(self):
        """Initialize chunk receiver."""
        self.client_chunks: Dict[str, Dict[int, List[np.ndarray]]] = {}
        self.client_chunk_info: Dict[str, Dict[str, Any]] = {}
        self.pending_chunk_requests: Dict[str, int] = {}
    
    def register_chunked_client(self, client_id: str, total_chunks: int, total_size_mb: float) -> None:
        """
        Register a client for chunked transmission.
        
        Args:
            client_id: Unique client identifier
            total_chunks: Total number of chunks expected
            total_size_mb: Total size in megabytes
        """
        print(f"[ChunkReceiver] Registering chunked client {client_id}: {total_chunks} chunks, {total_size_mb:.1f}MB")
        
        self.client_chunk_info[client_id] = {
            "total_chunks": total_chunks,
            "received_chunks": 0,
            "total_size_mb": total_size_mb
        }
        self.client_chunks[client_id] = {}
        self.pending_chunk_requests[client_id] = 0  # Start with chunk 0
        print(f"[ChunkReceiver] Set pending request for client {client_id}: chunk 0")
    
    def receive_chunk(self, client_id: str, chunk_id: int, chunk_parameters: List[np.ndarray], 
                     is_final_chunk: bool) -> Optional[List[np.ndarray]]:
        """
        Receive a chunk from a client.
        
        Args:
            client_id: Unique client identifier
            chunk_id: ID of the received chunk
            chunk_parameters: Parameters in this chunk
            is_final_chunk: Whether this is the final chunk
            
        Returns:
            Complete parameter list if all chunks received, None otherwise
        """
        if client_id not in self.client_chunk_info:
            logger.error(f"Received chunk from unregistered client {client_id}")
            return None
        
        chunk_bytes = sum(arr.nbytes for arr in chunk_parameters)
        total_chunks = self.client_chunk_info[client_id]["total_chunks"]
        
        print(f"[ChunkReceiver] Received chunk {chunk_id+1}/{total_chunks} from client {client_id}: "
              f"{chunk_bytes/1024/1024:.1f}MB, {len(chunk_parameters)} parameters")
        
        # Store chunk
        self.client_chunks[client_id][chunk_id] = chunk_parameters
        self.client_chunk_info[client_id]["received_chunks"] += 1
        
        print(f"[ChunkReceiver] Current state for client {client_id}: "
              f"received {self.client_chunk_info[client_id]['received_chunks']}/{total_chunks} chunks")
        
        if is_final_chunk:
            # Assemble complete parameters
            print(f"[ChunkReceiver] Assembling complete model from {total_chunks} chunks for client {client_id}")
            
            complete_params = []
            for i in range(total_chunks):
                if i in self.client_chunks[client_id]:
                    complete_params.extend(self.client_chunks[client_id][i])
                else:
                    print(f"[ChunkReceiver] ERROR: Missing chunk {i} from client {client_id}")
                    return None
            
            # Clean up
            del self.client_chunks[client_id]
            del self.client_chunk_info[client_id]
            if client_id in self.pending_chunk_requests:
                del self.pending_chunk_requests[client_id]
                print(f"[ChunkReceiver] Cleared pending request for client {client_id}")
            
            total_bytes = sum(arr.nbytes for arr in complete_params)
            print(f"[ChunkReceiver] Assembled complete model for client {client_id}: "
                  f"{total_bytes/1024/1024:.1f}MB, {len(complete_params)} parameters")
            
            return complete_params
        else:
            # Request next chunk
            next_chunk = chunk_id + 1
            self.pending_chunk_requests[client_id] = next_chunk
            print(f"[ChunkReceiver] Will request chunk {next_chunk} from client {client_id} in next round")
            return None
    
    def get_pending_chunk_request(self, client_id: str) -> Optional[int]:
        """
        Get pending chunk request for a client.
        
        Args:
            client_id: Unique client identifier
            
        Returns:
            Chunk ID to request, or None if no pending request
        """
        return self.pending_chunk_requests.get(client_id)
    
    def has_pending_requests(self) -> bool:
        """Check if there are any pending chunk requests."""
        pending = len(self.pending_chunk_requests) > 0
        if pending:
            print(f"[ChunkReceiver] Pending requests: {self.pending_chunk_requests}")
        else:
            print(f"[ChunkReceiver] No pending requests")
        return pending
    
    def clear_client(self, client_id: str) -> None:
        """
        Clear all data for a client (e.g., when switching to normal transmission).
        
        Args:
            client_id: Unique client identifier
        """
        if client_id in self.client_chunks:
            del self.client_chunks[client_id]
        if client_id in self.client_chunk_info:
            del self.client_chunk_info[client_id]
        if client_id in self.pending_chunk_requests:
            del self.pending_chunk_requests[client_id]
        
        logger.info(f"Cleared chunked transmission data for client {client_id}")


def test_chunking():
    """Test the chunking functionality with sample data."""
    logging.basicConfig(level=logging.INFO)
    
    # Create sample parameters of various sizes
    params = [
        np.random.rand(1000).astype(np.float32),  # Small param
        np.random.rand(500, 1000, 1000).astype(np.float32),  # ~2GB param
        np.random.rand(300, 1000, 1000).astype(np.float32),  # ~1.2GB param
        np.random.rand(400, 1000, 1000).astype(np.float32),  # ~1.6GB param
        np.random.rand(100, 1000, 1000).astype(np.float32),  # ~400MB param
    ]
    
    chunk_manager = ChunkManager()
    
    print("Testing chunking functionality:")
    print(f"Total parameters: {len(params)}")
    total_size = sum(p.nbytes for p in params)
    print(f"Total size: {total_size/1024/1024:.1f}MB")
    
    if chunk_manager.needs_chunking(params):
        print("Chunking needed!")
        total_chunks = chunk_manager.calculate_chunks(params)
        
        print(f"\nExtracting {total_chunks} chunks:")
        for chunk_id in range(total_chunks):
            chunk_params, is_final = chunk_manager.get_chunk(params, chunk_id)
            chunk_size = sum(p.nbytes for p in chunk_params)
            print(f"Chunk {chunk_id}: {chunk_size/1024/1024:.1f}MB, {len(chunk_params)} params, final={is_final}")
    else:
        print("No chunking needed")


if __name__ == "__main__":
    test_chunking()
