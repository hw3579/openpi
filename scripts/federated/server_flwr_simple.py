"""
Simplified Flower server for OpenPI training.

Leverages Flower 1.20's automatic large model transmission capabilities.
No manual chunking needed - Flower handles arbitrarily large models automatically.
"""
from __future__ import annotations

import argparse
from typing import List, Optional

import numpy as np
import flwr as fl

import google.protobuf

from chunked_transmission import ChunkReceiver, ChunkManager
from sync_chunked_transmission import SyncChunkManager
from fixed_size_chunked_transmission import FixedSizeChunkManager

print(f"Protobuf version: {google.protobuf.__version__}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--address", type=str, default="0.0.0.0:8080", help="gRPC server address")
    p.add_argument("--rounds", type=int, default=15, help="Number of FL rounds")
    p.add_argument("--min-fit-clients", type=int, default=1, help="Minimum clients for fit")
    p.add_argument("--fraction-fit", type=float, default=1.0, help="Fraction of clients per round")
    p.add_argument("--min-available-clients", type=int, default=1, help="Min available clients to start")
    return p.parse_args()


class SimpleFedAvg(fl.server.strategy.FedAvg):
    """Simple FedAvg strategy with chunked transmission support for large models."""

    def __init__(self, *, min_fit_clients: int, min_available_clients: int, fraction_fit: float):
        super().__init__(
            min_fit_clients=min_fit_clients,
            min_available_clients=min_available_clients,
            fraction_fit=fraction_fit
        )
        self._global_params: Optional[List[np.ndarray]] = None
        # Chunked transmission manager
        self._chunk_receiver = ChunkReceiver()
        self._chunk_manager = ChunkManager()
        self._sync_chunk_manager = SyncChunkManager()
        self._fixed_size_chunk_manager = FixedSizeChunkManager()

    def initialize_parameters(self, client_manager):
        """No initial broadcast - wait for first client upload."""
        return fl.common.ndarrays_to_parameters([])

    def configure_fit(self, server_round: int, parameters, client_manager):
        """Configure fit with current global parameters, handling chunked transmission."""
        print(f"[FL server] Round {server_round}: Configuring fit")
        
        # Sample clients for this round
        selected = client_manager.sample(
            num_clients=self.min_fit_clients,
            min_num_clients=self.min_fit_clients
        )
        
        fit_ins = []
        for client in selected:
            # Normal fit configuration - no chunking in configure_fit
            arrays_to_send = self._global_params if self._global_params else []
            
            if arrays_to_send:
                total_bytes = sum(arr.nbytes for arr in arrays_to_send)
                print(f"[FL server] Round {server_round}: Global model size: {total_bytes/1024/1024:.1f} MB")
                
                if self._chunk_manager.needs_chunking(arrays_to_send):
                    print(f"[FL server] Round {server_round}: Model too large, will use chunked transmission")
                    # For now, send empty params and let client know about chunking in config
                    params = fl.common.ndarrays_to_parameters([])
                    config = {
                        "server_round": server_round,
                        "chunked_download": True,
                        "total_size_mb": total_bytes/1024/1024
                    }
                else:
                    params = fl.common.ndarrays_to_parameters(arrays_to_send)
                    config = {"server_round": server_round}
            else:
                params = fl.common.ndarrays_to_parameters([])
                config = {"server_round": server_round}
            
            fit_ins.append((client, fl.common.FitIns(params, config)))
        
        return fit_ins

    def aggregate_fit(self, server_round: int, results, failures):
        """Aggregate client results using weighted averaging, handling multi-round chunked transmission."""
        print(f"[FL server] Round {server_round}: Received {len(results)} results, {len(failures)} failures")
        
        if failures:
            print(f"[FL server] Failures: {failures}")
        
        if not results:
            print("[FL server] No results to aggregate")
            return None, {}

        try:
            # Process each client result
            complete_clients = []
            
            for client_proxy, fit_res in results:
                client_id = str(client_proxy)
                
                if hasattr(fit_res, 'metrics') and fit_res.metrics:
                    metrics = fit_res.metrics
                    print(f"[FL server] Processing client {client_id}, metrics: {metrics}")
                    
                    if metrics.get("chunked_transmission", False):
                        # Client sent multi-round chunked data
                        print(f"[FL server] Received multi-round chunked transmission from client {client_id}")
                        
                        chunk_id = metrics.get("chunk_id", 0)
                        total_chunks = metrics.get("total_chunks", 1)
                        is_final_chunk = metrics.get("is_final_chunk", True)
                        original_param_count = metrics.get("original_param_count", 0)
                        
                        # Initialize chunk storage for this client if needed
                        if not hasattr(self, '_chunk_storage'):
                            self._chunk_storage = {}
                        
                        if client_id not in self._chunk_storage:
                            self._chunk_storage[client_id] = {
                                'chunks': {},
                                'index_metadata': None,
                                'total_chunks': total_chunks,
                                'original_param_count': original_param_count,
                                'fit_res_template': fit_res  # Keep for examples and other data
                            }
                        
                        # Store the chunk
                        chunk_data = fl.common.parameters_to_ndarrays(fit_res.parameters)[0]  # Single chunk
                        self._chunk_storage[client_id]['chunks'][chunk_id] = chunk_data
                        
                        # Store index metadata if provided (usually with first chunk)
                        if metrics.get("index_metadata"):
                            import json
                            try:
                                index_metadata = json.loads(metrics["index_metadata"])
                                self._chunk_storage[client_id]['index_metadata'] = index_metadata
                                print(f"[FL server] Stored index metadata for client {client_id}")
                            except (json.JSONDecodeError, TypeError) as e:
                                print(f"[FL server] Error parsing index metadata: {e}")
                        
                        print(f"[FL server] Stored chunk {chunk_id + 1}/{total_chunks} for client {client_id} (size: {chunk_data.nbytes / (1024*1024):.1f}MB)")
                        
                        # Check if we have all chunks for this client
                        stored_chunks = len(self._chunk_storage[client_id]['chunks'])
                        expected_chunks = self._chunk_storage[client_id]['total_chunks']
                        
                        if stored_chunks == expected_chunks:
                            print(f"[FL server] All chunks received for client {client_id}, reconstructing model")
                            
                            # Sort chunks by chunk_id and reconstruct
                            chunk_list = []
                            for i in range(expected_chunks):
                                if i in self._chunk_storage[client_id]['chunks']:
                                    chunk_list.append(self._chunk_storage[client_id]['chunks'][i])
                                else:
                                    print(f"[FL server] ERROR: Missing chunk {i} for client {client_id}")
                                    return None, {"error": f"Missing chunk {i}"}
                            
                            # Reconstruct parameters using index metadata
                            index_metadata = self._chunk_storage[client_id]['index_metadata']
                            if index_metadata:
                                reconstructed_params = self._fixed_size_chunk_manager.reassemble_from_chunks(chunk_list, index_metadata)
                                print(f"[FL server] Successfully reconstructed {len(reconstructed_params)} parameters for client {client_id}")
                            else:
                                print(f"[FL server] ERROR: No index metadata found for client {client_id}")
                                return None, {"error": "Missing index metadata"}
                            
                            # Add to complete clients
                            template_fit_res = self._chunk_storage[client_id]['fit_res_template']
                            complete_clients.append((client_proxy, reconstructed_params, template_fit_res.num_examples, template_fit_res.metrics))
                            
                            # Clean up storage for this client
                            del self._chunk_storage[client_id]
                        else:
                            print(f"[FL server] Waiting for more chunks from client {client_id} ({stored_chunks}/{expected_chunks} received)")
                            # Don't process this client yet, wait for more chunks
                            continue
                        
                    else:
                        # Normal client (no chunking)
                        print(f"[FL server] Processing normal client {client_id}")
                        params = fl.common.parameters_to_ndarrays(fit_res.parameters)
                        complete_clients.append((client_proxy, params, fit_res.num_examples, fit_res.metrics))
            
            # Now we can proceed with normal aggregation
            if not complete_clients:
                print("[FL server] No complete client results")
                return None, {}
            
            # Extract weights and parameters for aggregation
            weights = [num_examples for _, _, num_examples, _ in complete_clients]
            params_list = [params for _, params, _, _ in complete_clients]
            
            print(f"[FL server] Aggregating {len(complete_clients)} complete clients")
            print(f"[FL server] Weights: {weights}")
            
            if params_list:
                total_params = sum(arr.size for arr in params_list[0])
                total_bytes = sum(arr.nbytes for arr in params_list[0])
                print(f"[FL server] Received {len(params_list[0])} parameters, {total_params:,} elements, {total_bytes/1024/1024:.1f} MB")
            
            if not params_list or not weights:
                print("[FL server] Empty params or weights")
                return None, {}

            # Weighted average
            total_weight = sum(weights)
            if total_weight == 0:
                print("[FL server] Total weight is zero")
                return None, {}

            # Initialize aggregated parameters
            num_params = len(params_list[0])
            aggregated = []
            
            for param_idx in range(num_params):
                # Weighted sum
                weighted_sum = None
                for client_idx, params in enumerate(params_list):
                    if param_idx < len(params):
                        weight = weights[client_idx] / total_weight
                        contribution = np.asarray(params[param_idx]) * weight
                        if weighted_sum is None:
                            weighted_sum = contribution
                        else:
                            weighted_sum = weighted_sum + contribution
                
                if weighted_sum is not None:
                    aggregated.append(weighted_sum)

            # Store global parameters for next round
            self._global_params = aggregated
            print(f"[FL server] Updated global model with {len(aggregated)} parameters")
            
            # Collect metrics
            metrics = {}
            losses = []
            for _, _, _, client_metrics in complete_clients:
                if client_metrics and 'loss' in client_metrics:
                    losses.append(client_metrics['loss'])
            
            if losses:
                metrics['avg_loss'] = float(np.mean(losses))
                print(f"[FL server] Round {server_round}: Average loss = {metrics['avg_loss']:.4f}")

            return None, metrics
            
        except Exception as e:
            print(f"[FL server] Error in aggregate_fit: {type(e).__name__}: {e}")
            import traceback
            traceback.print_exc()
            return None, {"error": str(e)}


def main():
    args = parse_args()

    strategy = SimpleFedAvg(
        min_fit_clients=args.min_fit_clients,
        min_available_clients=args.min_available_clients,
        fraction_fit=args.fraction_fit,
    )

    print(f"Starting Flower server on {args.address}")
    print(f"Rounds: {args.rounds}")
    print("Using Flower 1.20's automatic large model transmission")

    fl.server.start_server(
        server_address=args.address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
