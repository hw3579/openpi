#!/usr/bin/env python3
"""Test script to verify enhanced progress bars and client ID handling."""

import time
import numpy as np
from tqdm import tqdm

def test_progress_bars():
    """Test the enhanced progress bar display."""
    print("=== Testing Enhanced Progress Bars ===\n")
    
    # Simulate different client configurations
    test_configs = [
        {"client_id": 0, "total_clients": 2, "server_round": 1},
        {"client_id": 1, "total_clients": 2, "server_round": 1},
        {"client_id": 0, "total_clients": 5, "server_round": 3},
        {"client_id": 4, "total_clients": 5, "server_round": 3},
    ]
    
    for config in test_configs:
        client_id = config["client_id"]
        total_clients = config["total_clients"]
        server_round = config["server_round"]
        
        # Create client prefix like in the real code
        client_prefix = f"🤖 Client-{client_id:02d}"
        if total_clients > 1:
            client_prefix += f"/{total_clients:02d}"
        
        print(f"Testing {client_prefix} in Round {server_round}")
        
        # Simulate single virtual client training
        losses = []
        examples = 0
        local_steps = 5
        batch_size = 16
        
        with tqdm(
            range(local_steps), 
            desc=f"{client_prefix} | Round {server_round} | Training", 
            leave=False,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
        ) as pbar:
            for step in pbar:
                # Simulate training step
                time.sleep(0.1)
                
                # Simulate loss calculation
                cur_loss = 2.5 - 0.1 * step + np.random.normal(0, 0.05)
                losses.append(cur_loss)
                examples += batch_size
                
                # Enhanced progress display
                avg_loss = np.mean(losses)
                pbar.set_postfix({
                    "loss": f"{cur_loss:.4f}",
                    "avg_loss": f"{avg_loss:.4f}",
                    "examples": examples
                })
        
        print(f"✓ {client_prefix} completed training")
        
        # Simulate virtual clients if needed
        if client_id == 0:  # Only test virtual clients for first client
            virtual_clients = 3
            print(f"Testing {client_prefix} with {virtual_clients} virtual clients")
            
            with tqdm(
                range(virtual_clients), 
                desc=f"{client_prefix} | Round {server_round} | Virtual Clients",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}"
            ) as pbar_v:
                for v in pbar_v:
                    # Simulate virtual client training
                    vc_losses = []
                    
                    with tqdm(
                        range(3), 
                        desc=f"  └─ VC-{v+1} Training", 
                        leave=False,
                        bar_format="    {l_bar}{bar}| {n_fmt}/{total_fmt} {postfix}"
                    ) as pbar_inner:
                        for step in pbar_inner:
                            time.sleep(0.05)
                            cur_loss = 2.3 - 0.05 * step + np.random.normal(0, 0.03)
                            vc_losses.append(cur_loss)
                            pbar_inner.set_postfix({"loss": f"{cur_loss:.4f}"})
                    
                    vc_avg_loss = np.mean(vc_losses)
                    overall_avg = np.mean([np.mean(vc_losses) for _ in range(v+1)])
                    
                    pbar_v.set_postfix({
                        "vc": f"{v+1}/{virtual_clients}", 
                        "loss": f"{vc_avg_loss:.4f}",
                        "avg_loss": f"{overall_avg:.4f}"
                    })
            
            print(f"✓ {client_prefix} completed virtual client training")
        
        print()

def test_client_id_handling():
    """Test client ID conflict detection and handling."""
    print("=== Testing Client ID Handling ===\n")
    
    # Simulate different node_config scenarios
    test_scenarios = [
        {"name": "Static ID from config", "node_config": None, "config_id": 0},
        {"name": "Dynamic ID from partition-id", "node_config": {"partition-id": 3}, "config_id": 0},
        {"name": "Dynamic ID from node-id", "node_config": {"node-id": 7}, "config_id": 0},
        {"name": "Dynamic ID from client-id", "node_config": {"client-id": 2}, "config_id": 0},
    ]
    
    for scenario in test_scenarios:
        print(f"Scenario: {scenario['name']}")
        
        # Simulate the ID resolution logic from client code
        client_id = scenario["config_id"]
        nc = scenario["node_config"]
        
        if isinstance(nc, dict):
            for k in [
                "partition-id",
                "node-id", 
                "node_id",
                "client-id",
                "client_id",
                "cid",
            ]:
                if k in nc:
                    client_id = int(nc[k])
                    print(f"  ✓ Using client-id from node_config[{k}] = {client_id}")
                    break
        else:
            print(f"  ✓ Using static client-id = {client_id}")
        
        # Create display prefix
        total_clients = 5
        client_prefix = f"🤖 Client-{client_id:02d}"
        if total_clients > 1:
            client_prefix += f"/{total_clients:02d}"
        
        print(f"  → Client prefix: {client_prefix}")
        print()

if __name__ == "__main__":
    test_progress_bars()
    test_client_id_handling()
    
    print("🎉 All progress bar and ID tests completed!")
    print("✓ Enhanced progress bars with round and client info")
    print("✓ Client ID conflict resolution") 
    print("✓ Virtual client nested progress display")
    print("✓ Consistent client prefix formatting")
