#!/usr/bin/env python3
"""Test script to verify server-side weight initialization."""

import sys
import pathlib

# Add OpenPI to path
_ROOT = pathlib.Path(__file__).resolve().parents[2]
_SRC = _ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Test imports
try:
    from server_flwr_cli import OpenPIFedAvg
    print("✓ Successfully imported OpenPIFedAvg")
except ImportError as e:
    print(f"✗ Failed to import OpenPIFedAvg: {e}")
    sys.exit(1)

# Test weight initialization
def test_weight_initialization():
    print("\n=== Testing Server Weight Initialization ===")
    
    try:
        # Create strategy with test config
        strategy = OpenPIFedAvg(
            config_name="pi0_libero_0813_fl",
            min_fit_clients=1,
            min_available_clients=1,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
        )
        print("✓ Created OpenPIFedAvg strategy")
        
        # Test weight initialization
        from unittest.mock import MagicMock
        mock_client_manager = MagicMock()
        
        print("Initializing parameters...")
        params = strategy.initialize_parameters(mock_client_manager)
        print("✓ Parameters initialized successfully")
        
        if params and hasattr(params, 'tensors') and params.tensors:
            print(f"✓ Loaded {len(params.tensors)} parameter tensors")
            
            # Print some stats
            total_params = sum(t.size for t in params.tensors)
            total_bytes = sum(t.nbytes for t in params.tensors)
            print(f"✓ Total parameters: {total_params:,}")
            print(f"✓ Total size: {total_bytes/1024/1024:.1f} MB")
        else:
            print("✗ No parameters loaded")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ Error during weight initialization: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_weight_initialization()
    if success:
        print("\n🎉 All tests passed!")
        sys.exit(0)
    else:
        print("\n❌ Tests failed!")
        sys.exit(1)
