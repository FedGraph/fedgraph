from .dp_mechanisms import DPMechanism, DPAccountant
from .server_dp import Server_DP
from .trainer_dp import Trainer_General_DP

__version__ = "1.0.0"
__author__ = "FedGraph Team"

__all__ = [
    "DPMechanism",
    "DPAccountant", 
    "Server_DP",
    "Trainer_General_DP",
]

# Module-level configuration
DEFAULT_DP_CONFIG = {
    "epsilon": 1.0,
    "delta": 1e-5,
    "mechanism": "gaussian",
    "sensitivity": 1.0,
    "clip_norm": 1.0,
}

def get_default_config():
    """Get default DP configuration."""
    return DEFAULT_DP_CONFIG.copy()

def validate_dp_config(config):
    """Validate DP configuration parameters."""
    required_keys = ["epsilon", "delta", "mechanism"]
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required DP parameter: {key}")
    
    if config["epsilon"] <= 0:
        raise ValueError("epsilon must be positive")
    if config["delta"] <= 0 or config["delta"] >= 1:
        raise ValueError("delta must be in (0, 1)")
    
    valid_mechanisms = ["gaussian", "laplace", "local"]
    if config["mechanism"] not in valid_mechanisms:
        raise ValueError(f"mechanism must be one of {valid_mechanisms}")
    
    return True

print(f"FedGraph Differential Privacy module loaded (v{__version__})")