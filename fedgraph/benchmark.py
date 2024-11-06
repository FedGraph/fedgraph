import time
import sys
import json
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Any
import numpy as np

@dataclass
class BenchmarkMetrics:
    # Encryption context metrics
    poly_modulus_degree: int
    coeff_mod_bit_sizes: List[int]
    global_scale: float
    
    # Pre-training metrics (feature aggregation)
    pretrain_encryption_time: float = 0.0
    pretrain_decryption_time: float = 0.0
    pretrain_aggregation_time: float = 0.0
    pretrain_upload_size: float = 0.0  # MB
    pretrain_download_size: float = 0.0  # MB
    
    # Training metrics (parameter aggregation)
    param_encryption_time: float = 0.0
    param_decryption_time: float = 0.0
    param_aggregation_time: float = 0.0
    param_upload_size: float = 0.0  # MB
    param_download_size: float = 0.0  # MB
    
    # Model performance metrics
    test_accuracy: float = 0.0
    test_loss: float = 0.0
    total_training_time: float = 0.0
    memory_usage: float = 0.0  # MB

class FedGraphBenchmark:
    def __init__(self, context, n_trainers):
        self.metrics = BenchmarkMetrics(
            poly_modulus_degree=context.poly_modulus_degree(),
            coeff_mod_bit_sizes=[int(x) for x in context.coeff_mod_bit_sizes()],
            global_scale=float(context.global_scale())
        )
        self.n_trainers = n_trainers
        self.start_time = time.time()
        
    def update_pretrain_metrics(self, 
                              enc_times: List[float],
                              dec_times: List[float],
                              agg_time: float,
                              upload_sizes: List[float],
                              download_size: float):
        self.metrics.pretrain_encryption_time = np.mean(enc_times)
        self.metrics.pretrain_decryption_time = np.mean(dec_times)
        self.metrics.pretrain_aggregation_time = agg_time
        self.metrics.pretrain_upload_size = sum(upload_sizes) / (1024 * 1024)  # Convert to MB
        self.metrics.pretrain_download_size = download_size * self.n_trainers / (1024 * 1024)
        
    def update_training_metrics(self,
                              enc_times: List[float],
                              dec_times: List[float],
                              agg_time: float,
                              upload_sizes: List[float],
                              download_size: float):
        self.metrics.param_encryption_time = np.mean(enc_times)
        self.metrics.param_decryption_time = np.mean(dec_times)
        self.metrics.param_aggregation_time = agg_time
        self.metrics.param_upload_size = sum(upload_sizes) / (1024 * 1024)
        self.metrics.param_download_size = download_size * self.n_trainers / (1024 * 1024)
        
    def update_performance_metrics(self, test_accuracy: float, test_loss: float):
        self.metrics.test_accuracy = test_accuracy
        self.metrics.test_loss = test_loss
        self.metrics.total_training_time = time.time() - self.start_time
        self.metrics.memory_usage = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        
    def save_results(self, filename: str = "benchmark_results.json"):
        results = {
            "encryption_context": {
                "poly_modulus_degree": self.metrics.poly_modulus_degree,
                "coeff_mod_bit_sizes": self.metrics.coeff_mod_bit_sizes,
                "global_scale": self.metrics.global_scale
            },
            "pretrain_metrics": {
                "encryption_time": self.metrics.pretrain_encryption_time,
                "decryption_time": self.metrics.pretrain_decryption_time,
                "aggregation_time": self.metrics.pretrain_aggregation_time,
                "upload_size_mb": self.metrics.pretrain_upload_size,
                "download_size_mb": self.metrics.pretrain_download_size
            },
            "training_metrics": {
                "encryption_time": self.metrics.param_encryption_time,
                "decryption_time": self.metrics.param_decryption_time,
                "aggregation_time": self.metrics.param_aggregation_time,
                "upload_size_mb": self.metrics.param_upload_size,
                "download_size_mb": self.metrics.param_download_size
            },
            "performance_metrics": {
                "test_accuracy": self.metrics.test_accuracy,
                "test_loss": self.metrics.test_loss,
                "total_training_time": self.metrics.total_training_time,
                "memory_usage_mb": self.metrics.memory_usage
            }
        }
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=4)
            
    def generate_latex_tables(self):
        # Context table
        context_df = pd.DataFrame({
            'Parameter': ['Polynomial Modulus Degree', 'Coefficient Modulus Bit Sizes', 'Global Scale'],
            'Value': [
                self.metrics.poly_modulus_degree,
                str(self.metrics.coeff_mod_bit_sizes),
                f"2^{np.log2(self.metrics.global_scale):.0f}"
            ]
        })
        
        # Computation costs table
        computation_df = pd.DataFrame({
            'Phase': ['Pre-training', 'Training'],
            'Encryption Time (s)': [
                f"{self.metrics.pretrain_encryption_time:.4f}",
                f"{self.metrics.param_encryption_time:.4f}"
            ],
            'Decryption Time (s)': [
                f"{self.metrics.pretrain_decryption_time:.4f}",
                f"{self.metrics.param_decryption_time:.4f}"
            ],
            'Aggregation Time (s)': [
                f"{self.metrics.pretrain_aggregation_time:.4f}",
                f"{self.metrics.param_aggregation_time:.4f}"
            ]
        })
        
        # Communication costs table
        communication_df = pd.DataFrame({
            'Phase': ['Pre-training', 'Training'],
            'Upload Size (MB)': [
                f"{self.metrics.pretrain_upload_size:.2f}",
                f"{self.metrics.param_upload_size:.2f}"
            ],
            'Download Size (MB)': [
                f"{self.metrics.pretrain_download_size:.2f}",
                f"{self.metrics.param_download_size:.2f}"
            ],
            'Total Size (MB)': [
                f"{self.metrics.pretrain_upload_size + self.metrics.pretrain_download_size:.2f}",
                f"{self.metrics.param_upload_size + self.metrics.param_download_size:.2f}"
            ]
        })
        
        # Performance metrics table
        performance_df = pd.DataFrame({
            'Metric': ['Test Accuracy', 'Test Loss', 'Total Training Time (s)', 'Memory Usage (MB)'],
            'Value': [
                f"{self.metrics.test_accuracy:.4f}",
                f"{self.metrics.test_loss:.4f}",
                f"{self.metrics.total_training_time:.2f}",
                f"{self.metrics.memory_usage:.2f}"
            ]
        })
        
        return {
            'context': context_df.to_latex(index=False),
            'computation': computation_df.to_latex(index=False),
            'communication': communication_df.to_latex(index=False),
            'performance': performance_df.to_latex(index=False)
        }