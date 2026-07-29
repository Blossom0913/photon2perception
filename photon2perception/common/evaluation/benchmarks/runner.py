"""
Standardized benchmark runner for Photon2Perception models.

Runs a trained model through all evaluation metrics and
outputs structured JSON results compatible with experiment tracking.
"""

import json
import torch
from pathlib import Path
from typing import Dict, Optional
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from photon2perception.evaluation.efficiency import full_efficiency_report


class BenchmarkRunner:
    """
    Runs standardized benchmarks on a trained model.

    Args:
        model: Trained model
        config: Experiment configuration
        output_dir: Directory for benchmark results
    """

    def __init__(
        self,
        model: torch.nn.Module,
        config: Dict,
        output_dir: str = './benchmarks',
    ):
        self.model = model
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_efficiency_benchmark(
        self,
        input_sizes: Optional[list] = None,
    ) -> Dict:
        """Run efficiency benchmarks at multiple input sizes."""
        if input_sizes is None:
            input_sizes = [(224, 224), (448, 448), (640, 640)]

        results = {}
        for h, w in input_sizes:
            input_shape = (1, 1, h, w)
            key = f"{h}x{w}"
            results[key] = full_efficiency_report(
                self.model,
                input_shape,
                input_format=self.config.get('input_format', 'bayer'),
            )

        return results

    def run_task_benchmark(
        self,
        dataloader,
        metrics: list,
    ) -> Dict:
        """
        Run task-specific evaluation.

        Args:
            dataloader: Validation dataloader
            metrics: List of metric names to compute ('AP', 'mIoU', etc.)
        Returns:
            Dict with metric results
        """
        # Placeholder — integrate with mmdet/mmseg evaluation
        results = {}
        return results

    def run_all(self, dataloader=None) -> str:
        """Run all benchmarks and save to JSON."""
        report = {
            'config': self.config,
            'efficiency': self.run_efficiency_benchmark(),
            'task_metrics': self.run_task_benchmark(dataloader, ['AP', 'AP50', 'AP75']),
        }

        output_path = self.output_dir / 'benchmark_report.json'
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        print(f"Benchmark report saved to {output_path}")
        return str(output_path)
