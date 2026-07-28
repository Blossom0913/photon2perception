#!/usr/bin/env python3
"""
Experiment Runner for Photon2Perception.

Manages and executes all 36 experiments defined in the manuscript.
Supports queuing, parallel execution across GPUs, and result aggregation.

Experiment groups:
- E01-E10: Main Results (comparisons against baselines)
- E11-E15: Ablation Studies
- E16-E25: Robustness & Generalization
- E26-E31: Efficiency Analysis
- E32-E36: Qualitative Analysis
"""

import os
import sys
import json
import subprocess
import itertools
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field, asdict


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment."""
    id: str                     # Experiment ID (e.g., 'E01')
    name: str                   # Short descriptive name
    section: str                # Paper section (4.2, 4.3, etc.)
    group: str                  # Experiment group (main, ablation, robustness, efficiency, qual)

    # Model overrides
    model_overrides: Dict[str, Any] = field(default_factory=dict)

    # Data overrides
    data_overrides: Dict[str, Any] = field(default_factory=dict)

    # Training overrides
    training_overrides: Dict[str, Any] = field(default_factory=dict)

    # Description
    description: str = ""

    # Dependencies (other experiment IDs that must complete first)
    depends_on: List[str] = field(default_factory=list)


# ----- Experiment Definitions -----

EXPERIMENTS: List[ExperimentConfig] = []

# === Section 4.2: Main Results ===

EXPERIMENTS.extend([
    ExperimentConfig(
        id='E01', name='rgb_baseline_comparison', section='4.2', group='main',
        description='Compare against RGB pipeline baseline',
        model_overrides={'input_format': 'rgb'},
    ),
    ExperimentConfig(
        id='E02', name='demosaic_raw_comparison', section='4.2', group='main',
        description='Compare against demosaic RAW pipeline',
        model_overrides={'input_format': 'demosaic'},
    ),
    ExperimentConfig(
        id='E03', name='learnable_isp_comparison', section='4.2', group='main',
        description='Compare against learnable ISP/adapter methods',
        model_overrides={'use_isp_adapter': True},
    ),
    ExperimentConfig(
        id='E04', name='direct_raw_comparison', section='4.2', group='main',
        description='Compare against direct RAW perception methods',
        model_overrides={},  # Our full method
    ),
    ExperimentConfig(
        id='E05', name='normal_light_results', section='4.2', group='main',
        description='Full results under normal lighting',
        data_overrides={'condition': 'normal'},
    ),
    ExperimentConfig(
        id='E06', name='low_light_results', section='4.2', group='main',
        description='Full results under low light',
        data_overrides={'condition': 'dark'},
    ),
    ExperimentConfig(
        id='E07', name='blur_results', section='4.2', group='main',
        description='Results under motion blur',
        data_overrides={'condition': 'blur'},
    ),
    ExperimentConfig(
        id='E08', name='occlusion_results', section='4.2', group='main',
        description='Results under partial occlusion',
        data_overrides={'condition': 'occlusion'},
    ),
    ExperimentConfig(
        id='E09', name='exposure_cross_camera', section='4.2', group='main',
        description='Exposure variation and cross-camera results',
        data_overrides={'condition': 'all'},
    ),
    ExperimentConfig(
        id='E10', name='pareto_frontier', section='4.2', group='main',
        description='Performance-efficiency Pareto frontier analysis',
        # Run after all other main results
        depends_on=['E01', 'E02', 'E03', 'E04'],
    ),
])

# === Section 4.3: Ablation Studies ===

EXPERIMENTS.extend([
    # E11: Input representation ablation
    ExperimentConfig(
        id='E11a', name='ablation_input_bayer', section='4.3', group='ablation',
        description='Ablation: RAW Bayer tokenization input',
        model_overrides={'input_format': 'bayer'},
    ),
    ExperimentConfig(
        id='E11b', name='ablation_input_demosaic', section='4.3', group='ablation',
        description='Ablation: Demosaic RAW input',
        model_overrides={'input_format': 'demosaic'},
    ),
    ExperimentConfig(
        id='E11c', name='ablation_input_rgb', section='4.3', group='ablation',
        description='Ablation: RGB input',
        model_overrides={'input_format': 'rgb'},
    ),

    # E12: Position encoding ablation
    ExperimentConfig(
        id='E12a', name='ablation_pe_none', section='4.3', group='ablation',
        description='Ablation: No position encoding',
        model_overrides={'use_rope_2d': False, 'use_directional': False},
    ),
    ExperimentConfig(
        id='E12b', name='ablation_pe_absolute', section='4.3', group='ablation',
        description='Ablation: 2D absolute position encoding',
        model_overrides={'use_rope_2d': False, 'use_absolute_pe': True, 'use_directional': False},
    ),
    ExperimentConfig(
        id='E12c', name='ablation_pe_rope2d', section='4.3', group='ablation',
        description='Ablation: 2D RoPE only',
        model_overrides={'use_rope_2d': True, 'use_directional': False},
    ),
    ExperimentConfig(
        id='E12d', name='ablation_pe_rope2d_directional', section='4.3', group='ablation',
        description='Ablation: 2D RoPE + directional enhancement',
        model_overrides={'use_rope_2d': True, 'use_directional': True},
    ),

    # E13: Sparse routing ablation
    ExperimentConfig(
        id='E13a', name='ablation_routing_dense', section='4.3', group='ablation',
        description='Ablation: Dense attention (no pruning)',
        model_overrides={'use_sparse_routing': False},
    ),
    ExperimentConfig(
        id='E13b', name='ablation_routing_heuristic', section='4.3', group='ablation',
        description='Ablation: Heuristic token pruning',
        model_overrides={'use_sparse_routing': True, 'router_type': 'heuristic'},
    ),
    ExperimentConfig(
        id='E13c', name='ablation_routing_saliency', section='4.3', group='ablation',
        description='Ablation: Saliency-aware routing',
        model_overrides={'use_sparse_routing': True, 'router_type': 'saliency'},
    ),
    ExperimentConfig(
        id='E13d', name='ablation_routing_uncertainty', section='4.3', group='ablation',
        description='Ablation: Uncertainty-aware routing',
        model_overrides={'use_sparse_routing': True, 'router_type': 'uncertainty'},
    ),

    # E14: CFA-aware design ablation
    ExperimentConfig(
        id='E14a', name='ablation_cfa_none', section='4.3', group='ablation',
        description='Ablation: No explicit Bayer phase modeling',
        model_overrides={'use_cfa_embed': False},
    ),
    ExperimentConfig(
        id='E14b', name='ablation_cfa_shared', section='4.3', group='ablation',
        description='Ablation: Shared phase embeddings',
        model_overrides={'use_cfa_embed': True, 'cfa_embed_mode': 'shared'},
    ),
    ExperimentConfig(
        id='E14c', name='ablation_cfa_separate', section='4.3', group='ablation',
        description='Ablation: Separate phase-specific encodings',
        model_overrides={'use_cfa_embed': True, 'cfa_embed_mode': 'separate'},
    ),

    # E15: Sensor metadata ablation
    ExperimentConfig(
        id='E15a', name='ablation_meta_none', section='4.3', group='ablation',
        description='Ablation: No sensor metadata',
        model_overrides={'use_metadata': False},
    ),
    ExperimentConfig(
        id='E15b', name='ablation_meta_all', section='4.3', group='ablation',
        description='Ablation: All sensor metadata (ISO, exposure, black level, WB)',
        model_overrides={'use_metadata': True},
    ),
])

# === Section 4.4: Robustness & Generalization ===

CONDITIONS = ['low_light', 'noise', 'blur', 'over_exp', 'weather']
CROSS_TESTS = ['camera', 'iso', 'exposure', 'resolution']

for cond in CONDITIONS:
    EXPERIMENTS.append(ExperimentConfig(
        id=f'E{16 + CONDITIONS.index(cond):02d}',
        name=f'robustness_{cond}',
        section='4.4', group='robustness',
        description=f'Robustness test: {cond}',
        data_overrides={'condition': cond},
    ))

for ct in CROSS_TESTS:
    EXPERIMENTS.append(ExperimentConfig(
        id=f'E{21 + CROSS_TESTS.index(ct):02d}',
        name=f'generalization_{ct}',
        section='4.4', group='robustness',
        description=f'Cross-{ct} generalization test',
        data_overrides={'cross_test': ct},
    ))

EXPERIMENTS.append(ExperimentConfig(
    id='E25', name='ood_stability', section='4.4', group='robustness',
    description='OOD benchmark stability analysis',
    data_overrides={'ood_benchmark': True},
))

# === Section 4.5: Efficiency Analysis ===

EFFICIENCY_EXPS = [
    'latency_profiling', 'sensor_bandwidth', 'memory_bandwidth',
    'flops_analysis', 'sparse_routing_benefit', 'cross_module_breakdown'
]
for i, exp in enumerate(EFFICIENCY_EXPS):
    EXPERIMENTS.append(ExperimentConfig(
        id=f'E{26 + i:02d}',
        name=f'efficiency_{exp}',
        section='4.5', group='efficiency',
        description=f'Efficiency analysis: {exp}',
        model_overrides={'efficiency_test': exp},
    ))

# === Section 4.6: Qualitative Analysis ===

QUALITATIVE_EXPS = [
    'token_routing_viz', 'attention_distribution', 'degradation_focusing',
    'raw_dark_advantage', 'raw_edge_advantage'
]
for i, exp in enumerate(QUALITATIVE_EXPS):
    EXPERIMENTS.append(ExperimentConfig(
        id=f'E{32 + i:02d}',
        name=f'qualitative_{exp}',
        section='4.6', group='qualitative',
        description=f'Qualitative analysis: {exp}',
        model_overrides={'qualitative_viz': exp},
    ))


# ----- Experiment Runner -----

class ExperimentRunner:
    """Manages experiment execution."""

    def __init__(self, config_dir: str = './configs', output_dir: str = './outputs'):
        self.config_dir = Path(config_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_config(self, exp: ExperimentConfig) -> Dict:
        """Generate merged config for an experiment."""
        # Load base config
        base_config_path = self.config_dir / 'detection' / 'photon2percept_det_bayer.yaml'

        import yaml
        with open(base_config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Apply overrides
        config['model'].update(exp.model_overrides)
        config['data'].update(exp.data_overrides)
        config['training'].update(exp.training_overrides)

        # Add experiment metadata
        config['experiment'] = {
            'id': exp.id,
            'name': exp.name,
            'section': exp.section,
            'group': exp.group,
            'description': exp.description,
        }

        return config

    def run_experiment(
        self,
        exp: ExperimentConfig,
        gpu_id: int = 0,
        dry_run: bool = False,
    ) -> Optional[str]:
        """Run a single experiment."""
        config = self.generate_config(exp)

        # Save merged config
        exp_dir = self.output_dir / exp.id
        exp_dir.mkdir(parents=True, exist_ok=True)
        config_path = exp_dir / 'config.yaml'

        import yaml
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        if dry_run:
            print(f"[DRY RUN] {exp.id}: {exp.description}")
            print(f"  Config saved to {config_path}")
            return None

        # Run training
        cmd = [
            sys.executable, '-m', 'tools.train',
            '--config', str(config_path),
            '--output_dir', str(exp_dir),
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

        print(f"[RUN] {exp.id}: {exp.description}")
        result = subprocess.run(cmd, env=env, capture_output=True, text=True)

        # Save logs
        (exp_dir / 'stdout.log').write_text(result.stdout)
        (exp_dir / 'stderr.log').write_text(result.stderr)

        return str(exp_dir)

    def run_all(
        self,
        gpu_ids: List[int] = None,
        dry_run: bool = False,
        groups: Optional[List[str]] = None,
    ):
        """Run all experiments, optionally filtered by group."""
        if gpu_ids is None:
            gpu_ids = [0]

        # Filter experiments
        exps = EXPERIMENTS
        if groups:
            exps = [e for e in exps if e.group in groups]

        # Sort by dependencies (simple topological)
        completed = set()
        results = {}

        # Run experiments
        for i, exp in enumerate(exps):
            gpu_id = gpu_ids[i % len(gpu_ids)]
            result = self.run_experiment(exp, gpu_id=gpu_id, dry_run=dry_run)
            results[exp.id] = result
            completed.add(exp.id)

        return results

    def list_experiments(self, group: Optional[str] = None):
        """List all experiments with descriptions."""
        exps = EXPERIMENTS
        if group:
            exps = [e for e in exps if e.group == group]

        print(f"\n{'='*80}")
        print(f"{'ID':<6} {'Section':<8} {'Group':<14} {'Name':<40}")
        print(f"{'='*80}")
        for exp in exps:
            print(f"{exp.id:<6} {exp.section:<8} {exp.group:<14} {exp.name:<40}")
        print(f"{'='*80}")
        print(f"Total: {len(exps)} experiments\n")

    def export_experiment_matrix(self, output_path: str):
        """Export experiment matrix as CSV."""
        import csv
        with open(output_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=[
                'id', 'name', 'section', 'group', 'description',
                'model_overrides', 'data_overrides'
            ])
            writer.writeheader()
            for exp in EXPERIMENTS:
                row = asdict(exp)
                row['model_overrides'] = json.dumps(row['model_overrides'])
                row['data_overrides'] = json.dumps(row['data_overrides'])
                writer.writerow(row)
        print(f"Experiment matrix exported to {output_path}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Photon2Perception Experiment Runner')
    parser.add_argument('action', choices=['list', 'run', 'export', 'dry-run'],
                        help='Action to perform')
    parser.add_argument('--group', type=str, default=None,
                        help='Filter by experiment group')
    parser.add_argument('--gpus', type=str, default='0',
                        help='Comma-separated GPU IDs')
    parser.add_argument('--exp-id', type=str, default=None,
                        help='Run specific experiment ID')
    parser.add_argument('--output', type=str, default='./outputs',
                        help='Output directory')
    args = parser.parse_args()

    runner = ExperimentRunner(output_dir=args.output)

    if args.action == 'list':
        runner.list_experiments(group=args.group)
    elif args.action == 'export':
        runner.export_experiment_matrix(f'{args.output}/experiment_matrix.csv')
    elif args.action in ('run', 'dry-run'):
        gpu_ids = [int(x) for x in args.gpus.split(',')]
        dry_run = args.action == 'dry-run'

        if args.exp_id:
            exp = next((e for e in EXPERIMENTS if e.id == args.exp_id), None)
            if exp is None:
                print(f"Experiment {args.exp_id} not found")
                sys.exit(1)
            runner.run_experiment(exp, gpu_id=gpu_ids[0], dry_run=dry_run)
        else:
            runner.run_all(gpu_ids=gpu_ids, dry_run=dry_run, groups=[args.group] if args.group else None)
