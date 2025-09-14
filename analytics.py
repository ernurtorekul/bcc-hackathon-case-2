import json
import pandas as pd
from typing import Dict, List, Any
from datetime import datetime
import numpy as np

class AnalyticsLayer:
    def __init__(self):
        self.results_history = []

    def log_result(self, module_name: str, input_path: str, output: Any, metrics: Dict[str, float] = None, execution_time: float = None):
        result_entry = {
            'timestamp': datetime.now().isoformat(),
            'module': module_name,
            'input_path': input_path,
            'output': output,
            'metrics': metrics or {},
            'execution_time': execution_time,
            'success': 'error' not in (output if isinstance(output, dict) else {})
        }
        self.results_history.append(result_entry)

    def get_module_performance_summary(self, module_name: str) -> Dict[str, Any]:
        module_results = [r for r in self.results_history if r['module'] == module_name]

        if not module_results:
            return {'error': f'No results found for module {module_name}'}

        successful_results = [r for r in module_results if r['success']]
        success_rate = len(successful_results) / len(module_results)

        all_metrics = {}
        for result in successful_results:
            for metric_name, metric_value in result.get('metrics', {}).items():
                if metric_name not in all_metrics:
                    all_metrics[metric_name] = []
                all_metrics[metric_name].append(metric_value)

        metric_summary = {}
        for metric_name, values in all_metrics.items():
            metric_summary[metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'count': len(values)
            }

        execution_times = [r['execution_time'] for r in successful_results if r['execution_time'] is not None]
        avg_execution_time = np.mean(execution_times) if execution_times else None

        return {
            'module': module_name,
            'total_runs': len(module_results),
            'successful_runs': len(successful_results),
            'success_rate': success_rate,
            'metrics_summary': metric_summary,
            'average_execution_time': avg_execution_time,
            'last_run': module_results[-1]['timestamp'] if module_results else None
        }

    def compare_modules(self, metric_name: str) -> Dict[str, Any]:
        module_metrics = {}

        for result in self.results_history:
            if not result['success'] or metric_name not in result.get('metrics', {}):
                continue

            module = result['module']
            if module not in module_metrics:
                module_metrics[module] = []

            module_metrics[module].append(result['metrics'][metric_name])

        comparison = {}
        for module, values in module_metrics.items():
            comparison[module] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'count': len(values)
            }

        return {
            'metric': metric_name,
            'module_comparison': comparison,
            'best_module': min(comparison.keys(), key=lambda k: comparison[k]['mean']) if comparison else None
        }

    def export_results(self, format_type: str = 'json') -> str:
        if format_type == 'json':
            return json.dumps(self.results_history, indent=2)
        elif format_type == 'csv':
            flattened_results = []
            for result in self.results_history:
                flat_result = {
                    'timestamp': result['timestamp'],
                    'module': result['module'],
                    'input_path': result['input_path'],
                    'success': result['success'],
                    'execution_time': result['execution_time']
                }
                for metric_name, metric_value in result.get('metrics', {}).items():
                    flat_result[f'metric_{metric_name}'] = metric_value

                flattened_results.append(flat_result)

            df = pd.DataFrame(flattened_results)
            return df.to_csv(index=False)
        else:
            raise ValueError(f"Unsupported format: {format_type}")

    def clear_history(self):
        self.results_history = []

    def get_recent_results(self, limit: int = 10) -> List[Dict]:
        return self.results_history[-limit:]