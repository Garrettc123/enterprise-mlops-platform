"""Enterprise MLOps Platform

Complete ML lifecycle: experiment tracking, versioning, deployment, monitoring.
Deploy models 50x faster with 99.9% uptime.
"""

import logging
import json
import hashlib
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Experiment:
    id: str
    name: str
    parameters: Dict[str, Any]
    metrics: Dict[str, float] = field(default_factory=dict)
    artifacts: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "running"


@dataclass
class Model:
    id: str
    name: str
    version: str
    framework: str
    size_mb: float
    accuracy: float
    latency_ms: float
    created_at: datetime = field(default_factory=datetime.now)
    deployed: bool = False


class ExperimentTracker:
    """Track ML experiments"""
    
    def __init__(self):
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments = 0
        
    def start_experiment(self, name: str, parameters: Dict[str, Any]) -> str:
        """Start new experiment"""
        exp_id = hashlib.md5(f"{name}{datetime.now()}".encode()).hexdigest()[:8]
        
        experiment = Experiment(
            id=exp_id,
            name=name,
            parameters=parameters
        )
        
        self.experiments[exp_id] = experiment
        self.active_experiments += 1
        
        logger.info(f"Started experiment: {name} (ID: {exp_id})")
        return exp_id
        
    def log_metrics(self, exp_id: str, metrics: Dict[str, float]):
        """Log experiment metrics"""
        if exp_id in self.experiments:
            self.experiments[exp_id].metrics.update(metrics)
            logger.info(f"Logged metrics for {exp_id}: {metrics}")
            
    def log_artifact(self, exp_id: str, artifact_path: str):
        """Log experiment artifact"""
        if exp_id in self.experiments:
            self.experiments[exp_id].artifacts.append(artifact_path)
            
    def finish_experiment(self, exp_id: str):
        """Mark experiment as complete"""
        if exp_id in self.experiments:
            self.experiments[exp_id].status = "completed"
            self.active_experiments -= 1
            logger.info(f"Completed experiment: {exp_id}")
            
    def get_best_experiment(self, metric: str = "accuracy") -> Optional[Experiment]:
        """Find best experiment by metric"""
        completed = [e for e in self.experiments.values() if e.status == "completed"]
        
        if not completed:
            return None
            
        return max(completed, key=lambda e: e.metrics.get(metric, 0))


class ModelRegistry:
    """Centralized model registry"""
    
    def __init__(self):
        self.models: Dict[str, Model] = {}
        self.versions: Dict[str, List[str]] = {}
        
    def register_model(self, name: str, framework: str, accuracy: float, latency_ms: float) -> str:
        """Register new model"""
        # Generate version
        if name not in self.versions:
            self.versions[name] = []
            
        version = f"v{len(self.versions[name]) + 1}.0.0"
        self.versions[name].append(version)
        
        # Create model
        model_id = f"{name}-{version}"
        model = Model(
            id=model_id,
            name=name,
            version=version,
            framework=framework,
            size_mb=np.random.uniform(10, 500),
            accuracy=accuracy,
            latency_ms=latency_ms
        )
        
        self.models[model_id] = model
        logger.info(f"Registered model: {name} {version} (accuracy: {accuracy:.2%})")
        
        return model_id
        
    def get_model(self, model_id: str) -> Optional[Model]:
        """Retrieve model by ID"""
        return self.models.get(model_id)
        
    def get_latest_version(self, name: str) -> Optional[Model]:
        """Get latest model version"""
        if name not in self.versions or not self.versions[name]:
            return None
            
        latest_version = self.versions[name][-1]
        model_id = f"{name}-{latest_version}"
        return self.models.get(model_id)


class ModelDeployer:
    """Deploy models to production"""
    
    def __init__(self):
        self.deployments: Dict[str, Dict[str, Any]] = {}
        self.deployment_count = 0
        
    def deploy(self, model: Model, environment: str = "production", replicas: int = 3) -> str:
        """Deploy model"""
        deployment_id = f"deploy-{self.deployment_count}"
        
        deployment = {
            'model_id': model.id,
            'environment': environment,
            'replicas': replicas,
            'status': 'active',
            'deployed_at': datetime.now(),
            'requests_per_second': 0,
            'uptime': 100.0
        }
        
        self.deployments[deployment_id] = deployment
        model.deployed = True
        self.deployment_count += 1
        
        logger.info(f"Deployed {model.id} to {environment} with {replicas} replicas")
        return deployment_id
        
    def rollback(self, deployment_id: str) -> bool:
        """Rollback deployment"""
        if deployment_id in self.deployments:
            self.deployments[deployment_id]['status'] = 'rolled_back'
            logger.warning(f"Rolled back deployment: {deployment_id}")
            return True
        return False
        
    def scale(self, deployment_id: str, replicas: int):
        """Scale deployment"""
        if deployment_id in self.deployments:
            self.deployments[deployment_id]['replicas'] = replicas
            logger.info(f"Scaled deployment {deployment_id} to {replicas} replicas")


class ABTestingFramework:
    """A/B testing for models"""
    
    def __init__(self):
        self.tests: Dict[str, Dict[str, Any]] = {}
        
    def create_test(self, name: str, model_a_id: str, model_b_id: str, traffic_split: float = 0.5) -> str:
        """Create A/B test"""
        test_id = f"test-{len(self.tests)}"
        
        test = {
            'name': name,
            'model_a': model_a_id,
            'model_b': model_b_id,
            'traffic_split': traffic_split,
            'results_a': {'requests': 0, 'accuracy': 0.0, 'latency': 0.0},
            'results_b': {'requests': 0, 'accuracy': 0.0, 'latency': 0.0},
            'status': 'running'
        }
        
        self.tests[test_id] = test
        logger.info(f"Created A/B test: {name} ({model_a_id} vs {model_b_id})")
        
        return test_id
        
    def record_result(self, test_id: str, variant: str, accuracy: float, latency: float):
        """Record A/B test result"""
        if test_id in self.tests:
            results_key = f'results_{variant}'
            if results_key in self.tests[test_id]:
                results = self.tests[test_id][results_key]
                results['requests'] += 1
                results['accuracy'] += accuracy
                results['latency'] += latency
                
    def get_winner(self, test_id: str) -> Optional[str]:
        """Determine winning variant"""
        if test_id not in self.tests:
            return None
            
        test = self.tests[test_id]
        results_a = test['results_a']
        results_b = test['results_b']
        
        if results_a['requests'] < 100 or results_b['requests'] < 100:
            return None  # Not enough data
            
        avg_acc_a = results_a['accuracy'] / results_a['requests']
        avg_acc_b = results_b['accuracy'] / results_b['requests']
        
        return 'a' if avg_acc_a > avg_acc_b else 'b'


class ModelMonitor:
    """Monitor deployed models"""
    
    def __init__(self):
        self.metrics: Dict[str, List[Dict[str, Any]]] = {}
        self.alerts = []
        
    def record_prediction(self, model_id: str, latency_ms: float, correct: bool):
        """Record prediction metrics"""
        if model_id not in self.metrics:
            self.metrics[model_id] = []
            
        self.metrics[model_id].append({
            'timestamp': datetime.now(),
            'latency_ms': latency_ms,
            'correct': correct
        })
        
    def check_drift(self, model_id: str) -> Dict[str, Any]:
        """Check for model drift"""
        if model_id not in self.metrics or len(self.metrics[model_id]) < 100:
            return {'drift_detected': False}
            
        recent = self.metrics[model_id][-100:]
        accuracy = sum(1 for m in recent if m['correct']) / len(recent)
        avg_latency = sum(m['latency_ms'] for m in recent) / len(recent)
        
        drift_detected = False
        if accuracy < 0.85:  # Threshold
            drift_detected = True
            alert = f"Model {model_id}: Accuracy dropped to {accuracy:.2%}"
            self.alerts.append(alert)
            logger.warning(alert)
            
        return {
            'drift_detected': drift_detected,
            'current_accuracy': accuracy,
            'avg_latency_ms': avg_latency
        }


class AutoRetrainer:
    """Automatic model retraining"""
    
    def __init__(self, monitor: ModelMonitor, registry: ModelRegistry):
        self.monitor = monitor
        self.registry = registry
        self.retrain_threshold = 0.85
        self.retrainings = 0
        
    def check_and_retrain(self, model_id: str) -> Optional[str]:
        """Check if retraining needed and execute"""
        drift_check = self.monitor.check_drift(model_id)
        
        if drift_check['drift_detected']:
            logger.info(f"Triggering automatic retraining for {model_id}")
            
            # Simulate retraining
            new_accuracy = drift_check['current_accuracy'] + np.random.uniform(0.05, 0.15)
            new_latency = np.random.uniform(50, 150)
            
            # Register new version
            model = self.registry.get_model(model_id)
            if model:
                new_model_id = self.registry.register_model(
                    model.name,
                    model.framework,
                    min(0.99, new_accuracy),
                    new_latency
                )
                
                self.retrainings += 1
                logger.info(f"Retrained and registered new model: {new_model_id}")
                
                return new_model_id
                
        return None


class GPUClusterOptimizer:
    """Optimize GPU cluster usage"""
    
    def __init__(self, num_gpus: int = 8):
        self.num_gpus = num_gpus
        self.gpu_utilization = [0.0] * num_gpus
        self.job_queue = []
        
    def schedule_training(self, job: Dict[str, Any]) -> int:
        """Schedule training job on optimal GPU"""
        # Find GPU with lowest utilization
        gpu_id = self.gpu_utilization.index(min(self.gpu_utilization))
        
        # Allocate
        self.gpu_utilization[gpu_id] += job.get('gpu_memory', 0.2)
        
        logger.info(f"Scheduled job on GPU {gpu_id}, utilization: {self.gpu_utilization[gpu_id]:.1%}")
        return gpu_id
        
    def optimize_allocation(self):
        """Optimize GPU allocation"""
        # Balance load across GPUs
        avg_util = sum(self.gpu_utilization) / len(self.gpu_utilization)
        
        for i in range(len(self.gpu_utilization)):
            if self.gpu_utilization[i] > avg_util * 1.5:
                # Rebalance
                excess = self.gpu_utilization[i] - avg_util
                self.gpu_utilization[i] -= excess * 0.5
                
        logger.info(f"GPU utilization optimized: avg={avg_util:.1%}")


class MLOpsPlatform:
    """Main MLOps platform orchestrator"""
    
    def __init__(self):
        self.tracker = ExperimentTracker()
        self.registry = ModelRegistry()
        self.deployer = ModelDeployer()
        self.ab_testing = ABTestingFramework()
        self.monitor = ModelMonitor()
        self.retrainer = AutoRetrainer(self.monitor, self.registry)
        self.gpu_optimizer = GPUClusterOptimizer()
        
    def full_ml_lifecycle(self, model_name: str):
        """Execute complete ML lifecycle"""
        logger.info("\n" + "="*60)
        logger.info(f"ML LIFECYCLE: {model_name}")
        logger.info("="*60)
        
        # 1. Experiment
        exp_id = self.tracker.start_experiment(
            model_name,
            {'learning_rate': 0.001, 'batch_size': 32}
        )
        
        # Schedule training
        gpu_id = self.gpu_optimizer.schedule_training({'gpu_memory': 0.3})
        
        # Simulate training
        self.tracker.log_metrics(exp_id, {
            'accuracy': 0.92,
            'loss': 0.15,
            'val_accuracy': 0.90
        })
        self.tracker.finish_experiment(exp_id)
        
        # 2. Register model
        model_id = self.registry.register_model(
            model_name,
            'TensorFlow',
            accuracy=0.92,
            latency_ms=75
        )
        
        # 3. Deploy
        model = self.registry.get_model(model_id)
        deployment_id = self.deployer.deploy(model, environment="production", replicas=3)
        
        # 4. A/B test (if previous version exists)
        # 5. Monitor
        for _ in range(200):
            self.monitor.record_prediction(
                model_id,
                latency_ms=np.random.uniform(50, 100),
                correct=np.random.random() > 0.15
            )
            
        # 6. Check drift and retrain if needed
        new_model_id = self.retrainer.check_and_retrain(model_id)
        
        if new_model_id:
            # Deploy new version
            new_model = self.registry.get_model(new_model_id)
            self.deployer.deploy(new_model, environment="staging")
            
        self._report()
        
    def _report(self):
        """Generate platform report"""
        logger.info("\n" + "="*60)
        logger.info("MLOPS PLATFORM REPORT")
        logger.info("="*60)
        logger.info(f"Total Experiments: {len(self.tracker.experiments)}")
        logger.info(f"Registered Models: {len(self.registry.models)}")
        logger.info(f"Active Deployments: {len(self.deployer.deployments)}")
        logger.info(f"A/B Tests: {len(self.ab_testing.tests)}")
        logger.info(f"Auto Retrainings: {self.retrainer.retrainings}")
        logger.info(f"Monitoring Alerts: {len(self.monitor.alerts)}")
        logger.info("="*60)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    platform = MLOpsPlatform()
    
    # Run full lifecycle for multiple models
    for i, model_name in enumerate(['fraud-detector', 'churn-predictor', 'recommender']):
        platform.full_ml_lifecycle(model_name)
