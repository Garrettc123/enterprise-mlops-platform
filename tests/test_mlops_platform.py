"""Tests for MLOps Platform"""
import pytest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from mlops_platform import (  # noqa: E402
    ExperimentTracker,
    ModelRegistry,
    ABTestingEngine,
    ModelMonitoring,
    GPUClusterManager,
    MLOpsPlatform,
    ModelStatus
)


class TestExperimentTracker:
    """Test ExperimentTracker functionality"""
    
    def test_create_experiment(self):
        """Test creating an experiment"""
        tracker = ExperimentTracker()
        exp = tracker.create_experiment(
            name="Test Experiment",
            model_type="test_model",
            hyperparameters={'lr': 0.001}
        )
        assert exp.name == "Test Experiment"
        assert exp.model_type == "test_model"
        assert tracker.active_experiments == 1
    
    def test_log_metrics(self):
        """Test logging metrics"""
        tracker = ExperimentTracker()
        exp = tracker.create_experiment(
            name="Test",
            model_type="test",
            hyperparameters={}
        )
        tracker.log_metrics(exp.id, {'accuracy': 0.95})
        assert tracker.experiments[exp.id].metrics['accuracy'] == 0.95
    
    def test_get_best_experiment(self):
        """Test getting best experiment"""
        tracker = ExperimentTracker()
        exp1 = tracker.create_experiment("Exp1", "model", {})
        exp2 = tracker.create_experiment("Exp2", "model", {})
        tracker.log_metrics(exp1.id, {'accuracy': 0.85})
        tracker.log_metrics(exp2.id, {'accuracy': 0.92})
        
        best = tracker.get_best_experiment('accuracy')
        assert best.id == exp2.id


class TestModelRegistry:
    """Test ModelRegistry functionality"""
    
    def test_register_model(self):
        """Test registering a model"""
        registry = ModelRegistry()
        model = registry.register_model(
            experiment_id="exp-1",
            accuracy=0.9,
            latency_ms=50
        )
        assert model.version == "v1"
        assert model.status == ModelStatus.VALIDATION
        assert registry.total_versions == 1
    
    def test_promote_to_production(self):
        """Test promoting model to production"""
        registry = ModelRegistry()
        model = registry.register_model("exp-1", 0.9, 50)
        result = registry.promote_to_production(model.id)
        assert result is True
        assert model.status == ModelStatus.DEPLOYED


class TestABTestingEngine:
    """Test A/B testing functionality"""
    
    def test_create_ab_test(self):
        """Test creating A/B test"""
        engine = ABTestingEngine()
        test_id = engine.create_ab_test(
            name="Test A/B",
            model_a="model-1",
            model_b="model-2"
        )
        assert test_id in engine.active_tests
        assert engine.active_tests[test_id]['traffic_split'] == 0.5
    
    def test_record_request(self):
        """Test recording A/B test requests"""
        engine = ABTestingEngine()
        test_id = engine.create_ab_test("Test", "m1", "m2")
        engine.record_request(test_id, 'a', True)
        engine.record_request(test_id, 'b', False)
        
        assert engine.active_tests[test_id]['requests_a'] == 1
        assert engine.active_tests[test_id]['success_a'] == 1
        assert engine.active_tests[test_id]['requests_b'] == 1
        assert engine.active_tests[test_id]['success_b'] == 0
    
    def test_analyze_test(self):
        """Test analyzing A/B test results"""
        engine = ABTestingEngine()
        test_id = engine.create_ab_test("Test", "m1", "m2")
        
        # Model A: 90% success rate
        for i in range(100):
            engine.record_request(test_id, 'a', i < 90)
        
        # Model B: 80% success rate
        for i in range(100):
            engine.record_request(test_id, 'b', i < 80)
        
        result = engine.analyze_test(test_id)
        assert result['winner'] == 'model_a'
        assert result['total_requests'] == 200


class TestModelMonitoring:
    """Test model monitoring functionality"""
    
    def test_record_prediction(self):
        """Test recording predictions"""
        monitor = ModelMonitoring()
        monitor.record_prediction("model-1", 50.0, True)
        assert len(monitor.metrics_history["model-1"]) == 1
    
    def test_get_model_health(self):
        """Test getting model health"""
        monitor = ModelMonitoring()
        # Record 100 predictions with 90% accuracy
        for i in range(100):
            monitor.record_prediction("model-1", 50.0, i < 90)
        
        health = monitor.get_model_health("model-1")
        assert health['accuracy'] == 0.9
        assert health['total_predictions'] == 100


class TestGPUClusterManager:
    """Test GPU cluster management"""
    
    def test_allocate_gpus(self):
        """Test GPU allocation"""
        manager = GPUClusterManager(num_gpus=4)
        gpus = manager.allocate_gpus(2)
        assert gpus is not None
        assert len(gpus) == 2
    
    def test_release_gpus(self):
        """Test GPU release"""
        manager = GPUClusterManager(num_gpus=4)
        gpus = manager.allocate_gpus(2)
        manager.release_gpus(gpus)
        # Check utilization decreased
        stats = manager.get_cluster_stats()
        assert stats['available_gpus'] == 4


@pytest.mark.asyncio
async def test_platform_workflow():
    """Test complete MLOps platform workflow"""
    platform = MLOpsPlatform()
    await platform.run_ml_workflow(num_experiments=3)
    
    # Verify experiments were created
    assert len(platform.experiment_tracker.experiments) == 3
    
    # Verify models were registered
    assert platform.model_registry.total_versions == 3
