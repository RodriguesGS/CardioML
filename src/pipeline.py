from .dataset import DataProcessor
from .metrics import MetricsCalculator
from .train import ModelTrain
from .plots import VisualizerGraphs

class Pipeline:

    def __init__(self):
        
        self.dataset = None
        self.train = None
        self.tree = None
        self.nn = None
        self.metrics = None
        
    def prepare_data(self):
        
        self.dataset = DataProcessor().process()
        return self.dataset
    
    def train_models(self):
        
        self.train = ModelTrain(self.dataset)
        models = self.train.train_all()
        self.tree = models['tree']
        self.nn = models['nn']
        self.train.save_models()
        
    def evaluate(self):
        
        calc = MetricsCalculator(self.dataset, self.tree, self.nn)
        self.metrics = {
            'tree': calc.calculate_tree(),
            'nn': calc.calculate_nn(),
        }
        return self.metrics
    
    def generate_graphs(self):
        
        VisualizerGraphs(
            self.dataset, self.tree, self.nn,
            self.metrics['tree'], self.metrics['nn']
        ).generate_all()
        