from .dataset import DataProcessor
from .train import ModelTrain
from .metrics import MetricsCalculator
from .plots import MetricsGenerator


class Pipeline:

    def run(self):

        dataset = DataProcessor().process()

        trainer = ModelTrain(dataset)

        tree = trainer.train_tree()
        nn = trainer.train_nn()

        trainer.save_models()

        calculator = MetricsCalculator(dataset, tree, nn)

        metrics_tree = calculator.calculate_tree()
        metrics_nn = calculator.calculate_nn()

        generator = MetricsGenerator(
            dataset,
            tree,
            nn,
            metrics_tree,
            metrics_nn
        )

        generator.generate_all()

        return {
            "tree": metrics_tree,
            "nn": metrics_nn
        }