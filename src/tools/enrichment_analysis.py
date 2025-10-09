import scanpy as sc
from .enrichment_analysis.factory_provider import get_factory

def run_enrichment(input_file: str, method: str):
    # load expression

    # get factory
    factory = get_factory(method)

    # run workflow
    analyzer = factory.create_analyzer()
    visualizer = factory.create_visualizer()
    evaluator = factory.create_evaluator()

    result = analyzer.analyze(input_file)
    visualizer.visualize(result)
    eval_res = evaluator.evaluate(result)

    return eval_res
