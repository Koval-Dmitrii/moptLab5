from internal.tasks import (
    baseline_models,
    batch_size_study,
    regularization_study,
    methods_comparison,
)
from internal.runner import run


def main():
    run([
        baseline_models,
        batch_size_study,
        regularization_study,
        methods_comparison,
    ], output_root="results", save_graphs=True, save_tables=True)


if __name__ == "__main__":
    main()
