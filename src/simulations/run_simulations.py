
from tqdm import tqdm
import click

DATASET_OPTIONS = ['50', '100', '200', '300']

@click.command()
@click.option('-d', '--dimensions', type=click.Choice(DATASET_OPTIONS),
              help='Choose the dataset dimensions: 50, 100, 200, or 300.')
@click.argument("output_filepath", type=click.Path())
def main(dimensions):
    """Run simulation with the specified dataset size."""
    if dimensions:
        print(f"Running simulation with dataset size: {dimensions}")
        # Your simulation code goes here
    else:
        print("Please choose a valid dataset size.")

if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.getcwd())
    from src.data import make_dataset
    
    main()