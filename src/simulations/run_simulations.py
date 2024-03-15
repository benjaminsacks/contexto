import click

import numpy as np
import pandas as pd
from tqdm import tqdm

DATASET_OPTIONS = ["50", "100", "200", "300"]


def _random_similar_word(word, glove_df, similarity_threshold=200):
    sorted_similarities = np.dot(glove_df, glove_df.loc[word]).argsort()[::-1]
    filtered_glove = glove_df.index[sorted_similarities[1 : similarity_threshold + 1]]
    similar_word = np.random.choice(filtered_glove)
    return similar_word


def _pick_final_word(
    result_vector, starting_word, intermediate_words, glove_df, num_operations
):
    equation_words = [starting_word] + list(intermediate_words)
    sorted_glove_indices = np.dot(glove_df, result_vector).argsort()[::-1]
    closest_words = glove_df.index[sorted_glove_indices[: num_operations + 2]]

    for final_word in closest_words:
        if final_word not in equation_words:
            return final_word
    return "ERROR: WORD NOT FOUND"


def _equation_to_string(starting_word, intermediate_words, final_word, operations):
    equation_string = starting_word

    for i, operation in enumerate(operations):
        equation_string += " + " if operation == 1 else " - "
        equation_string += intermediate_words[i]

    return equation_string + " = " + final_word


def simulate_game(glove_df, num_operations):
    starting_word = np.random.choice(glove_df.index)
    intermediate_words = np.array(
        [
            _random_similar_word(starting_word, glove_df, 100)
            for _ in range(num_operations)
        ]
    )
    operations = np.random.choice([-1, 1], size=num_operations)

    factor_matrix = np.array([glove_df.loc[word] for word in intermediate_words])
    factors_x_operations = np.dot(operations, factor_matrix)

    result_vector = glove_df.loc[starting_word] + factors_x_operations

    final_word = _pick_final_word(
        result_vector, starting_word, intermediate_words, glove_df, num_operations
    )
    similarity = np.dot(glove_df.loc[final_word], result_vector)

    return similarity, _equation_to_string(
        starting_word, intermediate_words, final_word, operations
    )

def batch_simulations(threshold, num_results, output_filepath, glove_df, num_operations):
    simulation_results = []

    with tqdm(total=num_results, desc="Equations found") as pbar:
        while len(simulation_results) < num_results:
            sim = simulate_game(glove_df, num_operations)
            if sim[0] > threshold:
                simulation_results.append(sim)
                pbar.update(1)
    results_df = pd.DataFrame(simulation_results, columns=["similarity", "equation"])

    sorted_results = results_df.sort_values(by="similarity", ascending=False)
    sorted_results.to_csv(output_filepath)

@click.command()
@click.argument("dimensions", type=click.Choice(DATASET_OPTIONS))
@click.argument("output_filepath", type=click.Path())
def main(dimensions, output_filepath):
    embeddings_index = make_dataset.parse_glove_data(
        f".\\data\\raw\\glove.6B.{dimensions}d.txt", True
    )

    words_array = make_dataset.get_words_array(embeddings_index)
    print("len(words_array) (raw):\t\t\t" + "{:,}".format(len(words_array)))
    words_array = make_dataset.filter_alphabetic(words_array)
    print("len(words_array) (filter_alphabetic):\t" + "{:,}".format(len(words_array)))
    words_array = make_dataset.filter_20k(words_array, ".\\data\\external\\20k.txt")
    print("len(words_array) (filter_20k):\t\t" + "{:,}".format(len(words_array)))

    glove_df = make_dataset.embeddings_to_dataframe(embeddings_index, words_array)

    batch_simulations(30, 100, output_filepath, glove_df, 2)

if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.getcwd())
    from src.data import make_dataset
    # python "c:/Users/nadob/Documents/Personal/Personal Projects/contexto/src/simulations/run_simulations.py" 300 .\data\processed\results.csv
    main()
