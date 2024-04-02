from pathlib import Path
import sys, os
project_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(project_dir))

from src.data import make_dataset
from config import *

import numpy as np
import pandas as pd
from tqdm import tqdm

# TODO : Add type hints to functions
# TODO : Add docstrings to functions
# TODO : Turn glove_df into a class?


def _euclidean_distance(a, b):
    return np.sqrt(np.sum(np.square(a - b)))


def _get_distances_series(glove_df, vector):
    # TODO : Combine with _euclidean_distance using series=TRUE argument
    return np.sqrt(np.sum(np.square(glove_df - vector), axis=1))


def _filter_similar_words(word, glove_df, similarity_threshold=SIMILARITY_THRESHOLD):
    distances = _get_distances_series(glove_df, glove_df.loc[word])
    sorted_distances = distances.sort_values()
    smallest_distances = sorted_distances.iloc[1 : similarity_threshold + 1]
    return glove_df.loc[smallest_distances.index]


def _get_result_vector(starting_word, intermediate_words, operations, glove_df):
    intermediate_embeddings_matrix = np.array(
        [glove_df.loc[word] for word in intermediate_words]
    )
    embeddings_dot_operations = np.dot(operations, intermediate_embeddings_matrix)
    return glove_df.loc[starting_word] + embeddings_dot_operations


def _pick_final_word(result_vector, used_words, glove_df):
    # TODO : Simplify function, lessen arguments, pick from filtered group of words
    sorted_distances = _get_distances_series(glove_df, result_vector).sort_values()
    sorted_words = sorted_distances.index
    closest_words = sorted_words[: len(used_words) + 1]

    for final_word in closest_words:
        if final_word not in used_words:
            return final_word


def _equation_to_string(starting_word, intermediate_words, final_word, operations):
    equation_string = starting_word

    for i, operation in enumerate(operations):
        equation_string += " + " if operation == 1 else " - "
        equation_string += intermediate_words[i]

    return equation_string + " = " + final_word


def simulate_game(glove_df, num_operations):
    # TODO : Simplify function with more subfunctions
    starting_word = np.random.choice(glove_df.index)

    filtered_similar_df = _filter_similar_words(starting_word, glove_df)
    intermediate_words = np.random.choice(
        filtered_similar_df.index, num_operations, replace=False
    )
    operations = np.random.choice([-1, 1], size=num_operations)

    result_vector = _get_result_vector(
        starting_word, intermediate_words, operations, glove_df
    )

    used_words = [starting_word] + intermediate_words.tolist()
    final_word = _pick_final_word(result_vector, used_words, filtered_similar_df)
    final_distance = _euclidean_distance(glove_df.loc[final_word], result_vector)

    return final_distance, _equation_to_string(
        starting_word, intermediate_words, final_word, operations
    )


def batch_simulations(
    glove_df,
    num_operations,
    output_filepath,
    distance_threshold=DISTANCE_THRESHOLD,
    num_results=NUM_RESULTS,
):
    simulation_results = []

    with tqdm(total=num_results, desc="Equations found") as pbar:
        while len(simulation_results) < num_results:
            sim = simulate_game(glove_df, num_operations)
            if sim[0] < distance_threshold:
                simulation_results.append(sim)
                pbar.update(1)
    results_df = pd.DataFrame(simulation_results, columns=["distance", "equation"])

    sorted_results = results_df.sort_values(by="distance", ascending=True)
    sorted_results.to_csv(output_filepath)


def main():
    if not os.path.exists(FILTERED_FILEPATH):
        make_dataset.main()
    glove_df = pd.read_pickle(FILTERED_FILEPATH)

    batch_simulations(glove_df, 2, OUTPUT_FILEPATH)


if __name__ == "__main__":
    main()
