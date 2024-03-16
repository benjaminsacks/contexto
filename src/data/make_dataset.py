# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
import re

from tqdm import tqdm

GLOVE_LENGTH = 400_000

def parse_glove_data(filepath, progress_bar=False):
    """
    Parse GloVe embedding data from a text file.

    Parameters:
    filepath (str): The path to the GloVe data file.

    Returns:
    dict: A dictionary mapping words to their embedding vectors.
    """
    embeddings_index = {}

    with open(filepath, encoding="utf8") as f:
        if not progress_bar:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        else:
            for line in tqdm(f, desc='Reading file', unit='line', total=GLOVE_LENGTH):
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

    return embeddings_index

def get_words_array(embeddings_index):
    """
    Convert keys of embeddings_index dictionary to a numpy array.

    Parameters:
    embeddings_index (dict): A dictionary containing word embeddings.

    Returns:
    numpy.array: Array of words from the embeddings_index keys.
    """
    return np.array(list(embeddings_index.keys()))

def filter_alphabetic(words_array):
    """
    Filter out non-alphabetic words from an array.

    Parameters:
    words_array (numpy.array): Array of words to filter.

    Returns:
    numpy.array: Filtered array containing only alphabetic words.
    """
    pattern = re.compile(r'^[a-z]+$')
    return np.array([x for x in words_array if pattern.match(x)])

def filter_20k(words_array, filepath_20k):
    """
    Filter words array based on the top 20,000 most common words.

    Parameters:
    words_array (numpy.array): Array of words to filter.
    filepath_20k (str): Path to a text file containing the top 20,000 most common English words.

    Returns:
    numpy.array: Array containing words from words_array that are in the top 20,000 list.
    """
    with open(filepath_20k, "r") as file:
        top_20k = file.read().splitlines()
    return np.intersect1d(words_array, top_20k)

def embeddings_to_dataframe(embeddings_index, words_array=None):
    """
    Convert word embeddings dictionary to a pandas DataFrame.

    Parameters:
    embeddings_index (dict): Dictionary containing word embeddings.
    words_array (numpy.array or None): Array of words to filter embeddings. If None, all embeddings are used.

    Returns:
    pandas.DataFrame: DataFrame containing word embeddings with words as index.
    """
    if words_array is None:
        return pd.DataFrame.from_dict(embeddings_index, orient="index")
    else:
        filtered_embeddings = {word: embeddings_index[word] for word in words_array if word in embeddings_index}
        return pd.DataFrame.from_dict(filtered_embeddings, orient="index")



@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info("making final data set from raw data")


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
