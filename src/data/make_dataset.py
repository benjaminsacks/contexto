# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
import re


def parse_glove_data(filepath):
    """
    Parse GloVe embedding data from a text file.

    Parameters:
    filepath (str): The path to the GloVe data file.

    Returns:
    dict: A dictionary mapping words to their embedding vectors.
    """
    embeddings_index = {}

    with open(filepath, encoding="utf8") as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            embeddings_index[word] = coefs

    return embeddings_index

def get_words_array(embeddings_index):
    return np.array(list(embeddings_index.keys()))

def filter_alphabetic(words_array):
    pattern = re.compile(r'^[a-z]+$')
    return np.array([x for x in words_array if pattern.match(x)])

def filter_20k(words_array, filepath_20k):
    with open(filepath_20k, "r") as file:
        top_20k = file.read().splitlines()
    return np.intersect1d(words_array, top_20k)

def embeddings_to_dataframe(embeddings_index, words_array=None):
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
