# -*- coding: utf-8 -*-
import click
import logging
from tqdm import tqdm
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import numpy as np
import pandas as pd
import re

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer

GLOVE_LENGTH = 400_000

#TODO : Add word list option to not read whole file each time
def parse_glove_data(filepath, progress_bar=False):
    """
    Parse GloVe embedding data from a text file.

    Parameters:
    - filepath (str): The path to the GloVe data file.

    Returns:
    - dict: A dictionary mapping words to their embedding vectors.
    """
    embeddings_index = {}

    with open(filepath, encoding="utf8") as f:
        if not progress_bar:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs
        else:
            for line in tqdm(f, desc="Reading file", unit="line", total=GLOVE_LENGTH):
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

    return embeddings_index


def get_word_list(embeddings_index):
    return list(embeddings_index.keys())


def filter_alphabetic(word_list):
    pattern = re.compile(r"^[a-z]+$")
    return [x for x in word_list if pattern.match(x)]


def filter_20k(word_list, filepath_20k):
    with open(filepath_20k, "r") as file:
        top_20k = file.read().splitlines()
    return np.intersect1d(word_list, top_20k)

def filter_wordnet(word_list):
    return [w for w in word_list if len(wordnet.synsets(w)) > 0]

def lemmatize_words(word_list):
    wnl = WordNetLemmatizer()
    lemmatized_set = set()

    for word in word_list:
        word_lemma_list = [wnl.lemmatize(word, pos=p) for p in "nvars"]
        lemmatized_set.add(min(word_lemma_list, key=len))

    return list(lemmatized_set)


def embeddings_to_dataframe(embeddings_index, words_array=None):
    if words_array is None:
        return pd.DataFrame.from_dict(embeddings_index, orient="index")
    else:
        filtered_embeddings = {
            word: embeddings_index[word]
            for word in words_array
            if word in embeddings_index
        }
        return pd.DataFrame.from_dict(filtered_embeddings, orient="index")

# TODO : Fix commented section. Either remove or implement
# @click.command()
# @click.argument("input_filepath", type=click.Path(exists=True))
# @click.argument("output_filepath", type=click.Path())
# def main(input_filepath, output_filepath):
def main(dimensions):
    # logger = logging.getLogger(__name__)
    # logger.info("making final data set from raw data")
    path_to_glove_pkl = ".\\data\\interim\\filtered_embeddings.pkl" # TODO : Add to .env
    embeddings_index = parse_glove_data(
        f".\\data\\raw\\glove.6B.{dimensions}d.txt", progress_bar=True
    )
    words_array = get_word_list(embeddings_index)
    words_array = filter_alphabetic(words_array)
    words_array = filter_20k(words_array, ".\\data\\external\\20k.txt")
    words_array = filter_wordnet(words_array)
    words_array = lemmatize_words(words_array)

    glove_df = embeddings_to_dataframe(embeddings_index, words_array)
    glove_df.to_pickle(path_to_glove_pkl)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # TODO : Explore if this can be used to remove os/sys from run_simulations.py
    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main(300)
