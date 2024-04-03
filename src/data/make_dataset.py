# -*- coding: utf-8 -*-
from pathlib import Path
import sys

project_dir = Path(__file__).resolve().parents[2]
sys.path.append(str(project_dir))
from config import *

from tqdm import tqdm
import numpy as np
import pandas as pd
import re
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer


# TODO : Add word list option to not read whole file each time
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
            for line in tqdm(
                f, desc="Reading file", unit="line", total=STANFORD_GLOVE_LENGTH
            ):
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, "f", sep=" ")
                embeddings_index[word] = coefs

    return embeddings_index


def get_word_list(embeddings_index):
    return list(embeddings_index.keys())


def filter_alphabetic(word_list):
    pattern = re.compile(r"^[a-z]+$")
    return [x for x in word_list if pattern.match(x)]


# TODO : Change "20k" to "to_list" or something
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
        shortest_word = min(word_lemma_list, key=len)
        if shortest_word in word_list:
            lemmatized_set.add(shortest_word)

    return list(lemmatized_set)


def filter_stopwords(word_list, embeddings_index):
    return [
        w for w in word_list if np.linalg.norm(embeddings_index[w]) > STOPWORD_THRESHOLD
    ]


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


def main():
    embeddings_index = parse_glove_data(RAW_DATA_FILEPATH, progress_bar=True)
    words_array = get_word_list(embeddings_index)
    words_array = filter_alphabetic(words_array)
    words_array = filter_20k(words_array, ".\\data\\external\\20k.txt")
    words_array = filter_wordnet(words_array)
    words_array = lemmatize_words(words_array)
    words_array = filter_stopwords(words_array, embeddings_index)

    glove_df = embeddings_to_dataframe(embeddings_index, words_array)
    print(glove_df.shape)
    glove_df.to_pickle(FILTERED_FILEPATH)


if __name__ == "__main__":

    main()
