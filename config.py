# make_dataset.py
GLOVE_DIMENSIONS = 50
STANFORD_GLOVE_LENGTH = 400_000
RAW_DATA_FILEPATH = f".\\data\\raw\\glove.6B.{GLOVE_DIMENSIONS}d.txt"
FILTERED_FILEPATH = f".\\data\\interim\\filtered_glove_{GLOVE_DIMENSIONS}d.pkl"

# run_simulations.py
SIMILARITY_THRESHOLD = 100
DISTANCE_THRESHOLD = 8.0
NUM_RESULTS = 100
OUTPUT_FILEPATH = ".\\data\\processed\\results.csv"