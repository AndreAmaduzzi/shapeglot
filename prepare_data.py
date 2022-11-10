import numpy as np
import os.path as osp
import pandas as pd

from shapeglot.simple_utils import unique_rows, unpickle_data, pickle_data, invert_dictionary, sort_dict_by_val
from shapeglot.in_out.rnn_data_preprocessing import make_dataset_for_rnn_based_model
from shapeglot.in_out.game_data_preprocessing import preprocess_geometry, preprocess_language
from shapeglot.in_out.game_data_preprocessing import basic_game_statistics
from shapeglot import vis_utils
from shapeglot.vis_utils import visualize_game_example

def main():
    # After running download_data.sh this is where the shapeglot_data should be.
    top_data_dir = './data/main_data_for_chairs' 

    # Downloaded files that will be used to prepare the data:
    game_interactions = osp.join(top_data_dir, 'language/shapenet_chairs.csv')
    misspelling_corrector = osp.join(top_data_dir, 'language/word_spell_manual_corrector_chairs.pkl')
    top_image_dir = osp.join(top_data_dir, 'images/shapenet')
    vis_utils.top_image_dir = top_image_dir

    tokenizer = 'naive'  # ['naive' or 'spacy'] (you can use spacy if it is installed)

    # Replace rare words with <UNK>
    replace_rare = 1  # If a word occurs less/equal to this, is rare. (use 0 to keep all words)


    # For comparative and superlative adjetives break their ending: 
    # nicer -> ['nice', 'er'], nicest -> ['nice', 'est']
    do_compar_superlative = False  # (if True, assumes nltk is installed)


    # Apply some spell-checking we manually created, 
    # or use 'spell_corrector=None' to ignore spelling mistakes.
    spell_corrector = next(unpickle_data(misspelling_corrector, python2_to_3=True))

    # Load the data
    game_data = pd.read_csv(game_interactions)

    # Convert ShapeNet code-names to integers.
    game_data, sn_model_to_int = preprocess_geometry(game_interactions)

    # Tokenize/process utterances.
    game_data, word_to_int = preprocess_language(game_data, 
                                                spell_corrector=spell_corrector,
                                                replace_rare=replace_rare,
                                                tokenizer=tokenizer,
                                                do_compar_superlative=do_compar_superlative)
    print ('Vocabulary, size:', len(word_to_int))

    # Make some auxiliary data-structures that are helpful for accessing the data
    int_to_sn_model = invert_dictionary(sn_model_to_int)
    sorted_sn_models = [k[1] for k in sort_dict_by_val(int_to_sn_model)]
    int_to_word = invert_dictionary(word_to_int)

    ## print some basic statistics of the resulting data.
    basic_game_statistics(game_data, word_to_int)

    rid = np.random.randint(len(game_data))
    #visualize_game_example(game_data, rid, sorted_sn_models, word_to_int) gives error in function "token_ints_to_sentence"
 
    # save the data to the top data directory as a pkl
    save_file = osp.join(top_data_dir, 'game_data.pkl')
    pickle_data(save_file, game_data, word_to_int, int_to_word, int_to_sn_model, sn_model_to_int, sorted_sn_models)

if __name__ =="__main__":
    main()