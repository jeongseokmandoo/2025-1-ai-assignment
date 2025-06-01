import numpy as np
import os # Ensure os is imported
import subprocess # Add subprocess back

# --- PARSING FUNCTIONS MOVED FROM parser.py --- 
def parse_mfcc_file(filepath):
    """
    Parses an MFCC feature file.

    Args:
        filepath (str): Path to the MFCC file.
                        The first line should be "num_frames num_dimensions".
                        Subsequent lines are MFCC vectors.

    Returns:
        list[list[float]]: A list of MFCC vectors (list of floats).
                           Returns an empty list if the file is not found or is malformed.
    """
    mfcc_vectors = []
    try:
        with open(filepath, 'r') as f:
            first_line = f.readline().strip().split()
            if len(first_line) != 2:
                # print(f"Warning: Malformed first line in {filepath}. Expected 2 numbers, got {len(first_line)}.")
                return []
            
            try:
                num_frames = int(first_line[0])
                num_dimensions = int(first_line[1])
            except ValueError:
                # print(f"Warning: Malformed first line in {filepath}. Could not parse num_frames/num_dimensions.")
                return []

            for line in f:
                parts = line.strip().split()
                if len(parts) != num_dimensions:
                    # print(f"Warning: Malformed data line in {filepath}. Expected {num_dimensions} values, got {len(parts)}.")
                    continue 
                try:
                    mfcc_vectors.append([float(p) for p in parts])
                except ValueError:
                    # print(f"Warning: Non-float value found in data line in {filepath}.")
                    continue
            
            if len(mfcc_vectors) != num_frames:
                # print(f"Warning: Number of frames in {filepath} ({len(mfcc_vectors)}) does not match header ({num_frames}).")
                pass

    except FileNotFoundError:
        # print(f"Error: MFCC file not found at {filepath}")
        return []
    except Exception as e:
        # print(f"An unexpected error occurred while parsing {filepath}: {e}")
        return []
    return mfcc_vectors

def parse_hmm_file(base_dir):
    """
    Parses the HMM parameter file (hmm.txt).

    Args:
        base_dir (str): The base directory where hmm.txt is located.

    Returns:
        dict: A dictionary where keys are phoneme names (str) and 
              values are dictionaries containing HMM parameters.
    """
    hmm_file_path = os.path.join(base_dir, "hmm.txt")
    hmms = {}
    current_hmm_name = None
    current_hmm_data = {}
    num_states_for_hmm = 0
    
    try:
        with open(hmm_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                if line.startswith("~h"):
                    if current_hmm_name:
                        if 'states' in current_hmm_data and len(current_hmm_data['states']) == num_states_for_hmm:
                             hmms[current_hmm_name] = current_hmm_data
                        elif current_hmm_name:
                            pass

                    current_hmm_name = line.split('"')[1]
                    current_hmm_data = {'states': []}
                    num_states_for_hmm = 0
                
                elif line.startswith("<NUMSTATES>"):
                    pass

                elif line.startswith("<STATE>"):
                    current_hmm_data['states'].append({'weights': [], 'means': [], 'variances': []})
                    num_states_for_hmm +=1

                elif line.startswith("<NUMMIXES>"):
                    pass

                elif line.startswith("<MIXTURE>"):
                    parts = line.split()
                    mix_weight = float(parts[2])
                    if current_hmm_data['states']:
                        current_hmm_data['states'][-1]['weights'].append(mix_weight)
                    else:
                        pass

                elif line.startswith("<MEAN>"):
                    parts = line.split()
                    dim = int(parts[1])
                    mean_vector_line = next(f).strip()
                    mean_vector = [float(x) for x in mean_vector_line.split()]
                    if len(mean_vector) != dim:
                        if current_hmm_data['states'] and current_hmm_data['states'][-1]['weights']:
                           current_hmm_data['states'][-1]['weights'].pop()
                        continue
                    if current_hmm_data['states']:
                         current_hmm_data['states'][-1]['means'].append(mean_vector)
                    else:
                        pass

                elif line.startswith("<VARIANCE>"):
                    parts = line.split()
                    dim = int(parts[1])
                    variance_vector_line = next(f).strip()
                    variance_vector = [float(x) for x in variance_vector_line.split()]
                    if len(variance_vector) != dim:
                        if current_hmm_data['states'] and current_hmm_data['states'][-1]['means']:
                           current_hmm_data['states'][-1]['means'].pop()
                           if current_hmm_data['states'][-1]['weights']:
                               current_hmm_data['states'][-1]['weights'].pop()
                        continue
                    if current_hmm_data['states']:
                        current_hmm_data['states'][-1]['variances'].append(variance_vector)
                        if not current_hmm_data['states'][-1]['weights'] and \
                           len(current_hmm_data['states'][-1]['means']) == 1 and \
                           len(current_hmm_data['states'][-1]['variances']) == 1:
                            current_hmm_data['states'][-1]['weights'].append(1.0)
                    else:
                        pass

                elif line.startswith("<TRANSP>"):
                    num_total_states_in_transp = int(line.split()[1])
                    trans_p = []
                    for _ in range(num_total_states_in_transp):
                        row_line = next(f).strip()
                        trans_p.append([float(x) for x in row_line.split()])
                    current_hmm_data['trans_p'] = trans_p
                    current_hmm_data['num_states'] = num_total_states_in_transp - 2

                elif line.startswith("<ENDHMM>"):
                    if current_hmm_name:
                        if 'trans_p' in current_hmm_data:
                            expected_emitting_states = len(current_hmm_data['trans_p']) - 2
                            if len(current_hmm_data['states']) != expected_emitting_states:
                                pass
                            
                            valid_hmm = True
                            if not current_hmm_data['states'] and expected_emitting_states > 0 :
                                valid_hmm = False

                            for i, state_gmm in enumerate(current_hmm_data.get('states', [])):
                                if not state_gmm['weights'] or not state_gmm['means'] or not state_gmm['variances']:
                                    valid_hmm = False
                                    break
                                if not (len(state_gmm['weights']) == len(state_gmm['means']) == len(state_gmm['variances'])):
                                    valid_hmm = False
                                    break
                            if valid_hmm:
                                hmms[current_hmm_name] = current_hmm_data
                        
                    current_hmm_name = None
                    current_hmm_data = {}
                    num_states_for_hmm = 0
            
            if current_hmm_name and 'trans_p' in current_hmm_data and current_hmm_data.get('states'):
                valid_hmm = True
                expected_emitting_states_last = len(current_hmm_data['trans_p']) - 2
                if len(current_hmm_data['states']) != expected_emitting_states_last :
                     pass
                if not current_hmm_data['states'] and expected_emitting_states_last > 0 :
                    valid_hmm = False

                for i, state_gmm in enumerate(current_hmm_data.get('states', [])):
                    if not state_gmm['weights'] or not state_gmm['means'] or not state_gmm['variances']:
                        valid_hmm = False
                        break
                    if not (len(state_gmm['weights']) == len(state_gmm['means']) == len(state_gmm['variances'])):
                        valid_hmm = False
                        break
                if valid_hmm:
                    hmms[current_hmm_name] = current_hmm_data

    except FileNotFoundError:
        return {}
    except Exception as e:
        return {}
    return hmms

def parse_vocabulary_file(base_dir):
    """
    Parses the vocabulary file (vocabulary.txt).

    Args:
        base_dir (str): The base directory where vocabulary.txt is located.

    Returns:
        list[str]: A list of words.
    """
    vocab_file_path = os.path.join(base_dir, "vocabulary.txt")
    vocabulary = []
    try:
        with open(vocab_file_path, 'r') as f:
            for line in f:
                word = line.strip()
                if word:
                    vocabulary.append(word)
    except FileNotFoundError:
        return []
    except Exception as e:
        return []
    return vocabulary

def parse_dictionary_file(base_dir):
    """
    Parses the pronunciation dictionary file (dictionary.txt).

    Args:
        base_dir (str): The base directory where dictionary.txt is located.

    Returns:
        dict: A dictionary where keys are words (str) and values are lists of 
              pronunciations (list[list[str]]).
    """
    dict_file_path = os.path.join(base_dir, "dictionary.txt")
    dictionary = {}
    try:
        with open(dict_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                
                word = parts[0]
                phonemes_str = parts[1]
                phonemes = phonemes_str.split()

                if not phonemes:
                    continue

                if word not in dictionary:
                    dictionary[word] = []
                dictionary[word].append(phonemes)
                
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {}
    return dictionary

def parse_bigram_file(base_dir):
    """
    Parses the bigram language model file (bigram.txt).
    Probabilities are converted to log probabilities.

    Args:
        base_dir (str): The base directory where bigram.txt is located.

    Returns:
        dict: A nested dictionary representing the bigram model.
    """
    bigram_file_path = os.path.join(base_dir, "bigram.txt")
    bigram_model = {}
    try:
        with open(bigram_file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) != 3:
                    continue
                
                prev_word = parts[0]
                next_word = parts[1]
                
                try:
                    prob = float(parts[2])
                    if prob <= 0:
                        continue
                    log_prob = np.log(prob)
                except ValueError:
                    continue

                if prev_word not in bigram_model:
                    bigram_model[prev_word] = {}
                bigram_model[prev_word][next_word] = log_prob
                
    except FileNotFoundError:
        return {}
    except Exception as e:
        return {}
    return bigram_model 

# --- Path Constants --- 
ASR_HOMEWORK_PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
ASR_HOMEWORK_DIR_NAME = "ASR_Homework-1"
ASR_HOMEWORK_DIR_PATH = os.path.join(ASR_HOMEWORK_PARENT_DIR, ASR_HOMEWORK_DIR_NAME)

HMM_FILE_PATH = os.path.join(ASR_HOMEWORK_DIR_PATH, "hmm.txt")
DICT_FILE_PATH = os.path.join(ASR_HOMEWORK_DIR_PATH, "dictionary.txt")
BIGRAM_FILE_PATH = os.path.join(ASR_HOMEWORK_DIR_PATH, "bigram.txt")
VOCAB_FILE_PATH = os.path.join(ASR_HOMEWORK_DIR_PATH, "vocabulary.txt")
REFERENCE_FILE_PATH = os.path.join(ASR_HOMEWORK_DIR_PATH, "reference.txt")
OUTPUT_MLF_FILE_PATH = os.path.join(ASR_HOMEWORK_PARENT_DIR, "recognized.txt")

# Define a small constant for variance flooring
VARIANCE_FLOOR = 1e-8 # You might need to tune this based on dataset/HMMs

def log_gaussian_pdf(x, mean, variance):
    """
    Computes the log probability density of a multivariate Gaussian distribution
    with a diagonal covariance matrix.

    Args:
        x (list[float] or np.ndarray): The observation vector (MFCC frame).
        mean (list[float] or np.ndarray): The mean vector of the Gaussian.
        variance (list[float] or np.ndarray): The diagonal elements of the 
                                               covariance matrix (variances).

    Returns:
        float: The log probability density log P(x | mean, variance).
               Returns -np.inf if dimensions mismatch or invalid variance.
    """
    x = np.asarray(x)
    mean = np.asarray(mean)
    variance = np.asarray(variance)

    if x.ndim != 1 or mean.ndim != 1 or variance.ndim != 1:
        # print("Error: Input vectors for log_gaussian_pdf must be 1-dimensional.")
        return -np.inf
        
    if not (x.shape == mean.shape == variance.shape):
        # print(f"Error: Dimension mismatch in log_gaussian_pdf. x: {x.shape}, mean: {mean.shape}, variance: {variance.shape}")
        return -np.inf

    # Apply variance flooring
    variance = np.maximum(variance, VARIANCE_FLOOR)
    
    D = len(x)
    
    # Log determinant of diagonal covariance matrix: sum(log(variances))
    log_det_cov = np.sum(np.log(variance))
    
    # (x - mean)^T * Sigma_inv * (x - mean) for diagonal covariance
    # Sigma_inv is diagonal with 1/variance_i
    # So, this term is sum(((x_i - mean_i)^2) / variance_i)
    term_exponent = np.sum(((x - mean)**2) / variance)
    
    log_prob = -0.5 * (D * np.log(2 * np.pi) + log_det_cov + term_exponent)
    
    return log_prob

def log_sum_exp(log_probs):
    """
    Computes log(sum(exp(log_probs))) in a numerically stable way.
    
    Args:
        log_probs (list[float] or np.ndarray): A list or array of log probabilities.
        
    Returns:
        float: The log sum exponentiated value.
               Returns -np.inf if log_probs is empty.
    """
    if not log_probs: # Handle empty list case
        return -np.inf
    
    log_probs = np.asarray(log_probs)
    max_log_prob = np.max(log_probs)
    
    if max_log_prob == -np.inf: # All probabilities are zero
        return -np.inf
        
    return max_log_prob + np.log(np.sum(np.exp(log_probs - max_log_prob)))

def calculate_observation_log_likelihood(mfcc_frame, state_gmm_params):
    """
    Calculates the log likelihood of observing an MFCC frame given the GMM
    parameters of an HMM state.
    log P(o_t | q_t=j) = log ( sum_k [ c_jk * N(o_t | mu_jk, Sigma_jk) ] )

    Args:
        mfcc_frame (list[float] or np.ndarray): The MFCC observation vector for time t.
        state_gmm_params (dict): GMM parameters for a single HMM state.
                                 Expected keys: 'weights' (list of float),
                                                'means' (list of lists/np.ndarrays of float),
                                                'variances' (list of lists/np.ndarrays of float).

    Returns:
        float: The total log observation likelihood for the state.
               Returns -np.inf if GMM parameters are invalid or empty.
    """
    weights = state_gmm_params.get('weights', [])
    means = state_gmm_params.get('means', [])
    variances = state_gmm_params.get('variances', [])

    if not weights or not means or not variances or \
       not (len(weights) == len(means) == len(variances)):
        # print("Warning: Invalid or empty GMM parameters in calculate_observation_log_likelihood.")
        return -np.inf

    log_likelihoods_for_mixures = []
    for k in range(len(weights)):
        weight_k = weights[k]
        mean_k = means[k]
        variance_k = variances[k]

        if weight_k <= 0: # Weights must be positive for log
            # This case should ideally not happen if HMM parsing is correct and models are valid
            # print(f"Warning: Non-positive GMM weight {weight_k} encountered. Skipping mixture component.")
            # We could assign a very small log probability or skip
            log_prob_data_given_mixture_k = -np.inf
        else:
            log_weight_k = np.log(weight_k)
            log_pdf_k = log_gaussian_pdf(mfcc_frame, mean_k, variance_k)
            
            if log_pdf_k == -np.inf : # If PDF is zero for this component
                 log_prob_data_given_mixture_k = -np.inf
            else:
                 log_prob_data_given_mixture_k = log_weight_k + log_pdf_k
        
        log_likelihoods_for_mixures.append(log_prob_data_given_mixture_k)

    # Use log-sum-exp to sum the likelihoods from all mixture components
    total_log_likelihood = log_sum_exp(log_likelihoods_for_mixures)
    
    return total_log_likelihood

LOG_ZERO = -np.inf

def safe_log(p):
    """Computes log(p), returning LOG_ZERO if p is zero or negative."""
    if p > 0:
        return np.log(p)
    return LOG_ZERO

def build_word_hmms(phone_hmms, dictionary, vocabulary):
    """
    Builds HMMs for each word in the vocabulary by concatenating phoneme HMMs.

    Args:
        phone_hmms (dict): Dictionary of phoneme HMM parameters.
                           (Output of parse_hmm_file)
        dictionary (dict): Pronunciation dictionary.
                           (Output of parse_dictionary_file)
        vocabulary (list[str]): List of words in the vocabulary.
                                (Output of parse_vocabulary_file)

    Returns:
        dict: A dictionary where keys are word names (str) and values are
              dictionaries containing the word's HMM parameters:
              - 'num_states': Total number of emitting states in the word HMM.
              - 'states_gmm_params': List of GMM parameters for each emitting state.
              - 'log_trans_p': Log transition probability matrix for the word HMM
                               ( (num_states+2) x (num_states+2) ).
    """
    word_hmms_output = {}

    for word_name in vocabulary:
        if word_name not in dictionary or not dictionary[word_name]:
            # print(f"Warning: Word '{word_name}' from vocabulary not found in dictionary or has no pronunciation. Skipping.")
            continue
        
        # Use the first pronunciation if multiple exist
        phoneme_sequence = dictionary[word_name][0]
        
        if not phoneme_sequence:
            # print(f"Warning: Empty phoneme sequence for word '{word_name}'. Skipping.")
            continue

        word_total_emitting_states = 0
        word_states_gmm_params = []
        phoneme_hmm_objects = [] # Store actual phoneme HMM data for easy access

        valid_sequence = True
        for phone_symbol in phoneme_sequence:
            if phone_symbol not in phone_hmms:
                # print(f"Warning: Phoneme '{phone_symbol}' for word '{word_name}' not found in phone_hmms. Skipping word.")
                valid_sequence = False
                break
            current_phone_hmm = phone_hmms[phone_symbol]
            phoneme_hmm_objects.append(current_phone_hmm)
            word_total_emitting_states += current_phone_hmm['num_states']
            word_states_gmm_params.extend(current_phone_hmm['states'])
        
        if not valid_sequence:
            continue
        
        if word_total_emitting_states == 0 and len(phoneme_sequence) > 0 :
             # This might happen if all phonemes in word (e.g. "sil") have 0 emitting states
             # For now, let's allow creating it, Viterbi might handle it or it's a model issue.
             # print(f"Warning: Word '{word_name}' results in 0 total emitting states. Phonemes: {phoneme_sequence}")
             pass


        # Initialize word log transition probability matrix
        # Size: (total_emitting_states + 2 non-emitting) x (total_emitting_states + 2 non-emitting)
        word_log_tp_size = word_total_emitting_states + 2
        word_log_trans_p = np.full((word_log_tp_size, word_log_tp_size), LOG_ZERO)

        current_word_emitting_state_offset = 0 # 0-indexed offset for emitting states within the word

        for i_ph, current_phone_hmm_obj in enumerate(phoneme_hmm_objects):
            phone_name = phoneme_sequence[i_ph]
            phone_num_emitting = current_phone_hmm_obj['num_states']
            # Convert phoneme trans_p to log, handling zeros
            phone_log_trans_p = np.array([[safe_log(p_val) for p_val in row] for row in current_phone_hmm_obj['trans_p']])

            # 1. Transitions from Word Start to First Phoneme's Emitting States
            if i_ph == 0:
                for j_emit in range(phone_num_emitting): # 0 to N_emit-1
                    # Phoneme emitting states are 1 to N_emit in its own trans_p matrix
                    prob_phone_start_to_emit = phone_log_trans_p[0, j_emit + 1]
                    if prob_phone_start_to_emit > LOG_ZERO:
                        # word_log_trans_p[word_start_idx=0][word_emit_idx]
                        word_log_trans_p[0, current_word_emitting_state_offset + j_emit + 1] = prob_phone_start_to_emit
            
            # 2. Internal Transitions within the Current Phoneme
            for src_emit_idx_in_phone in range(phone_num_emitting): # 0 to N_emit-1
                # word_src_state_abs = 1 (for start state) + current_offset + src_emit_idx_in_phone
                word_src_abs_idx = current_word_emitting_state_offset + src_emit_idx_in_phone + 1
                
                # Phoneme emitting states are 1 to N_emit in its own trans_p
                phone_src_in_phone_matrix = src_emit_idx_in_phone + 1

                for dest_emit_idx_in_phone in range(phone_num_emitting): # 0 to N_emit-1
                    # word_dest_state_abs = 1 (for start state) + current_offset + dest_emit_idx_in_phone
                    word_dest_abs_idx = current_word_emitting_state_offset + dest_emit_idx_in_phone + 1
                    phone_dest_in_phone_matrix = dest_emit_idx_in_phone + 1
                    
                    prob = phone_log_trans_p[phone_src_in_phone_matrix, phone_dest_in_phone_matrix]
                    if prob > LOG_ZERO:
                        word_log_trans_p[word_src_abs_idx, word_dest_abs_idx] = prob

            # 3. Transitions from Current Phoneme to Next Phoneme (if not last)
            if i_ph < len(phoneme_hmm_objects) - 1:
                next_phone_hmm_obj = phoneme_hmm_objects[i_ph + 1]
                next_phone_num_emitting = next_phone_hmm_obj['num_states']
                next_phone_log_trans_p = np.array([[safe_log(p_val) for p_val in row] for row in next_phone_hmm_obj['trans_p']])

                for src_emit_idx_in_curr_phone in range(phone_num_emitting): # 0 to N_curr_emit-1
                    word_src_abs_idx = current_word_emitting_state_offset + src_emit_idx_in_curr_phone + 1
                    phone_src_in_curr_phone_matrix = src_emit_idx_in_curr_phone + 1
                    
                    # Log prob from current phoneme's emitting state to its own end state
                    log_prob_curr_emit_to_curr_end = phone_log_trans_p[phone_src_in_curr_phone_matrix, phone_num_emitting + 1]

                    if log_prob_curr_emit_to_curr_end > LOG_ZERO:
                        for dest_emit_idx_in_next_phone in range(next_phone_num_emitting): # 0 to N_next_emit-1
                            word_dest_abs_idx = current_word_emitting_state_offset + phone_num_emitting + dest_emit_idx_in_next_phone + 1
                            phone_dest_in_next_phone_matrix = dest_emit_idx_in_next_phone + 1

                            # Log prob from next phoneme's start state to its emitting state
                            log_prob_next_start_to_next_emit = next_phone_log_trans_p[0, phone_dest_in_next_phone_matrix]

                            if log_prob_next_start_to_next_emit > LOG_ZERO:
                                combined_log_prob = log_prob_curr_emit_to_curr_end + log_prob_next_start_to_next_emit
                                # This assumes a direct path. If multiple paths could form this transition, logaddexp would be needed.
                                # For simple concatenation, this sum is usually what's intended.
                                word_log_trans_p[word_src_abs_idx, word_dest_abs_idx] = \
                                    np.logaddexp(word_log_trans_p[word_src_abs_idx, word_dest_abs_idx], combined_log_prob) \
                                    if word_log_trans_p[word_src_abs_idx, word_dest_abs_idx] > LOG_ZERO else combined_log_prob
            
            # 4. Transitions from Last Phoneme's Emitting States to Word End State
            elif i_ph == len(phoneme_hmm_objects) - 1: # Last phoneme
                for src_emit_idx_in_phone in range(phone_num_emitting): # 0 to N_emit-1
                    word_src_abs_idx = current_word_emitting_state_offset + src_emit_idx_in_phone + 1
                    phone_src_in_phone_matrix = src_emit_idx_in_phone + 1
                    
                    # Log prob from phoneme's emitting state to its own end state
                    log_prob_emit_to_phone_end = phone_log_trans_p[phone_src_in_phone_matrix, phone_num_emitting + 1]
                    
                    if log_prob_emit_to_phone_end > LOG_ZERO:
                        # word_log_trans_p[word_emit_idx][word_end_idx]
                        # word_end_idx is word_total_emitting_states + 1
                        word_log_trans_p[word_src_abs_idx, word_total_emitting_states + 1] = log_prob_emit_to_phone_end
            
            current_word_emitting_state_offset += phone_num_emitting

        word_hmms_output[word_name] = {
            'num_states': word_total_emitting_states,
            'states_gmm_params': word_states_gmm_params,
            'log_trans_p': word_log_trans_p.tolist() # Convert numpy array to list for consistency if desired
        }
        
    return word_hmms_output

def write_mlf_output(output_filepath, recognition_results):
    """
    Writes the recognition results to a file in MLF (Master Label File) format.

    Args:
        output_filepath (str): Path to the output MLF file.
        recognition_results (dict): A dictionary where keys are logical paths
                                    (e.g., "mfc/f/ak/1237743.lab") and values
                                    are lists of recognized word strings.
    """
    try:
        with open(output_filepath, 'w') as f:
            f.write("#!MLF!#\n")
            for logical_path, words in recognition_results.items():
                # Ensure the logical path in MLF is quoted
                f.write(f'"{logical_path}"\n')
                if words: # If words list is not empty
                    for word in words:
                        f.write(f"{word}\n")
                else: # Handle cases where recognition might return an empty list (e.g. error or no words)
                    # MLF typically expects at least one label or a silence model.
                    # For an empty recognition, we might output a special symbol or just the dot.
                    # For now, we'll just put the dot, assuming if it's empty, it's "silence" or unclassifiable.
                    # Or, if the spec requires specific handling for "no words", adjust here.
                    pass # Or f.write("sil\n") if 'sil' is a valid label for silence
                f.write(".\n")
        print(f"Recognition results successfully written to {output_filepath}")
    except IOError as e:
        print(f"Error writing MLF output to {output_filepath}: {e}")

# Constants for Viterbi
SENTENCE_START_SYMBOL = "<s>" 
# SENTENCE_END_SYMBOL = "</s>" # Not explicitly used in forward pass of this assignment's structure, but good to note.

def viterbi_decode(mfcc_frames, word_hmms, vocabulary, bigram_lm, 
                   lambda_lm_scale=1.0, word_insertion_penalty=0.0):
    """
    Performs Viterbi decoding to find the most likely sequence of words
    for the given MFCC frames.

    Args:
        mfcc_frames (list[list[float]]): Sequence of MFCC frames.
        word_hmms (dict): Word HMM parameters from build_word_hmms.
        vocabulary (list[str]): List of words.
        bigram_lm (dict): Bigram language model from parse_bigram_file.
        lambda_lm_scale (float): Scaling factor for language model probabilities.
        word_insertion_penalty (float): Penalty for inserting a word (applied as log_penalty).
                                        A positive value means a penalty (subtracted).

    Returns:
        list[str]: The recognized sequence of words.
                   Returns an empty list if decoding fails or no frames.
    """
    if not mfcc_frames:
        return []

    num_frames = len(mfcc_frames)
    
    vocab_list = [word for word in vocabulary if word in word_hmms] 
    if not vocab_list:
        return []
        
    word_to_idx = {word: i for i, word in enumerate(vocab_list)}
    idx_to_word = {i: word for i, word in enumerate(vocab_list)}
    num_words = len(vocab_list)

    V = [{} for _ in range(num_frames)]
    B = [{} for _ in range(num_frames)]

    log_word_insertion_penalty = -word_insertion_penalty

    current_mfcc_frame = mfcc_frames[0]
    for w_idx, word_str in enumerate(vocab_list):
        word_hmm = word_hmms[word_str]
        num_emitting_states_in_word = word_hmm['num_states']
        
        if num_emitting_states_in_word == 0: 
            continue

        log_lm_prob = LOG_ZERO
        if SENTENCE_START_SYMBOL in bigram_lm and word_str in bigram_lm[SENTENCE_START_SYMBOL]:
            log_lm_prob = bigram_lm[SENTENCE_START_SYMBOL][word_str]
        elif not bigram_lm: 
             log_lm_prob = 0.0 

        for emit_idx in range(num_emitting_states_in_word): 
            log_tp_word_start_to_emit = word_hmm['log_trans_p'][0][emit_idx + 1]
            
            if log_tp_word_start_to_emit > LOG_ZERO: 
                gmm_params_for_state = word_hmm['states_gmm_params'][emit_idx]
                log_obs_prob = calculate_observation_log_likelihood(current_mfcc_frame, gmm_params_for_state)
                
                if log_obs_prob > LOG_ZERO:
                    current_log_prob = (lambda_lm_scale * log_lm_prob) + \
                                       log_tp_word_start_to_emit + \
                                       log_obs_prob + \
                                       log_word_insertion_penalty
                                       
                    state_key = (w_idx, emit_idx) 
                    V[0][state_key] = current_log_prob
                    B[0][state_key] = (SENTENCE_START_SYMBOL, -1, SENTENCE_START_SYMBOL) 
    
    if not V[0]: 
        return []

    # Recursion (t=1 to num_frames-1)
    for t in range(1, num_frames):
        current_mfcc_frame = mfcc_frames[t]
        V_t = {} # V[t]
        B_t = {} # B[t]

        for curr_w_idx, curr_word_str in enumerate(vocab_list):
            curr_word_hmm = word_hmms[curr_word_str]
            num_curr_emit_states = curr_word_hmm['num_states']

            if num_curr_emit_states == 0:
                continue

            for curr_s_idx in range(num_curr_emit_states):
                curr_state_gmm_params = curr_word_hmm['states_gmm_params'][curr_s_idx]
                log_obs_prob = calculate_observation_log_likelihood(current_mfcc_frame, curr_state_gmm_params)

                if log_obs_prob == LOG_ZERO:
                    continue

                max_prev_log_prob_for_curr_state = LOG_ZERO
                best_prev_bp_tuple_for_curr_state = None
                current_state_key = (curr_w_idx, curr_s_idx)

                # 1. Intra-word transitions
                for prev_s_idx_in_curr_word in range(num_curr_emit_states):
                    prev_state_key_in_V = (curr_w_idx, prev_s_idx_in_curr_word)
                    if prev_state_key_in_V not in V[t-1]:
                        continue
                    
                    log_tp_intra = curr_word_hmm['log_trans_p'][prev_s_idx_in_curr_word + 1][curr_s_idx + 1]
                    if log_tp_intra == LOG_ZERO:
                        continue

                    candidate_log_prob = V[t-1][prev_state_key_in_V] + log_tp_intra + log_obs_prob
                    if candidate_log_prob > max_prev_log_prob_for_curr_state:
                        max_prev_log_prob_for_curr_state = candidate_log_prob
                        best_prev_bp_tuple_for_curr_state = (curr_w_idx, prev_s_idx_in_curr_word, curr_word_str)

                # 2. Inter-word transitions
                log_tp_curr_start_to_curr_emit = curr_word_hmm['log_trans_p'][0][curr_s_idx + 1]
                if log_tp_curr_start_to_curr_emit > LOG_ZERO:
                    for prev_w_idx_from_vocab, prev_word_str in enumerate(vocab_list):
                        prev_word_hmm = word_hmms[prev_word_str]
                        num_prev_emit_states = prev_word_hmm['num_states']
                        if num_prev_emit_states == 0:
                            continue

                        log_lm_prob_val = LOG_ZERO
                        if prev_word_str in bigram_lm and curr_word_str in bigram_lm[prev_word_str]:
                            log_lm_prob_val = bigram_lm[prev_word_str][curr_word_str]
                        elif not bigram_lm:
                            log_lm_prob_val = 0.0

                        for prev_s_idx_in_prev_word in range(num_prev_emit_states):
                            prev_state_key_in_V = (prev_w_idx_from_vocab, prev_s_idx_in_prev_word)
                            if prev_state_key_in_V not in V[t-1]:
                                continue
                            
                            log_tp_prev_emit_to_prev_end = prev_word_hmm['log_trans_p'][prev_s_idx_in_prev_word + 1][num_prev_emit_states + 1]
                            if log_tp_prev_emit_to_prev_end == LOG_ZERO:
                                continue
                            
                            candidate_log_prob = (V[t-1][prev_state_key_in_V] +
                                                 log_tp_prev_emit_to_prev_end +
                                                 (lambda_lm_scale * log_lm_prob_val) +
                                                 log_word_insertion_penalty +
                                                 log_tp_curr_start_to_curr_emit +
                                                 log_obs_prob)
                            
                            if candidate_log_prob > max_prev_log_prob_for_curr_state:
                                max_prev_log_prob_for_curr_state = candidate_log_prob
                                best_prev_bp_tuple_for_curr_state = (prev_w_idx_from_vocab, prev_s_idx_in_prev_word, prev_word_str)
                
                if max_prev_log_prob_for_curr_state > LOG_ZERO: # Using > LOG_ZERO which is -np.inf
                    V_t[current_state_key] = max_prev_log_prob_for_curr_state
                    B_t[current_state_key] = best_prev_bp_tuple_for_curr_state
        
        V[t] = V_t
        B[t] = B_t
        if not V[t]:
            return []
    
    # --- Termination ---
    best_final_log_prob = LOG_ZERO
    # Stores (word_idx, emitting_state_idx_0_based) for the best path at the last frame
    best_final_state_key = None 

    if not V[num_frames-1]: # No valid paths reached the final frame's states
        # print("Warning: Viterbi Termination: No valid paths found at the final frame.")
        return []

    for (w_idx, s_idx), log_prob_at_state in V[num_frames-1].items():
        word_str = idx_to_word[w_idx]
        word_hmm = word_hmms[word_str]
        num_emitting_states_in_word = word_hmm['num_states']

        # Log prob from this emitting state (s_idx, matrix index s_idx+1) 
        # to its word's end state (matrix index num_emitting_states_in_word+1)
        log_tp_to_word_end = word_hmm['log_trans_p'][s_idx + 1][num_emitting_states_in_word + 1]

        if log_tp_to_word_end > LOG_ZERO: # If this state can transition to word end
            final_candidate_log_prob = log_prob_at_state + log_tp_to_word_end
            # Note: No LM cost to exit the utterance, nor additional word insertion penalty here.
            # The last word's insertion penalty was already added when it was hypothesized.
            
            if final_candidate_log_prob > best_final_log_prob:
                best_final_log_prob = final_candidate_log_prob
                best_final_state_key = (w_idx, s_idx)
    
    if best_final_state_key is None:
        # print("Warning: Viterbi Termination: Could not find a valid path that ends an HMM.")
        return []

    # --- Backtracking ---
    recognized_sequence = []
    current_t = num_frames - 1
    current_state_key_in_V = best_final_state_key # (word_idx, 0-based_emit_idx)
    
    while current_t >= 0:
        prev_bp_info = B[current_t].get(current_state_key_in_V)
        if prev_bp_info is None:
            if current_t == 0 :
                 current_word_idx_for_t0 = current_state_key_in_V[0]
                 recognized_sequence.append(idx_to_word[current_word_idx_for_t0])
            break 

        prev_word_token = prev_bp_info[0] 
        prev_state_idx_0_based = prev_bp_info[1] 
        prev_word_actual_name = prev_bp_info[2] 

        current_word_idx = current_state_key_in_V[0]
        
        is_new_word_started_at_current_t = False
        if prev_word_token == SENTENCE_START_SYMBOL: 
            is_new_word_started_at_current_t = True
        elif isinstance(prev_word_token, int): 
            if prev_word_token != current_word_idx: 
                is_new_word_started_at_current_t = True
        
        if is_new_word_started_at_current_t:
            recognized_sequence.append(idx_to_word[current_word_idx])

        if prev_word_token == SENTENCE_START_SYMBOL: 
            break 
        
        current_state_key_in_V = (prev_word_token, prev_state_idx_0_based) 
        current_t -= 1
        
        if current_t < 0 and prev_word_token != SENTENCE_START_SYMBOL:
            pass

    return recognized_sequence[::-1]

# Replace the old main function with the new one using global path constants

def main():
    """
    Main function to run the speech recognition process.
    """
    # --- Configuration ---
    lambda_lm_scale_val = 10.0
    word_insertion_penalty_val = 2.5

    # --- 1. Load Models ---
    phone_hmms_data = parse_hmm_file(ASR_HOMEWORK_DIR_PATH)
    if not phone_hmms_data:
        print(f"Failed to load phone HMMs from {HMM_FILE_PATH}. Exiting.")
        return

    vocabulary_list = parse_vocabulary_file(ASR_HOMEWORK_DIR_PATH)
    if not vocabulary_list:
        print(f"Failed to load vocabulary from {VOCAB_FILE_PATH}. Exiting.")
        return

    dictionary_data = parse_dictionary_file(ASR_HOMEWORK_DIR_PATH)
    if not dictionary_data:
        print(f"Failed to load dictionary from {DICT_FILE_PATH}. Exiting.")
        return
        
    bigram_lm_data = parse_bigram_file(ASR_HOMEWORK_DIR_PATH)
    if not bigram_lm_data:
        print(f"Warning: Language model ({BIGRAM_FILE_PATH}) not loaded or empty.")
        pass # Continue without LM if not found or empty

    word_hmms_built = build_word_hmms(phone_hmms_data, dictionary_data, vocabulary_list)
    if not word_hmms_built:
        print("Failed to build word HMMs. Exiting.")
        return

    # --- 2. Prepare Test Data (Get list of MFCC files from reference.txt) ---
    mfcc_file_references = {} 
    try:
        with open(REFERENCE_FILE_PATH, 'r') as ref_f:
            current_lab_path = None
            for line in ref_f:
                line = line.strip()
                if line == "#!MLF!#":
                    continue
                if line.startswith('"') and line.endswith('"'):
                    current_lab_path = line[1:-1] 
                    mfcc_path_in_ref = current_lab_path.replace(".lab", ".txt")
                    actual_mfcc_file_path = os.path.join(ASR_HOMEWORK_DIR_PATH, mfcc_path_in_ref)
                    mfcc_file_references[current_lab_path] = actual_mfcc_file_path
                elif line == ".":
                    current_lab_path = None 
    except FileNotFoundError:
        print(f"Error: Reference MLF file not found at {REFERENCE_FILE_PATH}. Cannot get MFCC file list.")
        return
    
    if not mfcc_file_references:
        print("No MFCC files found from reference.txt. Exiting.")
        return

    # --- 3. Decoding Loop ---
    all_recognition_results = {} 
    print(f"\nStarting decoding with LM scale = {lambda_lm_scale_val}, Word Penalty = {word_insertion_penalty_val}...")

    for i, (lab_path, mfcc_file_path) in enumerate(mfcc_file_references.items()):
        print(f"  Processing file {i+1}/{len(mfcc_file_references)}: {mfcc_file_path} (Logical: {lab_path})") # Reverted print statement
        mfcc_data = parse_mfcc_file(mfcc_file_path)
        if not mfcc_data:
            print(f"    Warning: Could not parse MFCC data for {mfcc_file_path}. Skipping.")
            all_recognition_results[lab_path] = ["<ERROR_PARSING_MFCC>"] 
            continue

        recognized_words = viterbi_decode(mfcc_data, 
                                          word_hmms_built, 
                                          vocabulary_list, 
                                          bigram_lm_data,
                                          lambda_lm_scale=lambda_lm_scale_val,
                                          word_insertion_penalty=word_insertion_penalty_val)
        
        all_recognition_results[lab_path] = recognized_words
        print(f"    Recognized: {' '.join(recognized_words)}")

    # --- 4. Write Results ---
    write_mlf_output(OUTPUT_MLF_FILE_PATH, all_recognition_results)
    print(f"\nAll files processed. Results written to {OUTPUT_MLF_FILE_PATH}")

    # --- 5. Performance Evaluation ---
    print("\n--- Running HResults for performance evaluation ---")
    hresults_executable_path = os.path.join(ASR_HOMEWORK_DIR_PATH, "HResults.exe")
    
    # Check if HResults.exe exists
    if not os.path.exists(hresults_executable_path):
        print(f"Error: HResults.exe not found at {hresults_executable_path}")
        print("Please ensure HResults.exe is in the ASR_Homework-1 directory.")
        return

    # Command arguments for HResults.exe
    # The exact arguments might need adjustment based on HResults.exe documentation or typical usage.
    # A common set of arguments: -I ref.mlf -L word_list.txt rec.mlf
    # We might also need -d dictionary.txt if it uses phoneme level comparison or needs word definitions for scoring.
    # Let's try a common configuration. The order of arguments might matter.
    cmd_hresults = [
        "wine",                   # Added wine for macOS/Linux compatibility
        hresults_executable_path,
        "-p",                     # Added -p flag based on lecture slide
        "-I", REFERENCE_FILE_PATH,  # Reference MLF
        VOCAB_FILE_PATH,          # Vocabulary file (no -L flag, as per slide)
        # "-d", DICT_FILE_PATH, # Dictionary - uncomment if needed by HResults for your specific models/task
        OUTPUT_MLF_FILE_PATH      # Recognized MLF
    ]

    try:
        print(f"Executing: {' '.join(cmd_hresults)}")
        # Since HResults.exe is a Windows executable, it might not run directly on macOS/Linux.
        # If on macOS/Linux and Wine is installed, you might prepend "wine" to the command.
        # For now, let's assume it can be run directly or the user handles compatibility (e.g., via Wine).
        
        # We need to capture the output of HResults.exe to see the confusion matrix.
        result = subprocess.run(cmd_hresults, capture_output=True, text=True, check=False)
        
        print("\n--- HResults Output ---")
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print("--- HResults Errors (if any) ---")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"HResults.exe finished with exit code {result.returncode}")
        else:
            print("HResults.exe completed successfully.")
            print("Please check the output above for the confusion matrix and other statistics.")

    except FileNotFoundError:
        print(f"Error: Could not find or execute HResults.exe at {hresults_executable_path}.")
        print("If you are on macOS or Linux, you might need to run it using Wine (e.g., 'wine HResults.exe ...').")
    except subprocess.CalledProcessError as e:
        print(f"Error running HResults.exe: {e}")
        print("STDOUT:")
        print(e.stdout)
        print("STDERR:")
        print(e.stderr)
    except Exception as e:
        print(f"An unexpected error occurred while trying to run HResults.exe: {e}")

if __name__ == "__main__":
    main()
