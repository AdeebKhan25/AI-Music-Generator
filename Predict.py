# This module creates music using trained model

import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm

def prepare_sequences(notes, pitchnames, n_vocab):

    """ 
    Prepare the input sequences used by the Neural Network.

    Args:
        notes (list): A list of notes or chords as strings.
        pitchnames (list): A list of unique pitches in the dataset.
        n_vocab (int): The total number of unique pitches in the dataset.

    Returns:
        tuple: A tuple containing:
            - network_input (list): A list of input sequences, where each sequence is represented as a list of integers.
            - normalized_input (numpy.ndarray): A 3D array of reshaped input sequences, normalized to the range [0, 1].
            - output (list): A list of corresponding output values (integers) for the input sequences.
    """

    note_to_int = {note: number for number, note in enumerate(pitchnames)}
    sequence_length = 100
    network_input = []
    output = []

    for i in range(len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshape the input into a format compatible with LSTM layers
    normalized_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    normalized_input = normalized_input / float(n_vocab)

    return (network_input, normalized_input, output)

def create_network(n_vocab):

    """ 
    Create the structure of the neural network for music generation.

    This function builds a Sequential LSTM model with three LSTM layers,
    Batch Normalization, Dropout, and Dense layers. The model is designed to
    learn from sequences of musical notes.

    Args:
        n_vocab (int): The number of unique musical notes (or classes) in the dataset.

    Returns:
        model (Sequential): A compiled Keras Sequential model ready for inference.
    """
    
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(100, 1),  # Fixed input shape for LSTM layers
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    # Load the best weights to the model
    print(os.path.exists('Weights/best_weights.keras')) 
    model.load_weights('Weights/best_weights.keras')

    return model

def generate_notes(model, network_input, pitchnames, n_vocab):

    """ 
    Generate musical notes from the neural network based on a sequence of notes.

    This function selects a random sequence from the input data and uses the 
    trained model to predict the next notes, generating a specified number of 
    notes in total.

    Args:
        model (Sequential): The trained Keras model used for generating notes.
        network_input (list): The input sequences prepared for the neural network.
        pitchnames (list): A sorted list of unique musical notes (or classes) in the dataset.
        n_vocab (int): The number of unique musical notes (or classes) in the dataset.

    Returns:
        list: A list of generated musical notes as strings.
    """
    
    # Pick a random sequence from the input as a starting point for the prediction
    start = np.random.randint(0, len(network_input) - 1)

    int_to_note = dict((number, note) for number, note in enumerate(pitchnames))


    pattern = network_input[start]
    prediction_output = []

    # Generate 300 notes
    for note_index in range(300):

        prediction_input = np.reshape(pattern, (1, len(pattern), 1))
        prediction_input = prediction_input / float(n_vocab)

        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        pattern.append(index)
        pattern = pattern[1:]  # Keep the pattern length consistent

    return prediction_output

from music21 import note, chord, instrument, stream

def create_midi(prediction_output):

    """ 
    Convert the output from the prediction to musical notes and create a MIDI file 
    from those notes.

    This function processes the generated predictions from the model, which can 
    be individual notes or chords, and constructs a MIDI file. Each note or 
    chord is given a time offset to ensure they are played sequentially.

    Args:
        prediction_output (list): A list of predicted notes and chords as strings.

    Returns:
        None: The function saves the MIDI file in output folder.
    """

    offset = 0  
    output_notes = []  

    for pattern in prediction_output:
        # Check if the pattern is a chord (contains a dot) or a digit (note)
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []

            for current_note in notes_in_chord:
                # Create a new note object
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = instrument.Piano()  # Specify instrument
                notes.append(new_note)

            # Create a chord object and set its offset
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
        else:
            # Create a note object for a single note
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()  # Specify instrument
            output_notes.append(new_note)

        offset += 0.5

    # Create a stream from the output notes and save as MIDI
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp='Output/test_output.mid')

def generate():
    """ 
    Generate a piano MIDI file by loading notes, preparing sequences, 
    creating the model, generating notes, and creating a MIDI file from the notes.
    """
    # Load notes from file
    with open('Data/Notes', 'rb') as filepath:
        notes = pickle.load(filepath)

    pitchnames = sorted(set(item for item in notes))
    n_vocab = len(set(notes))

    network_input, normalized_input, _ = prepare_sequences(notes, pitchnames, n_vocab)
    model = create_network(n_vocab)
    prediction_output = generate_notes(model, network_input, pitchnames, n_vocab)
    create_midi(prediction_output)

if __name__ == '__main__':

    """
    Main function. Code will start running from here.
    """
    generate()
