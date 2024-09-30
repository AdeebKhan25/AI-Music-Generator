# This module prepares midi file data and feeds it to the neural network for training

import os
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt
from music21 import converter, instrument, note, chord
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
import argparse

def get_notes():

    """
    Extracts all notes and chords from MIDI files in the './Songs' directory and 
    saves them into a file using pickle. Each note and chord is represented as 
    a string.

    Notes:
    - Notes are represented by their pitch (e.g., 'C4', 'E5').
    - Chords are represented by the pitches they contain, separated by dots (e.g., '60.64.67').

    Returns:
        notes (list): A list containing all the extracted notes and chords as strings.
    """

    notes = []

    # Check if the notes file exists
    if os.path.exists('Data/Notes'):
        print("Notes file found.")
        with open('Data/Notes', 'rb') as filepath:
            notes = pickle.load(filepath)
            return notes
    
    for file in glob.glob("Songs/*.mid"):
        midi = converter.parse(file)
        print(f"Parsing {file}")

        notes_to_parse = None

        try:
            parts = instrument.partitionByInstrument(midi)
            notes_to_parse = parts.parts[0].recurse()  
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('Data/Notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):

    """ 
    Prepares the sequences used by the Neural Network.

    Args:
        notes (list): A list of notes or chords as strings.
        n_vocab (int): The total number of unique pitches in the dataset.

    Returns:
        tuple: A tuple containing the reshaped input sequences and the one-hot encoded output.
    """

    sequence_length = 100
    pitchnames = sorted(set(item for item in notes))
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)
    network_input = np.reshape(network_input, (n_patterns, sequence_length, 1))
    network_input = network_input / float(n_vocab)
    
    network_output = to_categorical(network_output, num_classes=n_vocab)

    return (network_input, network_output)

def create_network(network_input, n_vocab):
    
    """
    Create the structure of the neural network for music generation.

    This function builds a Sequential LSTM model with three LSTM layers,
    Batch Normalization, Dropout, and Dense layers. The model is designed to
    learn from sequences of musical notes.

    Parameters:
    network_input (numpy.ndarray): The input data shaped for LSTM layers,
                                    with dimensions (number of samples, 
                                    sequence length, number of features).
    n_vocab (int): The number of unique musical notes (or classes) in the dataset.

    Returns:
    model (Sequential): A compiled Keras Sequential model ready for training.
    """

    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
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

    return model

def train(model, network_input, network_output, epochs=10, weights_path=None):

    """ 
    Train the neural network to generate music, optionally loading pre-trained weights.

    This function fits the model using the provided input and output data,
    saving the best weights achieved during training.

    Parameters:
    ----------
    model : keras.models.Sequential
        The neural network model to be trained.
    
    network_input : numpy.ndarray
        The input sequences prepared for training the neural network.
    
    network_output : numpy.ndarray
        The output sequences corresponding to the input sequences.

    epochs : int
        No. of epochs the model is going to be trained
    
    weights_path : str
        Optional path to pre-trained weights. If provided, the model will load these weights.

    The function saves:
    - A dedicated file `best_weights.keras` that always holds the best 
      model weights achieved during training.

    Callbacks used:
    - ModelCheckpoint for saving best weight.
    """
    # Load pre-trained weights if a path is provided
    if weights_path:
        model.load_weights(weights_path)
        print(f"Loaded weights from {weights_path}")
    
    best_weights_filepath = "Weights/best_weights.keras"

    checkpoint_best = ModelCheckpoint(
        best_weights_filepath,
        monitor='loss',
        verbose=1,
        save_best_only=True,  # Save only the best one
        mode='min'
    )

    early_stopping = EarlyStopping(
        monitor='loss',
        patience=10,  
        verbose=1,
        mode='min',
        restore_best_weights=True
    )

    reduce_lr = ReduceLROnPlateau(
        monitor='loss',
        factor=0.1,
        patience=5,  
        verbose=1,
        mode='min',
        min_lr=1e-6  # Minimum learning rate
    )

    terminate_nan = TerminateOnNaN()

    callbacks_list = [checkpoint_best, early_stopping, reduce_lr, terminate_nan]

    # Train the model and store history
    print("Training started...")
    history = model.fit(network_input, network_output, 
                        epochs=epochs, 
                        batch_size=128, 
                        callbacks=callbacks_list)

    # Plotting the training loss
    plt.figure(figsize=(12, 6))
    plt.plot(history.history['loss'], label='Training Loss', color='blue')
    plt.title('Model Loss During Training')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid()
    plt.show()

def train_network(epochs):

    """ 
    Train a Neural Network to generate music, with the option to load pre-trained weights.
    
    Returns:
    None
    """

    notes = get_notes()
    n_vocab = len(set(notes))
    network_input, network_output = prepare_sequences(notes, n_vocab)
    model = create_network(network_input, n_vocab)
    
    # Define the path for the best weights file
    best_weights_filepath = "Weights/best_weights.keras"

    # Check if the best weights file exists
    if os.path.exists(best_weights_filepath):
        print("Best weights file found. Training will start using those weights...")
        train(model, network_input, network_output, epochs = epochs, weights_path=best_weights_filepath)
    else:
        print("No best weights file found. Training from scratch...")
        train(model, network_input, network_output, epochs = epochs)

if __name__ == '__main__':

    """
    Main function. Code will start running from here.
    """

    parser = argparse.ArgumentParser(description='Train a music generation model.')
    parser.add_argument('epochs', type=int, help='Number of epochs for training')
    args = parser.parse_args()

    train_network(args.epochs)