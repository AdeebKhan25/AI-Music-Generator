# AI Music Generator

Welcome to the AI Music Generator project! This repository demonstrates how to generate music using neural networks, resulting in MIDI compositions based on a trained model. By harnessing the power of deep learning, the project explores the creative fusion of music and AI.

## Features

- **AI-driven Music Generation**: Leverages neural networks to generate MIDI files based on patterns learned from musical data.
- **Optimized Training**: Only the best weights are saved, ensuring efficient management of model parameters.
- **Easy Customization**: Modify the model, dataset, or generation parameters to experiment with various musical styles.

## Getting Started

To get started with this project locally, follow these steps:

1. Clone the repository:

```bash
git clone https://github.com/AdeebKhan25/AI-Music-Generator.git
cd AI-Music-Generator
```

2. Download and install python.

3. Install dependencies:

```bash
pip install numpy matplotlib music21 tensorflow keras argparse
```

4. Train the model; provide no. of epochs as command line argument:

```bash
python LSTM.py 100
```

4. Generate the music (You can directly run this file as weights and notes are already there):

```bash
python Predict.py
```

## How it works

At the heart of the AI Music Generator lies a fascinating blend of creativity and mathematics. The project taps into the power of Recurrent Neural Networks (RNNs), specifically Long Short-Term Memory (LSTM) networks, to breathe life into music. Here’s a breakdown of the key concepts:

### Learning the Language of Music

Music, like language, is filled with patterns. Just as a sentence in English flows according to certain grammatical rules and patterns, a melody in music follows a structure based on the relationships between notes and chords over time. The AI model learns these patterns in music through a process called sequence modeling.

Training Phase: During training, the LSTM network analyzes sequences of musical notes, much like a human learning to play by ear. It “listens” to what comes before a note or chord and starts to recognize relationships: how a note typically follows another, how certain chords resolve to others, and what musical phrases sound harmonious.

Memory of Sequences: What makes LSTMs special is their ability to remember long-term dependencies. Unlike standard neural networks, which can only process fixed-size input, LSTMs can remember notes from much earlier in the sequence, allowing the model to develop a sense of musical coherence. It learns when to repeat motifs or when to introduce variations, much like how a composer might reintroduce a theme at different points in a piece.

Adjusting to the Music: During training, the LSTM doesn’t just learn which notes should follow each other—it learns to capture the emotion and flow of music. The network fine-tunes its understanding of transitions, such as the gradual build-up of tension or the sudden introduction of a chord that resolves a progression.

### The Act of Creation: Music Generation

After learning these intricate patterns during training, the LSTM is ready to compose. But how does it go from training to creating actual music?

Seed Sequence: When generating new music, the process begins with a seed sequence. Think of this as the starting point—a short melody, a chord progression, or just a random series of notes from the dataset. This sequence is the spark that the LSTM uses to ignite the rest of the composition.

Predicting the Next Note: The LSTM looks at the seed sequence and asks itself, “Based on everything I’ve learned about musical patterns, what note should come next?” It doesn’t just predict randomly—it uses its memory of what it has learned to make an informed prediction. This prediction is a continuation of the learned patterns, rather than something entirely new or disjointed.

Building the Composition: The process repeats—each time the model predicts a new note, it feeds that note back into itself along with the rest of the sequence. This is where the magic of LSTMs truly shines. By continuously remembering past notes, the model keeps generating music that feels connected and coherent. The result isn’t just a string of random notes but a flowing melody that could resemble something composed by a human.

### Emulating Human Creativity

What’s remarkable is how the AI mimics aspects of human creativity:

Patterns and Structure: Just like a human composer who might start a melody with a theme and then build upon it, the LSTM picks up on recurring motifs and patterns from its training data. It often generates sequences that feel familiar yet novel—just like when composers create variations on a theme.

Memory and Continuity: A human composer remembers the notes they’ve already written and chooses future notes accordingly to create harmony or contrast. Similarly, the LSTM relies on its long-term memory to ensure continuity in the generated music, producing coherent melodies that “make sense.”

Adaptation: The model is capable of adapting to the type of seed it is given. If the seed is upbeat and fast-paced, the model might continue with energetic rhythms. If the seed is slower or more contemplative, the generated music can follow suit. The model’s adaptability showcases its ability to internalize different musical moods.

### The Beauty of Imperfection

One of the beautiful aspects of AI-generated music is its imperfection. While the LSTM has learned the rules of music, it doesn’t always adhere to them strictly, sometimes producing unexpected notes or chord progressions that feel novel or experimental. This is where AI-generated music becomes more than just an imitation of human creativity—it starts to produce compositions that feel fresh and surprising, like an artist pushing the boundaries of what’s conventional.

## Future Improvements

- Expand the model's ability to generate longer compositions.
- Introduce support for generating multiple instrument tracks.
- Experiment with generative adversarial networks (GANs) to refine the music creation process.
- Convert MIDI output into other formats such as MP3.

### Credits

Took inspiration from similar project by [Sigurður Skúli](https://github.com/Skuldur).

### Further Assistance

If you need any more help or have other questions, feel free to ask. Enjoy creating AI-generated music! 🎼✨
