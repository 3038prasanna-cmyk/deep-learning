from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences, to_categorical
from keras.models import Sequential
from keras.layers import Embedding, SimpleRNN, Dense
import numpy as np

# Sample corpus
data = "Deep learning is amazing. Deep learning builds intelligent systems."

# Tokenize the text
tokenizer = Tokenizer()
tokenizer.fit_on_texts([data])

# Create sequences: progressive n-grams
sequences = []
words = data.lower().split()  # lowercase for consistency

for i in range(1, len(words)):
    seq = words[:i+1]
    sequences.append(' '.join(seq))

# Integer encode sequences
encoded = tokenizer.texts_to_sequences(sequences)

# Pad sequences to the max length
max_len = max(len(seq) for seq in encoded)
padded_sequences = pad_sequences(encoded, maxlen=max_len, padding='pre')

# Prepare input (X) and output (y)
X = padded_sequences[:, :-1]
y = padded_sequences[:, -1]

# One-hot encode the output
y = to_categorical(y, num_classes=len(tokenizer.word_index) + 1)

# Build the model
model = Sequential([
    Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=10, input_length=max_len - 1),
    SimpleRNN(50),
    Dense(len(tokenizer.word_index) + 1, activation='softmax')
])

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=200, verbose=0)

print("Training complete!")
