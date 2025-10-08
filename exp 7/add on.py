import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, Dense, Embedding
from keras.preprocessing.sequence import pad_sequences

# Sample data
input_texts = ['I love NLP', 'He plays football']
target_texts = [['PRON', 'VERB', 'NOUN'], ['PRON', 'VERB', 'NOUN']]

# Special tokens
PAD_WORD = '<PAD>'
PAD_TAG = '<PAD>'

# Build vocabularies with padding token
word_vocab = [PAD_WORD] + sorted(set(word for sent in input_texts for word in sent.split()))
tag_vocab = [PAD_TAG] + sorted(set(tag for tags in target_texts for tag in tags))

word2idx = {w: i for i, w in enumerate(word_vocab)}
tag2idx = {t: i for i, t in enumerate(tag_vocab)}

# Convert input texts to sequences of word indices
encoder_input_seqs = [[word2idx[word] for word in sent.split()] for sent in input_texts]

# Convert target tags to sequences of tag indices
decoder_target_seqs = [[tag2idx[tag] for tag in tags] for tags in target_texts]

# Pad sequences to max length
max_encoder_seq_length = max(len(seq) for seq in encoder_input_seqs)
max_decoder_seq_length = max(len(seq) for seq in decoder_target_seqs)

encoder_input_data = pad_sequences(encoder_input_seqs, maxlen=max_encoder_seq_length, padding='post', value=word2idx[PAD_WORD])
decoder_target_data = pad_sequences(decoder_target_seqs, maxlen=max_decoder_seq_length, padding='post', value=tag2idx[PAD_TAG])

# Create decoder input data by shifting target sequences right and prepending PAD_TAG (as start token)
decoder_input_data = np.zeros_like(decoder_target_data)
decoder_input_data[:, 1:] = decoder_target_data[:, :-1]
decoder_input_data[:, 0] = tag2idx[PAD_TAG]  # Could use a separate <START> token if you want

# Expand decoder target data dims for sparse_categorical_crossentropy
decoder_target_data = np.expand_dims(decoder_target_data, -1)
# Parameters
vocab_size = len(word_vocab)
tag_size = len(tag_vocab)
embedding_dim = 50
latent_dim = 64
# Encoder
encoder_inputs = Input(shape=(None,), name='encoder_inputs')
encoder_embedding = Embedding(vocab_size, embedding_dim, mask_zero=True, name='encoder_embedding')(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True, name='encoder_lstm')
_, state_h, state_c = encoder_lstm(encoder_embedding)
encoder_states = [state_h, state_c]

# Decoder
decoder_inputs = Input(shape=(None,), name='decoder_inputs')
decoder_embedding = Embedding(tag_size, embedding_dim, mask_zero=True, name='decoder_embedding')(decoder_inputs)
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True, name='decoder_lstm')
decoder_outputs, _, _ = decoder_lstm(decoder_embedding, initial_state=encoder_states)
decoder_dense = Dense(tag_size, activation='softmax', name='decoder_dense')
decoder_outputs = decoder_dense(decoder_outputs)
# Define model
model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Summary
model.summary()
# Train the model (tiny data, just for example)
model.fit(
    [encoder_input_data, decoder_input_data],
    decoder_target_data,
    batch_size=2,
    epochs=50,
    verbose=2
)
