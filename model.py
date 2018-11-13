import tensorflow as tf

class StoryGenerationModel:
    
    def __init__(self, sess, args):
        self.sess = sess
        
        self.NUM_SENTS = args.num_sents
        self.SENT_LEN = args.sent_len
        self.TEXT_VOCAB = args.vocab_size
        self.EMBED_DIM = args.embed_dim
        self.REPRESENTATION_DIM = args.rep_dim
        self.REPRESENTATION_LAYERS = args.rep_layers

    def construct_graph(self):

        # Input placeholders
        
        # Event representation module
        with tf.name_scope('event_representation'):

            # Placeholders for sentence input
            self.sentences = tf.placeholder(tf.int32, [None, self.NUM_SENTS, self.SENT_LEN])
            self.sentence_lengths = tf.placeholder(tf.int32, [None, self.NUM_SENTS])

            # Stack sentences along batch dimension to fit into RNN
            stacked_sentences = tf.reshape(self.sentences, [-1, self.SENT_LEN])
            stacked_sent_lens = tf.reshape(self.sentence_lengths, [-1])

            # Retrieve learned embeddings
            embed_var = tf.get_variable('word_embed', shape=[self.SENT_LEN, self.EMBED_DIM],
                                        dtype=tf.float32)
            word_embedding = tf.embedding_lookup(embed_var, stacked_sentences)

            # Pass through a multi layer LSTM
            lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.REPRESENTATION_DIM)
            cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell * self.REPRESENTATION_LAYERS])
            rep_out, rep_state = tf.nn.dynamic_rnn(cell, word_embedding, dtype=tf.float32,
                                                   sequence_length=stacked_sent_lens)

            # Get event representations
            stacked_representations = tf.squeeze(rep_out[:, -1, :], axis=1)
            self.event_reps = tf.reshape(stacked_representations,
                                         [-1, self.NUM_SENTS, self.REPRESENTATION_LAYERS])


        # Event target prediction module

        # Next event module

        # Sentence prediction
