import tensorflow as tf

class StoryGenerationModel:
    
    def __init__(self, sess, args):
        self.sess = sess
        
        # Data defined values
        self.NUM_SENTS = args.num_sents
        self.SENT_LEN = args.sent_len
        self.TEXT_VOCAB = args.vocab_size
        self.SUBJ_VOCAB = args.subj_vocab
        self.VERB_VOCAB = args.verb_vocab
        self.VOBJ_VOCAB = args.vobj_vocab
        self.MODI_VOCAB = args.modi_vocab

        # Network hyperparameters
        self.EMBED_DIM = args.embed_dim
        self.REPRESENTATION_DIM = args.rep_dim
        self.RNN_HIDDEN_DIM = args.rnn_hidden_dim
        self.REPRESENTATION_LAYERS = args.rep_layers
        self.NEXT_EVENT_LAYERS = args.next_event_layers
        self.RNN_HIDDEN_LAYERS = args.rnn_hidden_layers

        # Training hyperparameters
        self.LEARNING_RATE = args.learn_rate

        self.construct_graph()

    def construct_graph(self):

        # Placeholders for sentence input
        self.sentences = tf.placeholder(tf.int32, [None, self.NUM_SENTS, self.SENT_LEN])
        self.sentence_lengths = tf.placeholder(tf.int32, [None, self.NUM_SENTS])
        
        # Placeholder for number of sentences
        self.num_sentences = tf.placeholder(tf.int32, [None])

        # Placeholder for target events
        self.subj_target = tf.placeholder(tf.int32, [None, self.NUM_SENTS+1])
        self.verb_target = tf.placeholder(tf.int32, [None, self.NUM_SENTS+1])
        self.vobj_target = tf.placeholder(tf.int32, [None, self.NUM_SENTS+1])
        self.modi_target = tf.placeholder(tf.int32, [None, self.NUM_SENTS+1])

        # Placeholders for current sentence
        self.current_sentence = tf.placeholder(tf.int32, [None, self.SENT_LEN])
        self.curr_sent_lengths = tf.placeholder(tf.int32, [None])

        # Event representation module
        with tf.name_scope('event_representation'):

            # Stack sentences along batch dimension to fit into RNN
            stacked_sentences = tf.reshape(self.sentences, [-1, self.SENT_LEN])
            stacked_sent_lens = tf.reshape(self.sentence_lengths, [-1])

            # Retrieve learned embeddings
            embed_var = tf.get_variable('word_embed', shape=[self.SENT_LEN, self.EMBED_DIM],
                                        dtype=tf.float32)
            word_embedding = tf.embedding_lookup(embed_var, stacked_sentences)

            # Pass through a multi layer LSTM to get event representation
            rep_out, rep_state = self.make_multilayer_lstm(word_embedding, stacked_sent_lens,
                                                           self.REPRESENTATION_DIM,
                                                           self.REPRESENTATION_LAYERS)

            # Get event representations
            stacked_representations = tf.squeeze(rep_out[:, -1, :], axis=1)
            self.event_reps = tf.reshape(stacked_representations,
                                         [-1, self.NUM_SENTS, self.REPRESENTATION_LAYERS])

        # Next event module
        with tf.name_scope('next_event'):
        
            # Pass through a multi layer LSTM to get next event
            next_out, next_state = self.make_multilayer_lstm(self.event_reps, self.num_sentences,
                                                             self.REPRESENTATION_DIM,
                                                             self.NEXT_EVENT_LAYERS)

            # Get next event
            self.next_event = next_out[:, -1, :]

        # Event target prediction module
        with tf.name_scope('event_target'):

            # Stack events together
            all_events = tf.concat([self.event_reps, self.next_event], axis=1)
            stacked_all_events = tf.reshape(all_events, [-1, self.REPRESENTATION_DIM])

            # Get loss on event target predictions
            with tf.name_scope('subj'):
                stacked_subj_target = tf.reshape(self.subj_target, [-1])
                self.subj_loss = self.make_target_loss(stacked_all_events, stacked_subj_target,
                                                       self.SUBJ_VOCAB)
            with tf.name_scope('verb'):
                stacked_verb_target = tf.reshape(self.verb_target, [-1])
                self.verb_loss = self.make_target_loss(stacked_all_events, stacked_verb_target,
                                                       self.VERB_VOCAB)
            with tf.name_scope('vobj'):
                stacked_vobj_target = tf.reshape(self.vobj_target, [-1])
                self.vobj_loss = self.make_target_loss(stacked_all_events, stacked_vobj_target,
                                                       self.VOBJ_VOCAB)
            with tf.name_scope('modi'):
                stacked_modi_target = tf.reshape(self.modi_target, [-1])
                self.modi_loss = self.make_target_loss(stacked_all_events, stacked_modi_target,
                                                       self.MODI_VOCAB)

        # Sentence prediction module
        with tf.name_scope('sentence_prediction'):

            # Pass through a multi layer LSTM to predict word
            sent_input = self.current_sentence[:,:-1]
            sent_out, sent_state = self.make_multilayer_lstm(sent_input, self.curr_sent_lengths,
                                                             self.RNN_HIDDEN_DIM,
                                                             self.RNN_HIDDEN_LAYERS)

            # Get loss on next word predictions
            stacked_sent_pred = tf.reshape(sent_out, [-1, self.RNN_HIDDEN_DIM])
            stacked_sent_targets = tf.reshape(self.current_sentence[:,1:], [-1])
            self.sent_loss = self.make_target_loss(stacked_sent_pred, stacked_sent_targets,
                                                   self.TEXT_VOCAB)

        # Optimize over total loss
        with tf.name_scope('optimizer'):

            self.total_loss = (self.subj_loss + self.verb_loss + self.vobj_loss + self.modi_loss
                               + self.sent_loss)
            self.optim = tf.train.AdamOptimizer(self.LEARNING_RATE)
            self.train_step = self.optim.minimize(self.total_loss)

        # Logging for Tensorboard
        with tf.name_scope('logging'):
            tf.summary.scalar('subj_loss', self.subj_loss)
            tf.summary.scalar('verb_loss', self.verb_loss)
            tf.summary.scalar('vobj_loss', self.vobj_loss)
            tf.summary.scalar('modi_loss', self.modi_loss)
            tf.summary.scalar('sent_loss', self.sent_loss)
            tf.summary.scalar('total_loss', self.total_loss)

            self.merged = tf.summary.merge_all()

        # Initialize all variables
        self.sess.run(tf.global_variables_initializer()


    def make_multilayer_lstm(self, seq_input, seq_len, hid_dim, num_layers):
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(hid_dim)
        cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell * num_layers])
        out, state = tf.nn.dynamic_rnn(next_cell, seq_input, dtype=tf.float32,
                                       sequence_length=seq_len)
        return out, state


    def make_target_loss(self, pred_input, target, vocab):
        prediction = tf.layers.dense(pred_input, vocab)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=prediction)
        return tf.reduce_mean(loss)
