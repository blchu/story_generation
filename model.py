import tensorflow as tf
import os

class StoryGenerationModel:
    
    def __init__(self, sess, args):
        self.sess = sess
        
        # Data defined values
        self.NUM_SENTS = args['num_sents']
        self.SENT_LEN = args['sent_len']
        self.TEXT_VOCAB = args['vocab_size']
        self.SUBJ_VOCAB = args['subj_vocab']
        self.VERB_VOCAB = args['verb_vocab']
        self.VOBJ_VOCAB = args['vobj_vocab']
        self.MODI_VOCAB = args['modi_vocab']

        # Network hyperparameters
        self.EMBED_DIM = args['embed_dim']
        self.REPRESENTATION_DIM = args['rep_dim']
        self.RNN_HIDDEN_DIM = args['rnn_hidden_dim']
        self.REPRESENTATION_LAYERS = args['rep_layers']
        self.NEXT_EVENT_LAYERS = args['next_event_layers']
        self.RNN_HIDDEN_LAYERS = args['rnn_hidden_layers']

        # Training hyperparameters
        self.LEARNING_RATE = args['learn_rate']

        self.construct_graph()

        # Set up writer and saver
        self.writer = tf.summary.FileWriter(args['log_dir'], graph=self.sess.graph)
        self.saver = tf.train.Saver()

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
        self.current_sentence = tf.placeholder(tf.int32, [None, self.SENT_LEN+2])
        self.curr_sent_lengths = tf.placeholder(tf.int32, [None])

        # Event representation module
        with tf.name_scope('event_representation'):

            # Stack sentences along batch dimension to fit into RNN
            stacked_sentences = tf.reshape(self.sentences, [-1, self.SENT_LEN])
            stacked_sent_lens = tf.reshape(self.sentence_lengths, [-1])

            # Retrieve learned embeddings
            with tf.variable_scope('embeddings'):
                embed_var = tf.get_variable('word_embed', shape=[self.SENT_LEN, self.EMBED_DIM],
                                            dtype=tf.float32)
            word_embedding = tf.nn.embedding_lookup(embed_var, stacked_sentences)

            # Pass through a multi layer LSTM to get event representation
            rep_out, rep_state = self.make_multilayer_rnn(word_embedding, stacked_sent_lens,
                                                          self.REPRESENTATION_DIM,
                                                          self.REPRESENTATION_LAYERS,
                                                          cell_name='representation')

            # Get event representations
            # stacked_representations = tf.squeeze(rep_out[:, -1, :], axis=1)
            stacked_representations = rep_out[:, -1, :]
            self.event_reps = tf.reshape(stacked_representations,
                                         [-1, self.NUM_SENTS, self.REPRESENTATION_DIM])

        # Next event module
        with tf.name_scope('next_event'):
        
            # Pass through a multi layer LSTM to get next event
            next_out, next_state = self.make_multilayer_rnn(self.event_reps, self.num_sentences,
                                                            self.REPRESENTATION_DIM,
                                                            self.NEXT_EVENT_LAYERS,
                                                            cell_name='next_event')

            # Get next event
            self.next_event = next_out[:, -1, :]

        # Event target prediction module
        with tf.name_scope('event_target'):

            # Stack events together
            next_event_expanded = tf.expand_dims(self.next_event, axis=1)
            all_events = tf.concat([self.event_reps, next_event_expanded], axis=1)
            stacked_all_events = tf.reshape(all_events, [-1, self.REPRESENTATION_DIM])

            # Get loss on event target predictions
            with tf.name_scope('subj'):
                stacked_subj_target = tf.reshape(self.subj_target, [-1])
                _, self.subj_loss, self.subj_conf = self.make_target_loss(stacked_all_events,
                                                         stacked_subj_target, self.SUBJ_VOCAB)
            with tf.name_scope('verb'):
                stacked_verb_target = tf.reshape(self.verb_target, [-1])
                _, self.verb_loss, self.verb_conf = self.make_target_loss(stacked_all_events,
                                                         stacked_verb_target, self.VERB_VOCAB)
            with tf.name_scope('vobj'):
                stacked_vobj_target = tf.reshape(self.vobj_target, [-1])
                _, self.vobj_loss, self.vobj_conf = self.make_target_loss(stacked_all_events,
                                                         stacked_vobj_target, self.VOBJ_VOCAB)
            with tf.name_scope('modi'):
                stacked_modi_target = tf.reshape(self.modi_target, [-1])
                _, self.modi_loss, self.modi_conf = self.make_target_loss(stacked_all_events,
                                                         stacked_modi_target, self.MODI_VOCAB)


        # TODO: Hierarchical attention

        # Sentence prediction module
        with tf.name_scope('sentence_prediction'):

            # Pass through a multi layer LSTM to predict word
            sent_input = self.current_sentence[:,:-1]
            with tf.variable_scope('embeddings', reuse=True):
                embed_var = tf.get_variable('word_embed', shape=[self.SENT_LEN, self.EMBED_DIM],
                                            dtype=tf.float32)
            sent_embed = tf.nn.embedding_lookup(embed_var, sent_input)
            sent_predict_init = tf.layers.dense(self.next_event, self.RNN_HIDDEN_DIM)
            sent_out, sent_state = self.make_multilayer_rnn(sent_embed, self.curr_sent_lengths,
                                                            self.RNN_HIDDEN_DIM,
                                                            self.RNN_HIDDEN_LAYERS,
                                                            cell_name='sentence',
                                                            init_state=sent_predict_init)

            # Get loss on next word predictions
            stacked_sent_pred = tf.reshape(sent_out, [-1, self.RNN_HIDDEN_DIM])
            stacked_sent_targets = tf.reshape(self.current_sentence[:,1:], [-1])
            pred_logit, self.sent_loss, self.sent_conf = self.make_target_loss(stacked_sent_pred,
                stacked_sent_targets, self.TEXT_VOCAB)
            self.pred = tf.argmax(pred_logit, axis=1)

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

            event_conf = (self.subj_conf + self.verb_conf + self.vobj_conf + self.modi_conf) / 4
            tf.summary.scalar('event_prediction_confidence', event_conf)
            tf.summary.scalar('sentence_prediction_confidence', self.sent_conf)

            self.merged = tf.summary.merge_all()

    def make_multilayer_rnn(self, seq_input, seq_len, hid_dim, num_layers, cell_name=None,
                            init_state=None):
        """
        batch_size = tf.shape(seq_input)[0]
        gru = tf.nn.rnn_cell.GRUCell(hid_dim)
        cell = tf.nn.rnn_cell.MultiRNNCell([gru] * num_layers)
        if init_state is None:
            init_state = tf.tile(tf.get_variable('rnn_0', shape=[1, hid_dim], dtype=tf.float32),
                                 [batch_size, 1])
        print(init_state)
        init_states = init_state
        for i in range(1, num_layers):
            var = tf.get_variable('rnn_%d' % i, shape=[1, hid_dim], dtype=tf.float32)
            init_states.append(tf.tile(var, [batch_size, 1]))
        out, state = tf.nn.dynamic_rnn(cell, seq_input, dtype=tf.float32,
                                       sequence_length=seq_len, initial_state=init_states)
        return out, state
        """
        gru = tf.nn.rnn_cell.GRUCell(hid_dim, name=cell_name)
        out, state = tf.nn.dynamic_rnn(gru, seq_input, dtype=tf.float32,
                                       sequence_length=seq_len, initial_state=init_state)

        return out, state


    def make_target_loss(self, pred_input, target, vocab):
        prediction = tf.layers.dense(pred_input, vocab)
        probs = tf.nn.softmax(prediction)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target, logits=prediction)
        return prediction, tf.reduce_mean(loss), tf.reduce_mean(tf.reduce_max(probs, axis=1))

    def initialize_variables(self, ckpt_dir=None):
        if ckpt_dir:
            # Restore variables from checkpoint
            self.saver.restore(self.sess, ckpt_dir)
        else:
            # Initialize all variables
            self.sess.run(tf.global_variables_initializer())

    def train(self, feed_dict, step, write_summaries=False):
        if write_summaries:
            loss, _, summary = self.sess.run([self.total_loss, self.train_step, self.merged],
                                             feed_dict)
            self.writer.add_summary(summary, step)
        else:
            loss, _ = self.sess.run([self.total_loss, self.train_step], feed_dict)
        return loss

    def decode(self, feed_dict, start_token, end_token):
        sent = [[start_token] + [7 for _ in range(51)]]
        i = 1
        while i < self.SENT_LEN+2 and sent[0][i-1] is not end_token:
            #feed_dict[self.current_sentence] = sent
            feed_dict[self.current_sentence] = [[start_token] + [7 for _ in range(51)]]
            feed_dict[self.curr_sent_lengths] = [i]
            new_sent = self.sess.run(self.pred, feed_dict)
            print(new_sent)
            sent[0][i] = new_sent[i-1]
            i += 1
            print(sent[0][i-1])
        print(i)
        return sent
            

    def save_model(self, save_path):
        saved = self.saver.save(self.sess, os.path.join(save_path, 'model.ckpt'))
        return saved
