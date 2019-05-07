import collections

import tensorflow as tf

from data_utils import get_vocab_size


class input_tensors(collections.namedtuple("input_tensors",
                                          ("initializer",
                                           "source",
                                           "target",
                                           "source_sequence_length",
                                           "target_sequence_length",
                                           "source_file",
                                           "target_file"))):
    pass


class BaseModel(object):
    def __init__(self):
        self._input_tensors()

    def _input_tensors(self):
        self.source = tf.placeholder(dtype=tf.int32, shape=[None, None], name="source")
        self.target = tf.placeholder(dtype=tf.int32, shape=[None, None], name="target")

        self.source_sequence_length = tf.placeholder(tf.int32, [None], name='source_sequence_length')
        self.target_sequence_length = tf.placeholder(tf.int32, [None], name='target_sequence_length')

        self.global_step = tf.Variable(0, trainable=False)

        self.learning_rate = tf.placeholder(dtype=tf.float32, shape=[], name='learning_rate')
        self.input_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='input_keep_prob')
        self.output_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='output_keep_prob')
        self.state_keep_prob = tf.placeholder(dtype=tf.float32, shape=[], name='state_keep_prob')

    def _single_cell(self, num_units):
        cell = tf.nn.rnn_cell.LSTMCell(
            num_units,
            initializer=tf.orthogonal_initializer(),
            state_is_tuple=True,
            activation=tf.nn.tanh,
            forget_bias=1.0,
        )

        cell = tf.nn.rnn_cell.DropoutWrapper(
            cell,
            input_keep_prob=self.input_keep_prob,
            output_keep_prob=self.output_keep_prob,
            state_keep_prob=self.state_keep_prob,
        )

        return cell

    def _cell_stack(self, num_units, num_layers):
        cell_stack = []
        for _ in range(0, num_layers):
            cell = self._single_cell(
                num_units=num_units,
            )
            cell_stack.append(cell)

        stack = tf.nn.rnn_cell.MultiRNNCell(
            cell_stack,
            state_is_tuple=True
        )
        return stack

    def _build_encoder(self, hparams):
        embedding_inputs = tf.contrib.layers.embed_sequence(
            self.source,
            vocab_size=get_vocab_size(hparams.src_vocab_file),
            embed_dim=hparams.embedding_size,
        )

        if hparams.encoder_type == "normal":
            cell = self._cell_stack(
                num_units=hparams.num_units,
                num_layers=hparams.num_layers,
            )

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
                cell=cell,
                inputs=embedding_inputs,
                sequence_length=self.source_sequence_length,
                dtype=tf.float32,
            )
        elif hparams.encoder_type == "bi":
            fw_cell = self._cell_stack(
                num_units=hparams.num_units,
                num_layers=hparams.num_layers,
            )

            bw_cell = self._cell_stack(
                num_units=hparams.num_units,
                num_layers=hparams.num_layers,
            )

            encoder_outputs, encoder_state = tf.nn.bidirectional_dynamic_rnn(
                fw_cell,
                bw_cell,
                embedding_inputs,
                dtype=tf.float32,
                sequence_length=self.source_sequence_length,
            )
        else:
            raise ValueError("Unknown encoder_type %s" % hparams.encoder_type)

        return encoder_outputs, encoder_state

    def _decoder_layer_train(self, encoder_state, dec_cell, dec_embed_input,
                             maximum_iterations, output_layer):
        helper = tf.contrib.seq2seq.TrainingHelper(
            dec_embed_input,
            self.target_sequence_length
        )

        decoder = tf.contrib.seq2seq.BasicDecoder(
            dec_cell,
            helper,
            encoder_state,
            output_layer
        )

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            impute_finished=True,
            # maximum_iterations=maximum_iterations
        )

        return outputs

    def _decoder_layer_infer(self, encoder_state, dec_cell, dec_embed_input, start_of_sequence_id,
                         end_of_sequence_id, maximum_iterations, output_layer, batch_size):
        helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(
            dec_embed_input,
            tf.fill([batch_size], start_of_sequence_id),
            end_of_sequence_id
        )

        decoder = tf.contrib.seq2seq.BasicDecoder(
            dec_cell,
            helper,
            encoder_state,
            output_layer
        )

        outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(
            decoder,
            impute_finished=True,
            # maximum_iterations=maximum_iterations
        )

        return outputs

    def _build_decoder(self, hparams, encoder_state, maximum_iterations,
                       tgt_vocab_table):
        target_vocab_size = get_vocab_size(hparams.tgt_vocab_file)

        dec_embed_input = tf.contrib.layers.embed_sequence(
            self.target,
            vocab_size=target_vocab_size,
            embed_dim=hparams.embedding_size,
        )

        cell = self._cell_stack(
            num_units=hparams.num_units,
            num_layers=hparams.num_layers,
        )

        output_layer = tf.layers.Dense(target_vocab_size)

        train_output = self._decoder_layer_train(
            encoder_state,
            cell,
            dec_embed_input,
            maximum_iterations,
            output_layer,
        )

        infer_output = self._decoder_layer_infer(
            encoder_state,
            cell,
            dec_embed_input,
            tgt_vocab_table.lookup(tf.constant('<GO>')),
            tgt_vocab_table.lookup(tf.constant('<EOS>')),
            maximum_iterations,
            output_layer,
            hparams.batch_size,
        )

        return train_output, infer_output

    def _build_model(self, hparams, maximum_iterations, tgt_vocab_table):
        enc_outputs, enc_states = self._build_encoder(hparams)

        train_output, infer_output = self._build_decoder(
            hparams, enc_states, maximum_iterations, tgt_vocab_table
        )

        return train_output, infer_output

    def build_graph(self, graph, hparams, tgt_vocab_table):
        with graph.as_default():
            train_logits, inference_logits = self._build_model(hparams, None, tgt_vocab_table)

            training_logits = tf.identity(train_logits.rnn_output, name='logits')
            inference_logits = tf.identity(inference_logits.sample_id, name='predictions')

            masks = tf.sequence_mask(self.target_sequence_length, None, dtype=tf.float32, name='masks')

            cost = tf.contrib.seq2seq.sequence_loss(
                training_logits,
                self.target,
                masks
            )

            optimizer = tf.train.AdamOptimizer(self.learning_rate)
            gradients = optimizer.compute_gradients(cost)
            capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
            train_op = optimizer.apply_gradients(capped_gradients)

        return train_op, inference_logits






