import tensorflow as tf

from hparams_config import hparams
from model import BaseModel
from data_utils import get_iterator


if __name__ == "__main__":
    iterator, src_vocab_table, tgt_vocab_table = get_iterator(
        src_vocab_file=hparams.src_vocab_file,
        tgt_vocab_file=hparams.tgt_vocab_file,
        batch_size=hparams.batch_size, random_seed=666,
    )

    seq2seq = BaseModel()
    graph = tf.Graph()
    train_op, inference_logits = seq2seq.build_graph(graph, hparams, tgt_vocab_table)

    with tf.Session(graph=graph) as sess:
        sess.run(tf.global_variables_initializer())
        tf.tables_initializer().run()

        current_epoch = sess.run(seq2seq.global_step)

        while True:
            if current_epoch > hparams.epoch: break
            sess.run(iterator.initializer,
                     feed_dict={iterator.source_file: hparams.src_file,
                                iterator.target_file: hparams.tgt_file})

            while True:
                try:
                    source, target, src_seq_len, tgt_seq_len = \
                        sess.run([iterator.source,
                                  iterator.target,
                                  iterator.source_sequence_length,
                                  iterator.target_sequence_length])

                    feed_dict = {
                        seq2seq.source: source,
                        seq2seq.target: target,
                        seq2seq.source_sequence_length: src_seq_len,
                        seq2seq.target_sequence_length: tgt_seq_len,
                        seq2seq.learning_rate: hparams.learning_rate,
                        seq2seq.input_keep_prob: hparams.input_keep_prob,
                        seq2seq.output_keep_prob: hparams.output_keep_prob,
                        seq2seq.state_keep_prob: hparams.state_keep_prob,
                    }
                    sess.run(train_op,
                             feed_dict=feed_dict)
                except tf.errors.OutOfRangeError:
                    break


