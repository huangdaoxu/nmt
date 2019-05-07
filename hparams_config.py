import tensorflow as tf


hparams = tf.app.flags.FLAGS


tf.app.flags.DEFINE_string("src_file", "./data/casia2015_en.txt", "train source file dir")
tf.app.flags.DEFINE_string("tgt_file", "./data/casia2015_ch.txt", "train target file dir")
tf.app.flags.DEFINE_string("src_vocab_file", "./data/casia2015_en_vocab.txt", "vocab file dir")
tf.app.flags.DEFINE_string("tgt_vocab_file", "./data/casia2015_ch_vocab.txt", "vocab file dir")
tf.app.flags.DEFINE_string("encoder_type", "normal", "rnn mode")
tf.app.flags.DEFINE_integer("num_layers", 3, "number of rnn layer")
tf.app.flags.DEFINE_integer("num_units", 128, "hidden layer output dimension")
tf.app.flags.DEFINE_integer("batch_size", 32, "batch size")
tf.app.flags.DEFINE_float("input_keep_prob", 0.5, "input keep prob")
tf.app.flags.DEFINE_float("output_keep_prob", 1.0, "output keep prob")
tf.app.flags.DEFINE_float("state_keep_prob", 1.0, "state keep prob")
tf.app.flags.DEFINE_float("learning_rate", 0.001, "learning rate")
tf.app.flags.DEFINE_integer("epoch", 20, "number of training epoch")
tf.app.flags.DEFINE_integer("embedding_size", 200, "vocab vector embedding size")
