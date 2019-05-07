import os

import jieba
import nltk

import tensorflow as tf

import collections
from collections import Counter

UNK = "<UNK>"
SOS = "<GO>"
EOS = "<EOS>"
UNK_ID = 0


class BatchedInput(collections.namedtuple("BatchedInput",
                                          ("initializer",
                                           "source",
                                           "target",
                                           "source_sequence_length",
                                           "target_sequence_length",
                                           "source_file",
                                           "target_file"))):
    pass


def handle_datum2017(source_path, result_path):
    for i in range(1, 21, 1):
        with open(os.path.join(result_path, "Book_en.txt"), mode="a+") as f_en:
            with open(os.path.join(source_path, "Book{}_en.txt".format(i)), mode="r") as f:
                for line in f.readlines():
                    text = nltk.word_tokenize(line)
                    f_en.write(" ".join(text) + "\n")

        with open(os.path.join(result_path, "Book_cn.txt"), mode="a+") as f_cn:
            with open(os.path.join(source_path, "Book{}_cn.txt".format(i)), mode="r") as f:
                for line in f.readlines():
                    text = jieba.cut(line.replace("\n", ""), cut_all=False)
                    f_cn.write(" ".join(text) + "\n")


def handle_casia2015(source_path, result_path):
    with open(os.path.join(result_path, "casia2015_en.txt"), mode="a+") as f_en:
        with open(os.path.join(source_path, "casia2015_en.txt"), mode="r") as f:
            for line in f.readlines():
                text = nltk.word_tokenize(line)
                f_en.write(" ".join(text) + "\n")

    with open(os.path.join(result_path, "casia2015_ch.txt"), mode="a+") as f_cn:
        with open(os.path.join(source_path, "casia2015_ch.txt"), mode="r") as f:
            for line in f.readlines():
                text = jieba.cut(line.replace("\n", ""), cut_all=False)
                f_cn.write(" ".join(text) + "\n")


def create_vocab_files(file, vocab_file, vocab_freq):
    with open(file, encoding="utf-8", mode="r") as f:
        vocab = Counter(f.read().split())
        vocab = [k for k, v in vocab.items() if v > vocab_freq]
    vocab = [UNK, SOS, EOS] + vocab

    with open(vocab_file, encoding="utf-8", mode="w") as f:
        for word in vocab:
            f.write("%s\n" % word)

    return len(vocab)


def create_vocab_tables(src_vocab_file, tgt_vocab_file):
    src_vocab_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=src_vocab_file, default_value=UNK_ID)

    tgt_vocab_table = tf.contrib.lookup.index_table_from_file(
        vocabulary_file=tgt_vocab_file, default_value=UNK_ID)
    return src_vocab_table, tgt_vocab_table


def get_vocab_size(file):
    with open(file, encoding='utf-8') as f:
        count = len(f.readlines())
    return count


def get_iterator(src_vocab_file, tgt_vocab_file, batch_size,
                 buffer_size=None, random_seed=None,
                 num_threads=4, num_buckets=5):
    source_file = tf.placeholder(dtype=tf.string, shape=None, name="source_file")
    target_file = tf.placeholder(dtype=tf.string, shape=None, name="target_file")
    src_vocab_table, tgt_vocab_table = create_vocab_tables(src_vocab_file, tgt_vocab_file)
    src_vocab_size, tgt_vocab_size = get_vocab_size(src_vocab_file), get_vocab_size(tgt_vocab_file)

    if buffer_size is None:
        buffer_size = batch_size * 5

    src_dataset = tf.data.TextLineDataset(source_file)
    tgt_dataset = tf.data.TextLineDataset(target_file)
    src_tgt_dataset = tf.data.Dataset.zip((src_dataset, tgt_dataset))

    src_tgt_dataset = src_tgt_dataset.shuffle(
        buffer_size, random_seed)

    # split data
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            tf.string_split([src]).values,
            tf.string_split([tgt]).values),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # # get max len data
    # src_tgt_dataset = src_tgt_dataset.map(
    #     lambda src, tgt: (src[:src_max_len], tgt),
    #     num_parallel_calls=num_threads)
    # src_tgt_dataset.prefetch(buffer_size)

    # vocab table look up
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (tf.cast(src_vocab_table.lookup(src), tf.int32),
                          tf.cast(tgt_vocab_table.lookup(tgt), tf.int32)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    # calculate text line true length
    src_tgt_dataset = src_tgt_dataset.map(
        lambda src, tgt: (
            src, tgt, tf.size(src), tf.size(tgt)),
        num_parallel_calls=num_threads)
    src_tgt_dataset.prefetch(buffer_size)

    def batching_func(x):
        return x.padded_batch(
            batch_size,
            # The first three entries are the source and target line rows;
            # these have unknown-length vectors.  The last two entries are
            # the source and target row sizes; these are scalars.
            padded_shapes=(tf.TensorShape([None]),  # src
                           tf.TensorShape([None]),  # tgt_input
                           tf.TensorShape([]),
                           tf.TensorShape([])),  # src_len
            # Pad the source and target sequences with eos tokens.
            # (Though notice we don't generally need to do this since
            # later on we will be masking out calculations past the true sequence.
            padding_values=(src_vocab_size + 1,  # src
                            tgt_vocab_size + 1,
                            0,  # tgt_input
                            0))  # src_len -- unused

    def key_func(unused_1, unused_2, src_len):
        bucket_width = 10
        bucket_id = src_len // bucket_width
        return tf.to_int64(tf.minimum(num_buckets, bucket_id))

    def reduce_func(unused_key, windowed_data):
        return batching_func(windowed_data)

    # batched_dataset = src_tgt_dataset.apply(tf.contrib.data.group_by_window(
    #     key_func=key_func, reduce_func=reduce_func, window_size=batch_size
    # ))
    batched_dataset = batching_func(src_tgt_dataset)

    batched_iter = batched_dataset.make_initializable_iterator()
    (src_ids, tgt_ids, src_seq_len, tgt_seq_len) = (batched_iter.get_next())

    return BatchedInput(
        initializer=batched_iter.initializer,
        source=src_ids,
        target=tgt_ids,
        source_sequence_length=src_seq_len,
        target_sequence_length=tgt_seq_len,
        source_file=source_file,
        target_file=target_file), src_vocab_table, tgt_vocab_table


def process_decoder_input(target_data, tgt_vocab_table, batch_size):
    """
    Preprocess target data for encoding
    :return: Preprocessed target data
    """
    # get '<GO>' id
    go_id = 5

    after_slice = tf.strided_slice(target_data, [0, 0], [batch_size, -1], [1, 1])
    after_concat = tf.concat([tf.fill([batch_size, 1], go_id), after_slice], 1)

    return after_concat


if __name__ == "__main__":
    # handle_casia2015("/Users/hdx/data/casia2015/", "./data/")
    # create_vocab_files("./data/casia2015_ch.txt", "./data/casia2015_ch_vocab.txt", 10)
    # create_vocab_files("./data/casia2015_en.txt", "./data/casia2015_en_vocab.txt", 10)
    iterator, src_vocab_table, tgt_vocab_table = get_iterator(
        src_vocab_file="./data/casia2015_en_vocab.txt",
        tgt_vocab_file="./data/casia2015_ch_vocab.txt",
        batch_size=2, random_seed=666
    )

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        tf.tables_initializer().run()
        for i in range(0, 1):
            sess.run(iterator.initializer,
                     feed_dict={iterator.source_file: "./data/casia2015_en.txt",
                                iterator.target_file: "./data/casia2015_ch.txt"})
            while True:
                try:
                    source, target, src_seq_len, tgt_seq_len = \
                        sess.run([iterator.source,
                                  iterator.target,
                                  iterator.source_sequence_length,
                                  iterator.target_sequence_length])
                    print(source)
                    print(target)
                    print(src_seq_len)
                    print(tgt_seq_len)
                    print()
                except tf.errors.OutOfRangeError:
                    break
