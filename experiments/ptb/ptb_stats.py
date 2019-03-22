import tensorflow as tf

from graph_lm.data.treebank import read_treebank_files


def main(argv):
    sents = list(read_treebank_files(tf.flags.FLAGS.data_dir))
    max_len = max(len(s) for s in sents)
    min_len = min(len(s) for s in sents)
    print("Longest sentence: {}".format(max_len))
    print("Shortest sentence: {}".format(min_len))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string(
        'data_dir',
        '../../data/eng_news_txt_tbnk-ptb_revised/data/tokenized_source',
        'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/eng_news_txt_tbnk-data', 'Data directory')
    tf.flags.DEFINE_float('val_size', .10, 'val_size')
    tf.flags.DEFINE_float('test_size', .10, 'test_size')
    tf.flags.DEFINE_float('min_count', 20, 'test_size')
    tf.app.run()
