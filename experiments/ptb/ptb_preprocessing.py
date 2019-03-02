import tensorflow as tf

from graph_lm.data.preprocessing import preprocessing, generate_splits
from graph_lm.data.treebank import read_treebank_files


def main(argv):
    sents = list(read_treebank_files(tf.flags.FLAGS.data_dir))
    data_words = generate_splits(
        sents=sents,
        val_ratio=tf.flags.FLAGS.val_size,
        test_ratio=tf.flags.FLAGS.test_size
    )
    preprocessing(
        data_words=data_words,
        output_dir=tf.flags.FLAGS.output_dir,
        min_count=tf.flags.FLAGS.min_count)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string(
        'data_dir',
        '../../data/eng_news_txt_tbnk-ptb_revised/data/tokenized_source',
        'Data directory')
    tf.flags.DEFINE_string('output_dir', '../../data/eng_news_txt_tbnk-data', 'Data directory')
    tf.flags.DEFINE_float('val_size', .10, 'val_size')
    tf.flags.DEFINE_float('test_size', .10, 'test_size')
    tf.flags.DEFINE_float('min_count', 10, 'test_size')
    tf.app.run()
