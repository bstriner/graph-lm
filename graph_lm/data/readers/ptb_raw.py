import os

PTB_RAW_SPLITS = [
    list(range(0, 19)),
    list(range(19, 22)),
    list(range(22, 25))
]


def read_ptb_characters_fn(data_dir, dirs):
    def fn():
        return read_ptb_characters(data_dir, dirs)

    return fn


def read_ptb_characters(data_dir, dirs):
    # data_dir = tf.flags.FLAGS.data_dir
    for d in dirs:
        source_dir = os.path.join(
            data_dir,
            "{:02d}".format(d)
        )
        for file in os.listdir(source_dir):
            file = os.path.join(source_dir, file)
            if os.path.isfile(file):
                with open(file) as f:
                    for line in f:
                        line = line.strip()
                        if line == ".START":
                            pass
                        elif len(line) == 0:
                            pass
                        else:
                            yield line


if __name__ == '__main__':
    sets = PTB_RAW_SPLITS
    data_dir = r'D:\Projects\data\treebank2\raw\wsj'
    for s in sets:
        count = sum(1 for _ in read_ptb_characters(data_dir, s))
        longest = max(len(l) for l in read_ptb_characters(data_dir, s))
        print("Set: {}, Count: {}, Longest: {}".format(s, count, longest))
