import os

from tensorflow.python.lib.io.tf_record import TFRecordWriter


class ShardRecordWriter(object):
    def __init__(self, path_fmt, chunksize):
        self.path_fmt = path_fmt
        self.chunksize = chunksize
        self.writer = None
        self.chunks = 0
        self.items = 0

    def __enter__(self):
        self.open_writer()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_writer()

    def output_file(self):
        return self.path_fmt.format(self.chunks)

    def open_writer(self):
        output_file = self.output_file()
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        self.writer = TFRecordWriter(output_file)

    def close_writer(self):
        self.writer.close()
        self.writer = None

    def write(self, record):
        assert self.writer is not None
        if self.items >= self.chunksize:
            self.close_writer()
            self.items = 0
            self.chunks += 1
            self.open_writer()
        self.writer.write(record)
        self.items += 1
