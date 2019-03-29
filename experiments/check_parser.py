from timeit import timeit

import tensorflow as tf
import torch.cuda

from graph_lm.parser import get_client, get_pipeline, parse_docs, parse_docs_client


def main(_argv):
    print("GPU: {}".format(torch.cuda.is_available()))
    docs = ["Barack Obama was born in Hawaii.  He was elected president in 2008."]

    client = get_client()
    parsed_client = list(parse_docs_client(docs=docs, client=client))

    pipeline = get_pipeline()
    parsed_pipeline = list(parse_docs(docs=docs, pipeline=pipeline))

    number = 1000

    print("Parsed with client: {}".format(parsed_client))
    print("Parsed with pipeline: {}".format(parsed_pipeline))
    #pipeline_time = timeit(stmt=lambda: list(parse_docs(docs=docs, pipeline=pipeline)), number=number) / number
    #client_time = timeit(stmt=lambda: list(parse_docs_client(docs=docs, client=client)), number=number) / number
    #print("Client time: {}".format(client_time))
    #print("Pipeline time: {}".format(pipeline_time))


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('stanfordnlp_dir', r'D:\Projects\corenlp\stanfordnlp_resources', 'stanfordnlp_dir')
    tf.app.run()
