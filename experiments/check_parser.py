import tensorflow as tf
import torch.cuda

from graph_lm.data.parser import get_client, parse_docs_client


def main(_argv):
    print("GPU: {}".format(torch.cuda.is_available()))
    docs = ["Barack Obama was born in Hawaii.  He was elected president in 2008."]

    client = get_client()
    parsed_client = list(parse_docs_client(docs=docs, client=client))

    #pipeline = get_pipeline()
    #parsed_pipeline = list(parse_docs(docs=docs, pipeline=pipeline))

    number = 1000

    print("Parsed with client: {}".format(parsed_client))
    #print("Parsed with pipeline: {}".format(parsed_pipeline))
    #pipeline_time = timeit(stmt=lambda: list(parse_docs(docs=docs, pipeline=pipeline)), number=number) / number
    #client_time = timeit(stmt=lambda: list(parse_docs_client(docs=docs, client=client)), number=number) / number
    #print("Client time: {}".format(client_time))
    #print("Pipeline time: {}".format(pipeline_time))

#Parsed with client: [["Barack" (1->2, compound), "Obama" (2->4, nsubjpass), "was" (3->4, auxpass), "born" (4->0, root), "in" (5->6, case), "Hawaii" (6->4, nmod), "." (7->4, punct)], ["He" (1->3, nsubjpass), "was" (2->3, auxpass), "elected" (3->0, root), "president" (4->3, xcomp), "in" (5->6, case), "2008" (6->3, nmod), "." (7->3, punct)]]
#Parsed with client: [["Barack" (1->2, NNP), "Obama" (2->4, NNP), "was" (3->4, VBD), "born" (4->0, VBN), "in" (5->6, IN), "Hawaii" (6->4, NNP), "." (7->4, .)], ["He" (1->3, PRP), "was" (2->3, VBD), "elected" (3->0, VBN), "president" (4->3, NN), "in" (5->6, IN), "2008" (6->3, CD), "." (7->3, .)]]


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.flags.DEFINE_string('stanfordnlp_dir', r'D:\Projects\corenlp\stanfordnlp_resources', 'stanfordnlp_dir')
    tf.app.run()
