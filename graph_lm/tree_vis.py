from graph_lm.models.networks.utils.bintree_utils import infix_indices, tree_index_map
from graph_lm.data.calculate_vocabulary import UNK, BLANK
import math
from pygr

def is_parent(parent, child):
    return len(parent) < len(child) and parent[:len(child)] == len(child)


def has_children(node, nodes):
    for n in nodes.keys():
        if is_parent(node, n):
            return True
    return False


def clean_nodes(nodes: dict):
    flag = True
    while flag:
        dead = []
        for node, value in nodes.items():
            if value == BLANK and not has_children(node, nodes):
                dead.append(node)
        if len(dead) > 0:
            for node in dead:
                del nodes[node]
            flag = True
        else:
            flag = False
    return nodes


def main():
    txt="Japanese _BLANK_ _BLANK_ _BLANK_ domestic _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "institutions _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ , _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ including _BLANK_ _BLANK_ middle _BLANK_ banks _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ and _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ investment " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ management _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ firms _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ , _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ that _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ had _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ been _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ on _BLANK_ _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ the _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _UNK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ during _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ Monday _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ 's _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ fall _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ _BLANK_ _BLANK_ were _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ back _BLANK_ _BLANK_ _BLANK_ _BLANK_" \
        " _BLANK_ in _BLANK_ _BLANK_ _BLANK_ the _BLANK_ _BLANK_ _BLANK_ market _BLANK_ _BLANK_ , _BLANK_ analysts _BLANK_ _BLANK_ _BLANK_ " \
        "_BLANK_ _BLANK_ said _BLANK_ ."

    words = txt.split(" ")
    l = len(words)
    depth = int(math.log(l+1, 2))-1
    print("Len: {}".format(l))
    print("Depth: {}".format(depth))

    idxs = infix_indices(depth=depth)
    print("Idx: {}".format(len(idxs)))
    assert len(idxs) == l

    layers = [
        [None]*(int(math.pow(i+1,2)-1))
        for i in range(depth+1)
    ]
    assert len(layers)==depth+1

    nodes = {}
    for i,idx in enumerate(idxs):
        nodes[idx] = words[i]

if __name__=='__main__':
    main()