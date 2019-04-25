import itertools
import math
import matplotlib.pyplot as plt
import networkx as nx

from graph_lm.data.calculate_vocabulary import BLANK
from graph_lm.models.networks.utils.bintree_utils import infix_indices, tree_index_map

DEAD = "__DEAD__"
import textwrap


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
    txt = "Japanese _BLANK_ _BLANK_ _BLANK_ domestic _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ " \
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
    txt = "Yet _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ they _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _UNK_ think _BLANK_ that Wall _BLANK_ _BLANK_ _BLANK_ Street is _UNK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ to losing _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ the tax _BLANK_ cut that _BLANK_ _BLANK_ _BLANK_ seemed so _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ close Friday _BLANK_ morning _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ and is _BLANK_ now _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _BLANK_ _UNK_ _BLANK_ ."
    words = txt.split(" ")
    l = len(words)
    depth = int(math.log(l + 1, 2)) - 1
    idxs = infix_indices(depth=depth)
    assert len(idxs) == l

    layers = [
        [None] * (int(math.pow(2, i)))
        for i in range(depth + 1)
    ]
    assert len(layers) == depth + 1
    assert sum(len(layer) for layer in layers) == len(idxs)
    for i, idx in enumerate(idxs):
        depth, output_idx = tree_index_map(idx)
        layers[depth][output_idx] = words[i]
        # if words[i] != BLANK:
        # layers[idx] = words[i]

    mark_dead(layers)
    drop_layers(layers)
    draw_pyramid(layers)

    # nodes = {}
    # for i, idx in enumerate(idxs):
    #    nodes[idx] = words[i]


def mark_dead_helper(layers):
    flag = False
    depth = len(layers)
    for i, layer in enumerate(layers):
        for j, node in enumerate(layer):
            if node == DEAD:
                pass
            elif i == depth - 1 and node == BLANK:
                layer[j] = DEAD
                flag = True
            elif (
                    i + 1 < depth and
                    layers[i + 1][j * 2] == DEAD and
                    layers[i + 1][(j * 2) + 1] == DEAD and
                    node == BLANK):
                layer[j] = DEAD
                flag = True
            elif (
                    i + 1 < depth and
                    layers[i + 1][j * 2] == DEAD and
                    node == BLANK):
                bubble_up(layers, i + 1, (j * 2) + 1)
                flag = True
            elif (
                    i + 1 < depth and
                    layers[i + 1][(j * 2) + 1] == DEAD and
                    node == BLANK):
                bubble_up(layers, i + 1, j * 2)
                flag = True
    return flag


def mark_dead(layers):
    flag = True
    while flag:
        flag = mark_dead_helper(layers)


def drop_layers(layers: list):
    dropped = []
    for i, layer in enumerate(layers):
        if all(word == DEAD for word in layer):
            dropped.append(i)
    for i in reversed(dropped):
        del layers[i]
    return layers


def get_subtree(layers, i_val, j_val):
    subtree = []
    js = [j_val]
    depth = len(layers)
    for i in range(i_val, depth):
        subtree.append([layers[i][j] for j in js])
        js = list(itertools.chain.from_iterable([j * 2, (j * 2) + 1] for j in js))
    return subtree


def delete_subtree(layers, i_val, j_val):
    js = [j_val]
    depth = len(layers)
    for i in range(i_val, depth):
        for j in js:
            layers[i][j] = DEAD
        js = list(itertools.chain.from_iterable([j * 2, (j * 2) + 1] for j in js))
    return layers


def set_subtree(layers, subtree, i_val, j_val):
    js = [j_val]
    ks = [0]
    for i in range(i_val, i_val + len(subtree)):
        for j, k in zip(js, ks):
            layers[i][j] = subtree[i - i_val][k]
        js = list(itertools.chain.from_iterable([j * 2, (j * 2) + 1] for j in js))
        ks = list(itertools.chain.from_iterable([k * 2, (k * 2) + 1] for k in ks))
    return layers


def bubble_up(layers, i_val, j_val):
    subtree = get_subtree(layers, i_val, j_val)
    layers = delete_subtree(layers, i_val, j_val)
    layers = set_subtree(layers, subtree, i_val - 1, j_val // 2)
    return layers


"""
def hspring_group_iter(hgroup: dict):
    forces = {}
    for k, v in hgroup:
        

def hspring_layout(pos:dict, kmin=5):
    hgroups = {}
    for k,v in pos.items():
        if v[1] not in hgroups:
            hgroups[v[1]] = {}
        hgroups[v[1]][k] = v




    outpos = {}
    for group in hgroups.values():
        outpos.update(group)
"""


def draw_pyramid(layers):
    vspace = 10
    hspace = 4000
    font_size = 6
    node_size = 500
    wrap = 70

    G = nx.DiGraph()
    nodes = [["({},{})".format(i, j) for j, _ in enumerate(layer)] for i, layer in enumerate(layers)]
    for i, layer in enumerate(layers):
        for j, word in enumerate(layer):
            if word != DEAD:
                G.add_node(nodes[i][j])
            else:
                pass
    labels = {}
    pos = {}
    undead = []
    nonblank = []
    for i, layer in enumerate(layers):
        v = -vspace * i
        nodes_in_layer = 2 ** i
        h = hspace / (nodes_in_layer + 1)
        # layercount = 0
        # for word in layer:
        #    if word != DEAD:
        #        layercount+=1
        # h = hspace / (layercount + 1)
        # nondead = 0
        for j, word in enumerate(layer):
            if word != DEAD:
                # nondead += 1
                node = nodes[i][j]
                undead.append(node)
                if i > 0:
                    parent = nodes[i - 1][j // 2]
                    G.add_edge(parent, node)
                # pos[node] = (
                #    h * (j+1),
                #    v
                # )
                if word == BLANK:
                    pass
                    # labels[node] = ""
                else:
                    nonblank.append(node)
                    labels[node] = word

    depth = len(layers) - 1
    idxs = infix_indices(depth=depth)
    ordered_nodes = []
    for i, idx in enumerate(idxs):
        d, layer_idx = tree_index_map(idx)
        if layers[d][layer_idx] not in [DEAD, BLANK]:
            ordered_nodes.append([d, layer_idx])
    word_len = len(ordered_nodes)
    hs = hspace / (word_len + 1)
    pos = {}
    for i, (d, layer_idx) in enumerate(ordered_nodes):
        pos[nodes[d][layer_idx]] = (
            (i + 1) * hs,
            -d * vspace
        )
    for i, layer in reversed(list(enumerate(layers))):
        for j, word in enumerate(layer):
            if nodes[i][j] not in pos and word != DEAD:
                child_hs = []
                if nodes[i + 1][j * 2] in pos:
                    child_hs.append(pos[nodes[i + 1][j * 2]][0])
                if nodes[i + 1][(j * 2) + 1] in pos:
                    child_hs.append(pos[nodes[i + 1][(j * 2) + 1]][0])
                assert len(child_hs) > 0
                pos[nodes[i][j]] = (
                    sum(child_hs) / len(child_hs),
                    -i * vspace
                )

    fig = plt.figure(figsize=(12, 4))
    # pos = nx.rescale_layout(pos=pos, scale=1)
    # pos = nx.spring_layout(G, pos=pos, iterations=100, k=0.0000001)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, nodelist=nonblank)
    # pos, cmap=plt.get_cmap('jet'),
    # node_color=values, node_size=500)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=font_size)
    nx.draw_networkx_edges(G, pos)
    # nx.draw_networkx_edges(G, pos, edgelist=red_edges, edge_color='r', arrows=True)
    # nx.draw_networkx_edges(G, pos, edgelist=black_edges, arrows=False)
    text = " ".join(layers[i][j] for i, j in ordered_nodes)
    # plt.title(textwrap.fill(text, width=wrap))
    # plt.title(text)
    plt.text(hspace / 2, 0, textwrap.fill(text, width=wrap), wrap=True, ha='center', va='center')
    # bbox=Bbox(np.array([[0,0],[10,10]])))
    # bbox=Rectangle((0,0), 300, 200))
    plt.show()


if __name__ == '__main__':
    main()
