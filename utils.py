if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from shuzPy import Variable
import numpy as np
import os
import subprocess

def _dot_var(v, verbose=False):
    dot_var = '{} [label="{}", color=orange, style=filled]\n'

    name = '' if v.name is None else v.name
    if verbose and v.data is not None:
        if v.name is not None:
            name += ': '
        name += str(v.shape) + ' ' + str(v.dtype)
    return dot_var.format(id(v), name)

def _dot_func(f):
    dot_func = '{} [label="{}", color=lightblue, style=filled, shape=box]\n'
    txt = dot_func.format(id(f), f.__class__.__name__)

    dot_edge = '{} -> {}\n'
    for input in f.inputs:
        txt += dot_edge.format(id(input), id(f))
    for output in f.outputs:
        txt += dot_edge.format(id(f), id(output())) #output is weakref
    
    return txt

def get_dot_graph(output, verbose=True):
    txt = ''
    funcs = []
    seen_set = set()

    def add_func(f):
        if f not in seen_set:
            funcs.append(f)
            seen_set.add(f)

    add_func(output.creator)
    txt += _dot_var(output, verbose)

    while funcs:
        func = funcs.pop()
        txt += _dot_func(func)
        for input in func.inputs:
            txt += _dot_var(input, verbose)

            if input.creator is not None:
                add_func(input.creator)
    
    return 'digraph g {\n' + txt + '}'

def plot_dot_graph(output, verbose=True, to_file='graph.pdf'):
    dot_graph = get_dot_graph(output, verbose)

    # step1. save dot data to file
    tmp_dir = os.path.join(os.path.expanduser('~'), ('.shuzPy'))
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    graph_path = os.path.join(tmp_dir, 'tmp_graph.dot')

    with open(graph_path, 'w') as f:
        f.write(dot_graph)

    #step2. call dot command
    extension = os.path.splitext(to_file)[1][1:] #拡張子
    cmd = 'dot {} -T {} -o {}'.format(graph_path, extension, to_file)
    subprocess.run(cmd, shell=True)