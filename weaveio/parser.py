from collections import namedtuple, defaultdict, Counter, OrderedDict
from typing import List, Tuple, Set, Dict

import graphviz
import networkx as nx
from networkx import dfs_tree
from networkx.drawing.nx_pydot import to_pydot


def plot_graph(graph):
    g = nx.DiGraph()
    for n in graph.nodes():
        g.add_node(n)
    for e in graph.edges():
        g.add_edge(*e, **graph.edges[e])
    return graphviz.Source(to_pydot(g).to_string())

def graph2string(graph: nx.DiGraph):
    sources = {n for n in graph.nodes() if len(list(graph.predecessors(n))) == 0}
    return ','.join('->'.join(dfs_tree(graph, source)) for source in sources)

def make_node(graph: nx.DiGraph, parent, subgraph: nx.DiGraph, scalars: list,
              label: str, type: str, operation: str, **edge_data):
    _name = label
    i = graph.number_of_nodes()
    try:
        label = f'{i}\n{graph.nodes[label]["_name"]}'
    except KeyError:
        label = f"{i}\n{label}"
    path = graph2string(subgraph)
    label += f'\n{path}'
    graph.add_node(label, subgraph=subgraph, scalars=scalars, _name=_name, i=i)
    if parent is not None:
        graph.add_edge(parent, label, type=type, label=f"{type}-{operation}", operation=operation, **edge_data)
    return label

def add_start(graph: nx.DiGraph, name):
    g = nx.DiGraph()
    g.add_node(name)
    return make_node(graph, None, g, [], name, '', '')

def add_traversal(graph: nx.DiGraph, parent, path):
    subgraph = graph.nodes[parent]['subgraph'].copy()  # type: nx.DiGraph
    subgraph.add_edge(graph.nodes[parent]['_name'], path[0])
    for a, b in zip(path[:-1], path[1:]):
        subgraph.add_edge(a, b)
    cypher = f'OPTIONAL MATCH {path}'
    return make_node(graph, parent, subgraph, [], ''.join(path[-1:]), 'traversal', cypher)

def add_filter(graph: nx.DiGraph, parent, dependencies, operation):
    subgraph = graph.nodes[parent]['subgraph'].copy()
    n = make_node(graph, parent, subgraph, [], graph.nodes[parent]['_name'], 'filter',
                  f'WHERE {operation}')
    for d in dependencies:
        graph.add_edge(d, n, type='dep', style='dotted')
    return n

def add_aggregation(graph: nx.DiGraph, parent, wrt, operation, type='aggr'):
    subgraph = graph.nodes[wrt]['subgraph'].copy() # type: nx.DiGraph
    n = make_node(graph, parent, subgraph, graph.nodes[parent]['scalars'] + [operation],
                     operation, type, operation)
    graph.add_edge(n, wrt, type='wrt', style='dashed')
    return n

def add_operation(graph: nx.DiGraph, parent, dependencies, operation):
    subgraph = graph.nodes[parent]['subgraph'].copy()  # type: nx.DiGraph
    n = make_node(graph, parent, subgraph, graph.nodes[parent]['scalars'] + [operation],
                  operation, 'operation', f'WITH *, {operation} as ...')
    for d in dependencies:
        graph.add_edge(d, n, type='dep', style='dotted')
    return n

def add_unwind(graph: nx.DiGraph, wrt, sub_dag_nodes):
    sub_dag = nx.subgraph_view(graph, lambda n: n in sub_dag_nodes+[wrt]).copy()  # type: nx.DiGraph
    for node in sub_dag_nodes:
        for edge in sub_dag.in_edges(node):
            if graph.edges[edge]['type'] != 'unwind':  # in case someone else needs it
                graph.remove_edge(*edge) # will be collapsed, so remove the original edge
    others = list(graph.successors(sub_dag_nodes[0]))
    if any(i not in sub_dag_nodes for i in others):
        # if the node is going to be used later, add a way to access it again
        graph.add_edge(wrt, sub_dag_nodes[0], label='unwind', operation='unwind', type='unwind')  # TODO: input correct operation
    graph.remove_node(sub_dag_nodes[-1])
    for node in sub_dag_nodes[:-1]:
        if graph.in_degree[node] + graph.out_degree[node] == 0:
            graph.remove_node(node)

def parse_edge(graph: nx.DiGraph, a, b, dependencies):
    # TODO: will do it properly
    return graph.edges[(a, b)]['operation']

def aggregate(graph: nx.DiGraph, wrt, sub_dag_nodes):
    """
    modifies `graph` inplace
    """
    statement = parse_edge(graph, sub_dag_nodes[-2], sub_dag_nodes[-1], [])
    add_unwind(graph, wrt, sub_dag_nodes)
    return statement


def subgraph_view(graph: nx.DiGraph, excluded_edge_type=None, only_edge_type=None,
                  only_nodes: List = None, excluded_nodes: List = None,
                  only_edges: List[Tuple] = None, excluded_edges: List[Tuple] = None,
                  path_to = None,
                  ) -> nx.DiGraph:
    """
    filters out edges and nodes
    """
    excluded_edges = set([] if excluded_edges is None else excluded_edges)
    excluded_nodes = set([] if excluded_nodes is None else excluded_nodes)
    if excluded_edge_type is not None:
        excluded_edges |= {e for e in graph.edges if graph.edges[e].get('type', '') == excluded_edge_type}
    if only_edge_type is not None:
        excluded_edges |= {e for e in graph.edges if graph.edges[e].get('type', '') != only_edge_type}
    if only_nodes is not None:
        excluded_nodes |= {n for n in graph.nodes if n not in only_nodes}
    if only_edges is not None:
        excluded_edges |= {e for e in graph.edges if e not in only_edges}
    r = nx.restricted_view(graph, excluded_nodes, excluded_edges)  # type: nx.DiGraph
    if path_to:
        r = nx.subgraph_view(r, lambda n:  nx.has_path(graph, n, path_to))
    return r


class ParserError(Exception):
    pass


def get_node_i(graph, i):
    return next(n for n in graph.nodes if graph.nodes[n].get('i', -1) == i)

def node_dependencies(graph, node):
    dag = subgraph_view(graph, excluded_edge_type='wrt')
    return {n for n in graph.nodes if nx.has_path(dag, n, node)} - {node}

def verify_traversal(graph, traversal_order):
    edges = list(zip(traversal_order[:-1], traversal_order[1:]))
    if any(graph.edges[e]['type'] == 'dep' for e in edges):
        raise ParserError(f"Some dep edges where traversed. This is a bug")
    semi_dag = subgraph_view(graph, excluded_edge_type='dep')
    if set(semi_dag.edges) != set(edges):
        raise ParserError(f"Not all edges were traversed. This is a bug")
    done = set()
    for n in traversal_order:
        if n not in done:
            if not all(dep in done for dep in node_dependencies(graph, n)):
                raise ParserError(f"node {n} does not have all its dependencies satisfied. This is a bug")
            done.add(n)


class DeadEndException(Exception):
    pass


def traverse(graph, start=None, end=None, done=None):
    """
    traverse the traversal_graph with backtracking
    """
    dag = subgraph_view(graph, excluded_edge_type='wrt')
    backwards_graph = subgraph_view(graph, only_edge_type='wrt')
    traversal_graph = subgraph_view(dag, excluded_edge_type='dep')
    # semi_traversal = subgraph_view(graph, excluded_edge_type='dep')   # can go through wrt and traversals
    dep_graph = subgraph_view(graph, only_edge_type='dep')
    if start is None or end is None:
        naive_ordering = list(nx.topological_sort(dag))
        if start is None:
            start = naive_ordering[0]  # get top node
        if end is None:
            end = naive_ordering[-1]
    ordering = [start]
    node = start
    done = set() if done is None else done  # stores wrt edges and visited nodes
    while True:
        dependencies = dep_graph.predecessors(node)
        if not all(dep in done for dep in dependencies):
            raise DeadEndException
        options = [b for b in backwards_graph.successors(node) if (node, b) not in done]  # must do wrt first
        if not options:
            options = list(traversal_graph.successors(node))   # where to go next?
        if not options:
            # if you cant go anywhere and you're not done, then this recursive path is bad
            if node != end:
                raise DeadEndException
            else:
                return ordering
        elif len(options) == 1:
            # if there is only one option, go there... obviously
            edge = (node, options[0])
            if edge in done:
                # recursive path is bad if you have to go over the same wrt edge more than once
                raise DeadEndException
            elif graph.edges[edge]['type'] == 'wrt':
                done.add(edge)
            done.add(node)
            node = options[0]
            ordering.append(node)
        else:
            # open up recursive paths from each available option
            # this is such a greedy algorithm
            for option in options:
                try:
                    new_done = done.copy()
                    ordering += traverse(graph, option, end, new_done)
                    done.update(new_done)
                    node = ordering[-1]
                    break
                except DeadEndException:
                    pass  # try another option
            else:
                raise DeadEndException  # all options exhausted, entire recursive path is bad


def verify(graph):
    """
    Check that edges and nodes are allowed:
        - There is only one output node and one input node (no hanging nodes)
        - There is a path from input->output
        - can only aggregate to a parent
        - There are no cyclic dependencies in the dag
        - can only use an aggregation when it's wrt is a parent
        - all operations must be aggregated
        - Multiple inputs into a node should comprise:
            all deps that are aggregated
            one other (can be anything)
        - For an agg node, there is only one wrt
        - You can have > 1 inputs when they are ops

        - Multiple outputs from a node:
            no more than one out-path should be unaggregated in the end
            (i.e. there should only be one path from start-output which contains no aggregations)
    """
    dag = subgraph_view(graph, excluded_edge_type='wrt')
    traversal = subgraph_view(dag, excluded_edge_type='dep')
    if not nx.is_arborescence(traversal):
        raise ParserError(f"Invalid query: The DAG for this query is not a directed tree with max 1 parent per node")
    starts = [n for n in dag.nodes if dag.in_degree(n) == 0]
    ends = [n for n in dag.nodes if dag.out_degree(n) == 0]
    if len(starts) != 1:
        raise ParserError("Only one input node is allowed")
    if len(ends) != 1:
        raise ParserError("Only one output node is allowed")
    backwards = subgraph_view(graph, only_edge_type='wrt')
    without_agg = subgraph_view(dag, excluded_edge_type='aggr')
    main_paths = nx.all_simple_paths(without_agg, starts[0], ends[0])
    try:
        next(main_paths)
        next(main_paths)
    except StopIteration:
        pass
    else:
        # there can be 0 in the case where the output is itself an aggregation
        raise ParserError(f"There can only be at maximum one path from {starts[0]} to {ends[0]} that is not aggregated")
    if not nx.is_directed_acyclic_graph(dag):
        raise ParserError(f"There are cyclical dependencies")
    if not nx.has_path(dag, starts[0], ends[0]):
        raise ParserError(f"There must be a path from {starts[0]} to {ends[0]}")
    for agg, wrt in backwards.edges:
        if not nx.has_path(graph, wrt, agg):
            raise ParserError(f"{wrt} must be a parent of {agg} in order to aggregate")
        for node in dag.successors(agg):
            if not nx.has_path(graph, wrt, node):
                raise ParserError(f"{node} can an only use what is aggregated above it. failure on {agg} (parent={wrt})")
    for node in graph.nodes:
        inputs = [graph.edges[i]['type'] for i in graph.in_edges(node)]
        inputs = [i for i in inputs if i != 'wrt']
        outputs = [graph.edges[i]['type'] for i in graph.out_edges(node)]
        if sum(o == 'wrt' for o in outputs) > 1:
            raise ParserError(f"Cannot put > 1 wrt paths as output from an aggregation")
        outputs = [o for o in outputs if o != 'wrt']
        nfilters = sum(i == 'filter' for i in inputs)
        ntraversals = sum(i == 'traversal' for i in inputs)
        ndeps = sum(i == 'dep' for i in inputs)
        nops = sum(i == 'operation' for i in inputs)
        naggs = sum(i == 'aggr' for i in inputs)
        if naggs > 1:
            raise ParserError(f"Cannot aggregate more than one node at a time: {node}")
        elif naggs:
            if not all(o in ['dep', 'operation'] for o in outputs):
                raise ParserError(f"Can only use aggregations as a dependency/operation afterwards {node}")
        if nfilters > 2:
            raise ParserError(f"Can only have one filter input: {node}")
        elif nfilters:
            if ntraversals + nops + naggs > 0:
                raise ParserError(f"A filter can only take dependencies not traversals/ops/aggregations: {node}")
        if ntraversals > 2:
            raise ParserError(f"Can only have one traversal input: {node}")
        elif ntraversals:
            if len(inputs) > 1:
                raise ParserError(f"Can only traverse with one input: {node}")
        if nops > 1:
            raise ParserError(f"Can only have one op input: {node}")
        elif nops:
            try:
                if graph.edges[list(graph.out_edges(node))[0]]['type'] != 'aggr':
                    raise ParserError(f"All operations must be aggregated back: {node}")
            except IndexError:
                raise ParserError(f"All operations must be aggregated back: {node}")
            if ntraversals + naggs + nfilters > 1:
                raise ParserError(f"Can only have dependencies as input for an operation: {node}")
        if ndeps:
            if ntraversals or naggs:
                raise ParserError(f"A traversal/aggregation cannot take any other inputs: {node}")
            if not (nops ^ nfilters):
                raise ParserError(f"A dependency link necessitates an operation or filter: {node}")

Store = namedtuple('Store', ['state', 'aggs', 'reference', 'chain'])
Aggregation = namedtuple('Aggregation', ['edge', 'reference'])
Load = namedtuple('Load', ['reference', 'state', 'store'])


def merge_overlapping_sequences(sequences):
    sequences.sort(key=len)
    keep = []
    for i, seq in enumerate(sequences):
        matching = {i for i, s in enumerate(sequences) if all(x in s for x in seq)}
        matching.remove(i)
        if not matching:
            keep.append(seq)
    return keep




def get_patterns(sequence, banned=(Store, Load)) -> Dict[int,List[Tuple[Tuple, List[int]]]]:
    """
    finds returns repeated patterns
    :returns:
        final index of the first use of each pattern Dict[last_index,pattern]
        other indexes of other uses of each pattern Dict[first_index, pattern]
    the last_index is where the save/load should occur (do shortest patterns first)
    the first_index is where the load should occur)
    """
    min_length = 1
    counter = defaultdict(set)
    for length in range(min_length, len(sequence)):
        for i in range(len(sequence)):
            seq = tuple(sequence[i:i+length])
            if not any(isinstance(x, banned) for x in seq):
                if not any(x in seq[:z] for z, x in enumerate(seq)):
                    counter[seq].add(i)
    counter = {k: v for k, v in counter.items() if len(v) > 1}
    earliest = {k: min(v) for k, v in counter.items()}
    for seq, index in earliest.items():
        counter[seq].remove(index)
    others = defaultdict(list)
    for k, vs in counter.items():
        for v in vs:
            others[v].append(k)
    others = dict(others)
    for index, seqs in others.items():
        merged_seqs = merge_overlapping_sequences(seqs)
        assert len(merged_seqs) == 1, f"a path starting at {index} can only go to one place"
        others[index] = merged_seqs[0]
    _others = defaultdict(list)
    for k, v in others.items():
        _others[v].append(k)

    earliest = {seq: i for seq, i in earliest.items() if seq in others.values()}  # only if used later
    first_use_last_index = defaultdict(list)
    for seq, i in earliest.items():
        first_use_last_index[i+len(seq)].append(seq)
    grouped = {}
    for first_use_last_i, patterns in first_use_last_index.items():
        patterns.sort(key=len)
        grouped[first_use_last_i] = [(pattern, _others[pattern]) for pattern in patterns]
    # {first_use_last_index: [(pattern, other_first_indexes), ...]}
    g = OrderedDict()
    for k in sorted(grouped.keys()):
        g[k] = grouped[k]
    return g




def find_and_replace_repetition_edges(traversal_edges_order: List[Tuple]):
    """
    To avoid repeating effort, paths that are repeated should be replaced by a `load`
    Given a list of nodes, identify repeating edges and add save/load edges in their place
    """
    new = traversal_edges_order.copy()
    try:
        patterns = get_patterns(new)  # type: OrderedDict
    except ValueError:
        return new
    iadd = 0
    for first_use_last_index, others in patterns.items():
        for pattern, other_use_first_indices in others:
            start, end = pattern[0][0], pattern[-1][-1]
            store = Store([end], [], start, pattern)
            load = Load(start, [end], store)
            for other_use_first_index in other_use_first_indices:
                new[other_use_first_index+iadd] = load
                for i in range(other_use_first_index+iadd+1, other_use_first_index+len(pattern)+iadd):
                    new[i] = None
            new.insert(first_use_last_index+iadd, store)
            new.insert(first_use_last_index+iadd+1, load)
            iadd += 2
    new = [n for n in new if n is not None]
    i = 0
    while i < len(new)-2:
        if new[i] == new[i+1]:
            del new[i+1]
            i = -1
        if i < len(new)-3:
            if isinstance(new[i], Store) and isinstance(new[i+1], Load) and not isinstance(new[i+2], (Store, Load)):
                if new[i+1].store == new[i] and new[i+2][0] in new[i+1].state and new[i+2][1] == new[i+1].reference:
                    new[i].aggs.append(new[i+2])
                    del new[i+1]
                    del new[i+1]
                    i = -1
        i += 1
    return new

def insert_load_saves(traversal_order: List[str]):
    traversal_edges_order = list(zip(traversal_order[:-1], traversal_order[1:]))
    new_traversal_edges_order = find_and_replace_repetition_edges(traversal_edges_order)
    return new_traversal_edges_order

def collapse_chains_of_loading(traversal_edges_order):
    """
    looks for consecutive chains of Loads
    flattens them
    inserts the appropriate Store
    removes unused
    """
    new = traversal_edges_order.copy()
    i = 0
    while i < len(new)-1:
        a, b = new[i], new[i+1]
        if isinstance(a, Load) and isinstance(b, Load):
            if a.state[0] == b.reference:
                load = Load(a.reference, b.state, a.store)
                new[i] = load
                del new[i+1]
        i += 1
    return [n for n in new if n is not None]


def verify_saves(traversal_edges_order, original_traversal_order):
    original_traversal_edges_order = list(zip(original_traversal_order[:-1], original_traversal_order[1:]))
    unwrapped = []
    for i, o in enumerate(traversal_edges_order):
        if isinstance(o, Store):
            if o.aggs:
                for a in o.aggs:
                    unwrapped.append(a)
        elif isinstance(o, Load):
            if traversal_edges_order[i-1] == o.store:
                if not traversal_edges_order[i-1].aggs:
                    continue # only append stuff if its not just saving/loading
            for edge in o.store.chain:
                unwrapped.append(edge)
        else:
            unwrapped.append(o)
    previous = None
    for i, o in enumerate(traversal_edges_order):
        if isinstance(o, Load):
            a = o.reference
            b = o.state[0]
            if o.store not in traversal_edges_order[:i]:
                raise ParserError(f"Cannot load {o} before it is stored. This is a bug")
        elif isinstance(o, Store):
            a = o.state[0]
            b = o.reference
            s = {o.reference, *o.state} | {x for ch in list(o.chain)+o.aggs for x in ch}
            if not all(any(n in before for before in traversal_edges_order[:i]) for n in s):
                raise ParserError(f"Cannot store {o} before it is traversed. This is a bug")
        else:
            a, b = o
        if previous is not None:
            if previous[1] != a:
                raise ParserError(f"Cannot traverse from {previous} to {o}")
        previous = a, b
    if original_traversal_edges_order != unwrapped:
        raise ParserError(f"Saved/Loaded query does not equal the original repetitive query. This is a bug")




class QueryGraph:
    """
    Rules of adding nodes/edges:
    Traversal:
        Can only traverse to another hierarchy object if there is a path between them
        Always increases/maintains cardinality
    Aggregation:
        You can only aggregate back to a predecessor of a node (the parent)
        Nodes which require another aggregation node must share the same parent as just defined above

    Golden rule:
        dependencies of a node must share an explicit parent node
        this basically says that you can only compare nodes which have the same parents

    optimisations:
        If the graph is duplicated in multiple positions, attempt to not redo effort
        For instance, if you traverse and then agg+filter back to a parent and the traverse the same path
        again after filtering, then the aggregation is changed to conserve the required data and the duplicated traversal is removed

    """

    def __init__(self):
        self.G = nx.DiGraph()
        self.start = add_start(self.G, 'data')

    def export(self, fname):
        return plot_graph(self.G).render(fname)

    def add_traversal(self, path, parent=None):
        if parent is None:
            parent = self.start
        return add_traversal(self.G, parent, path)

    def add_operation(self, parent, dependencies, operation):
        # do not allow
        return add_operation(self.G, parent, dependencies, operation)

    def add_aggregation(self, parent, wrt, operation):
        return add_aggregation(self.G, parent, wrt, operation)

    def add_filter(self, parent, dependencies, operation):
        return add_filter(self.G, parent, dependencies, operation)

    def traverse_query(self, result_node=None):
        if result_node is not None:
            graph = nx.subgraph_view(G.G, path_to=result_node)
        else:
            graph = G.G
        verify(graph)
        return traverse(graph)

if __name__ == '__main__':
    G = QueryGraph()

    # # # 0
    # obs = G.add_traversal(['OB'])  # obs = data.obs
    # runs = G.add_traversal(['run'], obs)  # runs = obs.runs
    # spectra = G.add_traversal(['spectra'], runs)  # runs.spectra
    # result = spectra

    # #1
    # obs = G.add_traversal(['OB'])  # obs = data.obs
    # runs = G.add_traversal(['run'], obs)  # runs = obs.runs
    # spectra = G.add_traversal(['spectra'], runs)  # runs.spectra
    # l2 = G.add_traversal(['l2'], runs)  # runs.l2
    # runid2 = G.add_operation(runs, [], 'runid*2 > 0')  # runs.runid * 2 > 0
    # agg = G.add_aggregation(runid2, wrt=obs, operation='all(run.runid*2 > 0)')
    # spectra = G.add_filter(spectra, [agg], 'spectra = spectra[all(run.runid*2 > 0)]')
    # agg_spectra = G.add_aggregation(spectra, wrt=obs, operation='any(spectra.snr > 0)')
    # result = G.add_filter(l2, [agg_spectra], 'l2[any(ob.runs.spectra[all(ob.runs.runid*2 > 0)].snr > 0)]')

    # # 2
    # obs = G.add_traversal(['OB'])  # obs = data.obs
    # runs = G.add_traversal(['run'], obs)  # runs = obs.runs
    # red_runs = G.add_filter(runs, [], 'run.camera==red')
    # red_snr = G.add_aggregation(G.add_operation(red_runs, [], 'run.snr'), obs, 'mean(run.camera==red, wrt=obs)')
    # spec = G.add_traversal(['spec'], runs)
    # spec = G.add_filter(spec, [red_snr], 'spec[spec.snr > red_snr]')
    # result = G.add_traversal(['l2'], spec)

    # # 3
    # # obs = data.obs
    # # x = all(obs.l2s[obs.l2s.ha > 2].hb > 0, wrt=obs)
    # # y = mean(obs.runs[all(obs.runs.l1s[obs.runs.l1s.camera == 'red'].snr > 0, wrt=runs)].l1s.snr, wrt=obs)
    # # z = all(obs.targets.ra > 0, wrt=obs)
    # # result = obs[x & y & z]
    # obs = G.add_traversal(['OB'])  # obs = data.obs
    # l2s = G.add_traversal(['l2'], obs)  # l2s = obs.l2s
    # has = G.add_traversal(['ha'], l2s)  # l2s = obs.l2s.ha
    # above_2 = G.add_aggregation(G.add_operation(has, [], '> 2'), l2s, 'single')  # l2s > 2
    # hb = G.add_traversal(['hb'], G.add_filter(l2s, [above_2], ''))
    # hb_above_0 = G.add_operation(hb, [], '> 0')
    # x = G.add_aggregation(hb_above_0, obs, 'all')
    #
    # runs = G.add_traversal(['runs'], obs)
    # l1s = G.add_traversal(['l1'], runs)
    # camera = G.add_traversal(['camera'], l1s)
    # red = G.add_aggregation(G.add_operation(camera, [], '==red'), l1s, 'single')
    # red_l1s = G.add_filter(l1s, [red], '')
    # red_snrs = G.add_operation(red_l1s, [], 'snr > 0')
    # red_runs = G.add_filter(runs, [G.add_aggregation(red_snrs, runs, 'all')], '')
    # red_l1s = G.add_traversal(['l1'], red_runs)
    # y = G.add_aggregation(G.add_operation(red_l1s, [], 'snr'), obs, 'mean')
    #
    # targets = G.add_traversal(['target'], obs)
    # z = G.add_aggregation(G.add_operation(targets, [], 'target.ra > 0'), obs, 'all')
    #
    # # TODO: need to somehow make this happen in the syntax
    # op = G.add_aggregation(G.add_operation(obs, [x, y, z], 'x&y&z'), obs, 'single')
    # # op = G.add_aggregation(G.add_operation(obs, [x], 'x'), obs, 'single')
    #
    # result = G.add_filter(obs, [op], '')

    #
    # 4
    obs = G.add_traversal(['ob'])  # obs
    exps = G.add_traversal(['exp'], obs)  # obs.exps
    runs = G.add_traversal(['run'], exps)  # obs.exps.runs
    l1s = G.add_traversal(['l1'], runs)  # obs.exps.runs.l1s
    snr = G.add_operation(l1s, [], 'snr')  # obs.exps.runs.l1s.snr
    avg_snr_per_exp = G.add_aggregation(snr, exps, 'avg')  # x = mean(obs.exps.runs.l1s.snr, wrt=exps)
    avg_snr_per_run = G.add_aggregation(snr, runs, 'avg')  # y = mean(obs.exps.runs.l1s.snr, wrt=runs)

    exp_above_1 = G.add_aggregation(G.add_operation(avg_snr_per_exp, [], '> 1'), exps, 'single')  # x > 1
    run_above_1 = G.add_aggregation(G.add_operation(avg_snr_per_run, [], '> 1'), runs, 'single')  # y > 1
    l1_above_1 = G.add_aggregation(G.add_operation(snr, [], '> 1'), l1s, 'single')  # obs.exps.runs.l1s.snr > 1

    # cond = (x > 1) & (y > 1) & (obs.exps.runs.l1s.snr > 1)
    condition = G.add_aggregation(G.add_operation(l1s, [l1_above_1, run_above_1, exp_above_1], '&'), l1s, 'single')  # chosen the lowest
    l1s = G.add_filter(l1s, [condition], '')  # obs.exps.runs.l1s[cond]
    result = G.add_traversal(['l2'], l1s)



    # used to use networkx 2.4
    G.export('parser')
    dag = subgraph_view(G.G, excluded_edge_type='wrt')
    backwards = subgraph_view(G.G, only_edge_type='wrt')
    traversal_graph = subgraph_view(dag, excluded_edge_type='dep')
    dep_graph = subgraph_view(G.G, only_edge_type='dep')
    plot_graph(traversal_graph).render('parser-traversal')


    ordering = []
    import time
    start_time = time.perf_counter()
    for n in G.traverse_query():
        end_time = time.perf_counter()
        print(G.G.nodes[n]["i"])
        ordering.append(n)
    verify_traversal(G.G, ordering)
    print(end_time - start_time)

    # ordering = [1, 2, 3, 4, 1, 2, 3, 4, 5, 3, 4, 6, 7]
    ordering = [G.G.nodes[o]['i'] for o in ordering]
    edge_ordering = insert_load_saves(ordering)
    for o in edge_ordering:
        if isinstance(o, Store):
            print(f"Store: {o.state} -> {o.reference}")
        elif isinstance(o, Load):
            print(f"Load: {o.reference} --> {o.state}")
        else:
            print(f"Trav: {o[0]} -> {o[1]}")
    verify_saves(edge_ordering, ordering)
    # edge_ordering = collapse_chains_of_loading(edge_ordering)
    for e in edge_ordering:
        print(e)