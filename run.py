#! /usr/bin/python3

'''
This script performs the experiments in the paper "Graph Topology Controls
Survival in the May-Leonard Model". It is used for generating data that will
the be analyzed by the analysis scripts. It expects to be run inside the
examples folder of the Viridicle repository, so that it has access to the utils
and graphs scripts in that folder. It extends the various
run_*_may_and_leonard.py scripts by adding topological data analysis code.
'''

# name=Viridicle
# image=viridicle
# size=c6g.4xlarge
# maximum_runtime=24:00:00


# Total runtime on an AWS EC2 m6g.4xlarge is approximately 1050 sec per 16
# experiments with width 256, mu 60.0. Approximate cost is $1.83 per 256
# experiments.

import argparse
import csv
import io
import multiprocessing as mp
import os
import sys
from typing import Dict, List
import uuid

import numpy as np
from tabulate import tabulate
import viridicle
from viridicle._C import cluster_geo, grow_clusters, merge_small

import graphs
import utils

# Cloudmap is part of our internal tooling. It's used for uploading results to
# S3 buckets.

try:
    import cloudmap
    has_cloudmap = True
except ImportError:
    has_cloudmap = False


FIELDNAMES = [
    'graph', 'pattern', 'duration', 'elapsed_time', 'censored', 'width',
    'mobility', 'absorbed', 'initialization', 'seed', 'warmup_mobility',
    'warmup_seed', 'warmup_time',
]
GRAPHS = {
    'klein': graphs.KleinGeography,
    'torus': viridicle.LatticeGeography,
    'sphere': graphs.SphericalGeography,
}


def clean(sites: np.ndarray, **params) -> np.ndarray:
    '''
    Cleans the record set to be suitable for calculating the Betti numbers.
    This merges small clusters into larger clusters so the fundamental topology
    can be calculated. Specifically, we define a "cluster" as a maximal
    connected subgraph where every vertex has the same state.

    Args:
        sites (:class:`numpy.ndarray`): The returned record of site states,
        with the first dimension indexing the time slice.
        params (dict): Parameters from Geography.encode that are needed to be
        passed to the C layer to encode the graph structure.

    Returns:
        The sites array after cleaning.
    '''
    n, _, w = sites.shape

    for i in range(n):
        merge_small(
            sites=sites[i, ...],
            min_size=int(w**2 // 256),
            merge_size=0,
            empty_state=255,
            **params
        )
        grow_clusters(
            sites=sites[i, ...],
            num_steps=(w // 8),
            empty_state=255,
            **params
        )

    return sites


def get_betti_numbers(sites: np.ndarray, graph: str, **params) -> \
        (np.ndarray, np.ndarray):
    '''
    Calculates the 0th and 1st Betti numbers of the sites as they evolve over
    time.

    Args:
        sites (:class:`numpy.ndarray`): The returned record of site states,
        with the first dimension indexing the time slice.
        graph (str): The graph type.
        params (dict): Parameters from Geography.encode that are needed to be
        passed to the C layer to encode the graph structure.

    Returns:
        A pair of :class:`numpy.ndarray`s, with the first containing the 0th
        Betti numbers and the second containing the 1st Betti numbers.
    '''
    n, h, w = sites.shape

    if graph == 'torus':
        get_chi = get_chi_torus
    elif graph == 'sphere':
        get_chi = get_chi_sphere
    elif graph == 'klein':
        get_chi = get_chi_klein
    else:
        raise ValueError(graph)

    # Calculate b_0.
    b_0 = np.empty((n, 4), dtype=np.int64)
    # We keep clusters as a specified array to stabilize memory useage.
    clusters = np.empty(sites.shape[1:], dtype=np.uint64)
    for i in range(n):
        _, cluster_states = cluster_geo(
            sites=sites[i, ...], **params, out=clusters
        )
        for s in range(4):
            b_0[i, s] = (cluster_states == s).sum()

    b_1 = np.empty((n, 4), dtype=np.int64)

    # The Euler characteristic is:
    #
    # chi = (# Vertices) - (# Edges) + (# Faces)
    #
    # And is equal to:
    #
    # chi = b_0 - b_1 + b_2 - ...
    #
    # All higher Betti numbers are 0 in the graphs we consider, since they
    # are discretizations of 2-dimensional manifolds. Therefore, we obtain:
    #
    # b_1 = b_0 + b_2 - chi

    for s in range(4):
        is_s = (sites == s)
        # b_2: Second Betti number. This is non-zero if and only if the
        # subgraph is non-contractible, which occurs if and only if the entire
        # graph is the same state, and the entire graph is orientable (i.e.,
        # not the Klein bottle) and has no boundary.
        if graph in {'sphere', 'torus'}:
            b_2 = is_s.all((1, 2)).astype(int)
        elif graph == 'klein':
            b_2 = 0
        chi = get_chi(is_s)
        b_1[:, s] = b_2 + b_0[:, s] - chi

    return b_0, b_1


def get_chi_klein(is_s: np.ndarray) -> np.ndarray:
    '''
    Calculates the Euler characteristic of a subgraph of the Klein bottle.

    Args:
        is_s (:class:`numpy.ndarray`): A boolean graph giving a mask of the
        subgraph as it changed over time, with the 0th dimension indexing the
        time.

    Returns:
        A :class:`numpy.ndarray` giving the Euler characteristic over time.
    '''
    # We need to be careful here to keep memory consumption under control.
    # Hence the use of in-place operations wherever possible.

    # n_v: Number of vertices.
    n_v = is_s.sum((1, 2))

    # e_x: Edge to the right of the vertex is part of the subgraph. We find
    # this in two stages: first the interior, then the right edge where it
    # wraps around. We construct this as an array so we can reuse it for
    # finding the faces.
    e_x = is_s.copy()
    e_x[:, :, :-1] &= is_s[:, :, 1:]
    e_x[:, :, -1] &= is_s[:, ::-1, 0]

    # n_e: Number of edges.
    n_e = e_x.sum((1, 2))
    # Add the vertical edges to n_e.
    n_e += (is_s[:, :-1, :] & is_s[:, 1:, :]).sum((1, 2))
    n_e += (is_s[:, -1, :] & is_s[:, 0, :]).sum(1)

    # n_f: Number of faces. A horizontal edge above a second horizontal edge
    # implies a face.
    n_f = (e_x[:, :-1, :] & e_x[:, 1:, :]).sum((1, 2))
    n_f += (e_x[:, -1, :] & e_x[:, 0, :]).sum(1)

    return n_v - n_e + n_f


def get_chi_sphere(is_s: np.ndarray) -> np.ndarray:
    '''
    Calculates the Euler characteristic of a subgraph of the sphere.

    Args:
        is_s (:class:`numpy.ndarray`): A boolean graph giving a mask of the
        subgraph as it changed over time, with the 0th dimension indexing the
        time.

    Returns:
        A :class:`numpy.ndarray` giving the Euler characteristic over time.
    '''
    n, h, w = is_s.shape
    # n_v: Number of vertices that are state s.
    n_v = is_s.sum((1, 2))

    # e_x: Edge to the right of the vertex is part of the subgraph. We find
    # this in two stages: first the interior, then the right edge where it
    # wraps around. We construct this as an array so we can reuse it for
    # finding the faces.
    e_x = is_s.copy()
    e_x[:, :, :-1] &= is_s[:, :, 1:]
    e_x[:, :, -1] &= is_s[:, :, 0]

    # We begin by calculating the number of interior edges and faces, then
    # updating them for the harder, but memory-light polar regions.

    # n_e: Number of edges.
    n_e = e_x.sum((1, 2)) + (is_s[:, :-1, :] & is_s[:, 1:, :]).sum((1, 2))
    # n_f: Number of faces.
    n_f = (e_x[:, 1:, :] & e_x[:, :-1, :]).sum((1, 2))

    # Now we do the polar regions.

    # *_pole: Restrict vertices to polar vertices for syntactical convenience.
    n_pole, s_pole = is_s[:, 0, :], is_s[:, -1, :]
    # Radius of the pole, giving the distance from the edge to the point in the
    # center where the fold begins.
    r_x = w // 2
    # e_y_n: Vertical edges on north pole. Note there are only r_x of these,
    # not w.
    e_y_n = (n_pole[:, :r_x] & n_pole[:, :(r_x - 1):-1])
    # e_y_s: Vertical edges on south pole. Note there are only r_x of these,
    # not w.
    e_y_s = (s_pole[:, :r_x] & s_pole[:, :(r_x - 1):-1])
    # Add polar edges to n_e. n_e is now finished.
    n_e += e_y_n.sum(1) + e_y_s.sum(1)
    # Add face on the north pole. Does not include the folded faces at the
    # center.
    n_f += (e_y_n[:, :-1] & e_y_n[:, 1:]).sum(1)
    # Now on the south pole.
    n_f += (e_y_s[:, :-1] & e_y_s[:, 1:]).sum(1)

    # Now the folded faces.
    n_f += (e_y_n[:, 0].astype(np.int64) + e_y_n[:, -1].astype(np.int64))
    n_f += (e_y_s[:, 0].astype(np.int64) + e_y_s[:, -1].astype(np.int64))

    return n_v - n_e + n_f


def get_chi_torus(is_s: np.ndarray) -> np.ndarray:
    '''
    Calculates the Euler characteristic of a subgraph of the torus.

    Args:
        is_s (:class:`numpy.ndarray`): A boolean graph giving a mask of the
        subgraph as it changed over time, with the 0th dimension indexing the
        time.

    Returns:
        A :class:`numpy.ndarray` giving the Euler characteristic over time.
    '''
    # We need to be careful here to keep memory consumption under control.
    # Hence the use of in-place operations wherever possible.

    # n_v: Number of vertices.
    n_v = is_s.sum((1, 2))

    # e_x: Edge to the right of the vertex is part of the subgraph. We find
    # this in two stages: first the interior, then the right edge where it
    # wraps around. We construct this as an array so we can reuse it for
    # finding the faces.
    e_x = is_s.copy()
    e_x[:, :, :-1] &= is_s[:, :, 1:]
    e_x[:, :, -1] &= is_s[:, :, 0]

    # n_e: Number of edges.
    n_e = e_x.sum((1, 2))
    # Add the vertical edges to n_e.
    n_e += (is_s[:, :-1, :] & is_s[:, 1:, :]).sum((1, 2))
    n_e += (is_s[:, -1, :] & is_s[:, 0, :]).sum(1)

    # n_f: Number of faces. A horizontal edge above a second horizontal edge
    # implies a face.
    n_f = (e_x[:, :-1, :] & e_x[:, 1:, :]).sum((1, 2))
    n_f += (e_x[:, -1, :] & e_x[:, 0, :]).sum(1)

    return n_v - n_e + n_f


def get_switches(b_0: np.ndarray, b_1: np.ndarray, absorbed: np.ndarray) \
        -> (np.ndarray, np.ndarray, np.ndarray):
    '''
    Finds switches between different patterns.

    Args:
        b_0 (:class:`numpy.ndarray`): The record of 0th Betti numbers over the
        course of the run.
        b_1 (:class:`numpy.ndarray`): The record of the 1st Betti numbers over
        the course of the run.
        absorbed (:class:`numpy.ndarray`): The record of whether the graph is
        absorbed yet over the course of the run.

    Returns:
        A tripl of :class:`numpy.ndarray`. The first gives the indices of the
        switches, with each entry giving the index of the first entry of the
        Betti numbers with the new pattern. The second gives the pattern code
        for each switch. The third gives whether the system is absorbed at that
        switch.
    '''
    # Sort b_0, b_1 so that 001 and 010 are equivalent, since we want this to
    # be invariant to permutation of states.
    codes = (b_1[:, 1:] * 16) + b_0[:, 1:]
    idxs = codes.argsort(axis=1) + 1
    n = codes.shape[0]
    b_0[:, 1:] = b_0[np.arange(n).repeat(3), idxs.flatten()].reshape(n, 3)
    b_1[:, 1:] = b_1[np.arange(n).repeat(3), idxs.flatten()].reshape(n, 3)

    # bs concatenates b_0, b_1, and absorbed for simplicity's sake
    bs = np.concatenate((b_0, b_1, absorbed.reshape(-1, 1)), axis=1)

    # Find places where the system switches between Betti numbers. Each switch
    # marks the index BEFORE the switch at this point.
    switches, = np.where((bs[:-1, :] != bs[1:, :]).any(1))
    # Increment so they mark the START.
    switches += 1
    # Add 0 to the front and n to the back.
    switches = np.concatenate(([0], switches))

    return switches, bs[switches, :-1], bs[switches, -1]


def worker(args) -> List[Dict]:
    '''
    This runs the actual experiment and then performs analysis on it.
    '''
    def initialize(init, rng):
        if init == 'random':
            return rng.integers(
                low=0, high=4, size=(args.width, args.width), dtype=np.uint8
            )
        elif init == 'block':
            return utils.get_block_initialization(
                4, (args.width, args.width), rng
            )
        else:
            raise ValueError(args.initialization)

    if args.warmup_seed is not None:
        rng = np.random.default_rng(args.warmup_seed)
        sites = initialize(args.initialization, rng)
        diffusion_rate = 2 * args.warmup_mobility * args.width**2
        rules = viridicle.may_leonard_rules(3, 1, diffusion_rate)
        geo = GRAPHS[args.graph](sites, rules, rng)

        geo.run(
            elapsed_time=args.warmup_time,
            return_sites=False,
            return_counts=False,
        )

        diffusion_rate = 2 * args.mobility * args.width**2
        geo.rules = viridicle.may_leonard_rules(3, 1, diffusion_rate)
        geo.generator = np.random.default_rng(args.seed)

    else:
        rng = np.random.default_rng(args.seed)
        sites = initialize(args.initialization, rng)
        diffusion_rate = 2 * args.mobility * args.width**2
        rules = viridicle.may_leonard_rules(3, 1, diffusion_rate)
        geo = GRAPHS[args.graph](sites, rules, rng)

    b_0s, b_1s, absorbeds = [], [], []
    # How long to run between checking for absorption.
    delta = 100

    params = geo.encode()

    # Periodically check for absorption, to avoid running the experiment after
    # collapse has occurred. Also improves memory consumption, since the Betti
    # number calculation generates some intermediate products.
    for t in np.arange(0, args.elapsed_time, delta):
        run_time = min(args.elapsed_time - t, delta)

        params['sites'] = geo.run(
            elapsed_time=run_time,
            report_every=args.report_every,
            return_sites=True,
            return_counts=False,
        )
        # The first entry in sites will be the state of the system before
        # anything has happened. We discard that value since otherwise our
        # entries would repeat.
        params['sites'] = params['sites'][1:, ...]
        n = params['sites'].shape[0]

        # Check if absorption has occurred. Note the importance of doing this
        # BEFORE the cleaning operation, since a rarefied state may be wiped
        # out completely by cleaning.
        absorbed = (params['sites'] != 1).all((1, 2))
        absorbed |= (params['sites'] != 2).all((1, 2))
        absorbed |= (params['sites'] != 3).all((1, 2))
        absorbeds.append(absorbed)

        # Perform cleaning. This is necessary before calculating the Betti
        # numbers.
        params['sites'] = clean(**params)

        # Calculate Betti numbers.
        b_0, b_1 = get_betti_numbers(graph=args.graph, **params)
        b_0s.append(b_0)
        b_1s.append(b_1)

        # If absorption has occurred, we're done.
        if absorbed.any():
            break

    b_0 = np.concatenate(b_0s)
    b_1 = np.concatenate(b_1s)
    absorbed = np.concatenate(absorbeds)
    n = absorbed.shape[0]

    # Find places where the system switches state. Note that this will modify
    # b_0, b_1 in-place to make them invariant to permutation of the non-empty
    # states.
    switches, patterns, absorbed = get_switches(b_0, b_1, absorbed)

    durations = np.concatenate((
        switches[1:] - switches[:-1], [n - switches[-1]]
    ))
    durations = durations * args.report_every
    censored = np.array(([False] * (len(switches) - 1)) + [True])

    rtn_args = {k: v for k, v in vars(args).items() if k in FIELDNAMES}
    patterns = [''.join(map(str, pattern)) for pattern in patterns]
    n = len(patterns)

    rtn, elapsed_time = [], 0.0
    for i, (pattern, duration) in enumerate(zip(patterns, durations)):
        rtn.append({
            'pattern': str(pattern),
            'duration': str(duration),
            'censored': str(censored[i]),
            'absorbed': str(absorbed[i]),
            'elapsed_time': str(elapsed_time),
            **rtn_args
        })
        elapsed_time += duration

    return rtn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-experiments', type=int, default=1,
                        help='Number of experiments to run')
    parser.add_argument('--width', type=int, default=256,
                        help='Width of lattice')
    parser.add_argument('--num-workers', type=int, default=1,
                        help='Number of worker processes')
    parser.add_argument('--random-seeds', type=int, nargs='+', default=None,
                        help='Random seeds to use - defaults to 0, 1, ...')
    parser.add_argument('--report-every', type=float, default=1.0,
                        help='How often to take a snapshot')
    parser.add_argument('--graph', type=str, default='torus',
                        choices=['klein', 'torus', 'sphere'],
                        help='Graph structure to use')
    parser.add_argument('--initialization', type=str, default='random',
                        choices=['block', 'random'],
                        help='Initialization of the graph state for warmup')
    parser.add_argument('--mobility', type=float, default=5e-4,
                        help='Mobility')
    parser.add_argument('--warmup-url', type=str, default=None,
                        help='URL to retrieve warmup values from')
    parser.add_argument('--warmup-seeds', type=int, nargs='+', default=None,
                        help='Random seeds to use in warmup')
    parser.add_argument('--warmup-times', type=float, nargs='+', default=None,
                        help='How long to run the system in warmup')
    parser.add_argument('--warmup-mobilities', type=float, default=None,
                        nargs='+',
                        help='Mobility during warmup period')
    parser.add_argument('--bucket', type=str, default=None,
                        help='Bucket to upload results to')
    parser.add_argument('--elapsed-time', type=float, default=1000,
                        help='Number of generations to run for')
    parser.add_argument('--output-path', type=str, default='.',
                        help='Path to write output file to')
    parser.add_argument('--label', type=str, default=None,
                        help='Label to use in record file')
    args = parser.parse_args()

    print(tabulate(vars(args).items()))

    if args.warmup_url:
        driver = cloudmap.get_storage_driver()
        warmup_csv = cloudmap.retrieve_to_bytes(driver, args.warmup_url)
        warmup_csv = io.StringIO(warmup_csv.decode('utf-8'))
        args.warmup_mobilities, args.warmup_seeds, args.warmup_times = zip(
            *csv.reader(warmup_csv)
        )
        args.warmup_mobilities = [float(m) for m in args.warmup_mobilities]
        args.warmup_seeds = [int(s) for s in args.warmup_seeds]
        args.warmup_times = [float(t) for t in args.warmup_times]
        _, remote_path = args.warmup_url.split('://')
        container, remote_path = remote_path.split('/', 1)
        container = driver.get_container(container)
        container.get_object(remote_path).delete()

    if args.bucket and not has_cloudmap:
        print('Cloudmap is required for pushing results to bucket.')
        sys.exit(1)

    if args.random_seeds is None:
        args.random_seeds = list(range(args.num_experiments))
    elif len(args.random_seeds) == 1:
        start, = args.random_seeds
        args.random_seeds = list(range(start, start + args.num_experiments))

    if args.num_experiments is None:
        args.num_experiments = len(args.random_seeds)
    elif len(args.random_seeds) != args.num_experiments:
        print(
            f'Disagreement between number of experiments '
            f'{args.num_experiments} and number of random seeds '
            f'{args.random_seeds}'
        )
        sys.exit(1)

    warmup = (args.warmup_mobilities, args.warmup_seeds, args.warmup_times)
    if all(x is None for x in warmup):
        args.warmup_mobilities = [None] * args.num_experiments
        args.warmup_seeds = [None] * args.num_experiments
        args.warmup_times = [None] * args.num_experiments
    elif all(x is not None for x in warmup):
        if any(len(x) != args.num_experiments for x in warmup):
            print(
                'Disagreement between number of experiments and number of '
                'warmup parameters.'
            )
            sys.exit(1)
    else:
        print('Either all or no warmup parameters should be specified.')
        sys.exit(1)

    def _add_seed(args, seed, warmup_seed, warmup_time, warmup_mobility):
        args.seed, args.warmup_seed = seed, warmup_seed
        args.warmup_time, args.warmup_mobility = warmup_time, warmup_mobility
        return args
    experiments = (
        _add_seed(args, s, i, t, m) for s, i, t, m in
        zip(args.random_seeds, args.warmup_seeds, args.warmup_times,
            args.warmup_mobilities)
    )

    if args.label is not None:
        out_name = (args.graph, str(args.width), args.label,
                    str(args.mobility), f'{uuid.uuid1()}.csv')
    else:
        out_name = (args.graph, str(args.width), str(args.mobility),
                    f'{uuid.uuid1()}.csv')
    out_name = '-'.join(out_name)
    out_path = os.path.join(args.output_path, out_name)

    with open(out_path, 'w', newline='') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        if args.num_workers > 1:
            with mp.Pool(args.num_workers) as pool:
                for result in pool.imap_unordered(worker, experiments):
                    for row in result:
                        writer.writerow(row)
        else:
            for result in map(worker, experiments):
                for row in result:
                    writer.writerow(row)

    if args.bucket is not None:
        # Upload results to S3 bucket.
        driver = cloudmap.cloud.get_storage_driver()
        container = driver.get_container(args.bucket)
        driver.upload_object(out_path, container, out_name)
