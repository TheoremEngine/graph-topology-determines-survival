'''
Analyzes the data files for the experiments with block initializations,
generating the survival and extinction curves. Optionally also generates the
seeds for the second set of experiments.
'''
# image=ubuntu:viridicle

import argparse
from collections import Counter, defaultdict
import csv
from itertools import chain
import math
import os
import sys

import matplotlib.pyplot as plot
import numpy as np

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='Path to record')
    parser.add_argument('out_path', type=str, help='Path to write images')
    parser.add_argument('--warmup-duration', type=float, default=200,
                        help='Assumed system warmup period')
    parser.add_argument('--no-marching-bands', action='store_true',
                        help='Exclude experiments that entered marching bands '
                             'pattern')
    parser.add_argument('--make-seeds', action='store_true',
                        help='Generate seeds for second round of experiments')
    args = parser.parse_args()

    os.makedirs(args.out_path, exist_ok=True)

    # Load experiments
    experiments = utils.load_record(args.target)
    # Remove transitory fluctuations
    experiments = {
        key: [utils.clean_experiment(exp) for exp in exps]
        for key, exps in experiments.items()
    }
    # Aggregate results. Besides generating the returned values, this also
    # attaches extra metadata to each experiment.
    durations, censored, transitions = utils.aggregate_experiments(
        experiments, start_at=args.warmup_duration
    )

    # Remove marching bands patterns if desired.
    if args.no_marching_bands:
        experiments = {
            key: [x for x in xs if not x.marching_bands]
            for key, xs in experiments.items()
        }
        # Rerun aggregation.
        durations, censored, transitions = utils.aggregate_experiments(
            experiments, start_at=args.warmup_duration
        )

    ###################
    # SURVIVAL CURVES #
    ###################

    # per_abs: This will be a two-layer nested dictionary. The outer key will
    # be (graph, width). The inner key will be mobility. The value will be a
    # numpy.ndarray, where value[t] gives the percentage of experiments that
    # have absorbed by time t.
    per_abs = defaultdict(dict)
    # Collect these so we have a complete list of them to iterate over later.
    # ms is short for mobilities.
    graphs, widths, ms = set(), set(), set()

    for (graph, initialization, m, width), exps in experiments.items():
        graphs.add(graph)
        widths.add(width)
        ms.add(m)
        # Construct the value array. _per_abs[t] will eventually give the
        # percent of experiments that have absorbed by time t, but at first it
        # just contains the count.
        _per_abs = np.zeros(1000, dtype=np.int64)
        for exp in exps:
            if exp.time_to_absorption is not None:
                _per_abs[math.ceil(exp.time_to_absorption):] += 1
        # Convert counts to percents and store in per_abs.
        per_abs[(graph, width)][m] = _per_abs.astype(np.float64) / len(exps)

    graphs, widths, ms = sorted(graphs), sorted(widths), sorted(ms)
    # We only plot survival curves for the largest width.
    width = max(widths)

    # Construct the actual plots. We split these up into separate png files for
    # ease in formatting the TeX.

    for graph in graphs:
        plot.figure(figsize=(6, 6))
        ax = plot.subplot()
        ax.set_title(graph)
        ax.set_ylim((0, 1))
        plot.xlabel('Time')
        plot.ylabel('Probability of Extinction')
        for m in ms:
            if m not in per_abs[(graph, width)]:
                continue
            ax.plot(
                np.arange(len(per_abs[(graph, width)][m])),
                per_abs[(graph, width)][m],
                color=utils.MOBILITY_CMAP.to_rgba(m),
                label=f'{m:.5f}'
            )

        plot.tight_layout()
        path = os.path.join(args.out_path, f'survival_{graph}.png')
        plot.savefig(path)
        plot.close()

    # Save the colorbar.
    plot.figure(figsize=(2, 6))
    ax = plot.subplot()
    cbar = plot.colorbar(utils.MOBILITY_CMAP, orientation='vertical', cax=ax)
    cbar.set_label('Mobility')
    plot.tight_layout()
    path = os.path.join(args.out_path, 'survival_colorbar.png')
    plot.savefig(path)
    plot.close()

    '''
    n_g, width = len(graphs), max(widths)
    plot.figure(figsize=(6 * n_g, 6))
    width_ratios = ([4] * n_g) + [0.2]
    gs = matplotlib.gridspec.GridSpec(1, n_g + 1, width_ratios=width_ratios)

    for graph, _gs in zip(graphs, gs):
        ax = plot.subplot(_gs)
        ax.set_title(graph)
        ax.set_ylim((0, 1))
        plot.xlabel('Time')
        plot.ylabel('Probability of Extinction')

        for m in ms:
            if m not in per_abs[(graph, width)]:
                continue
            ax.plot(
                np.arange(len(per_abs[(graph, width)][m])),
                per_abs[(graph, width)][m],
                color=utils.MOBILITY_CMAP.to_rgba(m),
                label=f'{m:.5f}'
            )

    ax = plot.subplot(gs[-1])
    cbar = plot.colorbar(utils.MOBILITY_CMAP, orientation='vertical', cax=ax)
    cbar.set_label('Mobility')

    plot.tight_layout()
    path = os.path.join(args.out_path, 'survival.png')
    plot.savefig(path)
    plot.close()
    '''

    print('Made survival curves.')

    #####################
    # EXTINCTION CURVES #
    #####################

    for graph in graphs:
        plot.figure(figsize=(6, 6))
        ax = plot.subplot()
        ax.set_title(graph)
        ax.set_ylim((0, 1))
        plot.ylabel('Probability of Extinction')
        plot.xlabel('Mobility')

        for width in widths:
            _ms, _pas = zip(*sorted(per_abs[(graph, width)].items()))
            plot.plot(
                _ms,
                [p[-1] for p in _pas],
                utils.WIDTH_STYLES[width],
                color=utils.WIDTH_COLORS[width],
                label=f'{width}x{width}'
            )

        # Add vertical line denoting critical mobility threshold
        dieout = [[per_abs[(graph, w)][m][-1] for m in ms] for w in widths]
        dieout = ms[np.where((np.array(dieout) >= 0.99).all(0))[0][0]]
        plot.axvline(x=dieout)
        plot.text(
            dieout + 1e-5, 0.5, f'{dieout:.5f}', fontsize='medium'
        )

        if graph == 'sphere':
            plot.legend()

        plot.tight_layout()

        path = os.path.join(args.out_path, f'extinction_{graph}.png')
        plot.savefig(path)
        plot.close()

    print('Made extinction curves.')

    #################
    # PATTERN SEEDS #
    #################

    # If we're not making the seeds, then we're done.
    if not args.make_seeds:
        sys.exit(0)

    # Begin by finding the stable patterns. We look for patterns that:
    #
    # 1. Last for longer than 100 time.
    # 2. Are not partially absorbed.
    # 3. At mobility >= 2.5e-4.

    # stable_patterns will be a set of (graph, pattern) pairs.
    stable_patterns, max_w = set(), max(widths)

    for graph in graphs:
        for m in ms:
            # Skip mobilities less than 2.5e-4.
            if m < 0.00025:
                continue
            key = (graph, m, max_w)
            # We want to go through all patterns, both those that were censored
            # and those that were not.
            patterns = chain(durations[key].items(), censored[key].items())
            for pattern, times in patterns:
                if not pattern.startswith('0'):
                    continue
                if times and (max(times) >= 100):
                    stable_patterns.add((graph, pattern))

    # Okay, we've figured out what patterns we're going to use. Now we're going
    # to collect their initializations.

    # pattern_seeds will have (graph, pattern) keys. The value will be a list
    # of initializations, given as (initialization, mobility, width, random
    # seed, time) tuples.
    pattern_seeds = defaultdict(list)
    # We'll also track the highest mobility we've seen a particular pattern at.
    highest_stable_mobility = defaultdict(float)

    for (graph, init, m, width), exps in experiments.items():
        for experiment in exps:
            elapsed_time = 0.0
            for pat, duration, _ in experiment:
                if ((graph, pat) in stable_patterns) and (duration >= 100):
                    pattern_seeds[(graph, pat)].append(
                        (init, m, width, experiment.seed, elapsed_time)
                    )
                    highest_stable_mobility[(graph, pat, width)] = max(
                        highest_stable_mobility[(graph, pat, width)], m
                    )
                elapsed_time += duration

    # Now we save the seeds as a csv.
    record_path = os.path.join(args.out_path, 'pattern_seeds.csv')
    fieldnames = ['graph', 'pattern', 'initialization', 'mobility', 'width',
                  'seed', 'elapsed_time']
    seed_counts = Counter()
    with open(record_path, 'w', newline='') as record_file:
        writer = csv.DictWriter(record_file, fieldnames=fieldnames)
        writer.writeheader()

        for (graph, pattern), seeds in pattern_seeds.items():
            for (init, m, width, seed, elapsed_time) in seeds:
                writer.writerow({
                    'graph': graph, 'pattern': pattern, 'initialization': init,
                    'mobility': m, 'width': width, 'seed': seed,
                    'elapsed_time': elapsed_time
                })
                seed_counts[(graph, pattern, width)] += 1

    print('Found long-term stable patterns.')
