'''
Calculates the median duration of a pattern from a record file. This assumes
the initial pattern in each experiment will be that pattern, and discards
experiments where it is not.
'''
# image=ubuntu:viridicle

import argparse
from collections import Counter, defaultdict
import os

import matplotlib.pyplot as plot
from tabulate import tabulate

import utils


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, help='Path to record')
    parser.add_argument('out_path', type=str, help='Path to write image to')
    parser.add_argument('pattern', type=str, help='Pattern to analyze')
    parser.add_argument('--allowed-startup-time', type=float, default=5)
    args = parser.parse_args()

    # Load, clean, and aggergate experiments
    experiments = utils.load_record(args.target)
    experiments = {
        key: [utils.clean_experiment(exp, removal_threshold=1000)
              for exp in exps]
        for key, exps in experiments.items()
    }
    utils.aggregate_experiments(experiments, start_at=0)

    durations, censored = defaultdict(list), defaultdict(list)
    failed_startups = Counter()
    widths, mobilities, successors = set(), set(), set()
    n_experiments, n_extinctions = Counter(), Counter()
    transitions = defaultdict(Counter)

    for (_, _, mobility, width), xs in experiments.items():
        widths.add(width)
        mobilities.add(mobility)

        for x in xs:
            pattern = successor = None
            for pat, dur, is_cen in x:
                if pat == 'transitory':
                    continue
                if pattern is None:
                    pattern, duration, is_censored = pat, dur, is_cen
                else:
                    successor = pat
                    break

            if pattern == args.pattern:
                n_experiments[(width, mobility)] += 1
                if successor is not None:
                    transitions[(width, mobility)][successor] += 1
                    successors.add(successor)
                if is_censored:
                    censored[(width, mobility)].append(duration)
                else:
                    durations[(width, mobility)].append(duration)
                if x.absorbed:
                    n_extinctions[(width, mobility)] += 1
            else:
                failed_startups[width] += 1

    widths = sorted(widths)
    mobilities = sorted(mobilities)
    successors = sorted(successors)

    ###################
    # MEDIAN DURATION #
    ###################

    plot.figure(figsize=(6, 6))
    plot.subplot(1, 1, 1).set_ylim((0, 1000))
    plot.title(args.pattern)
    plot.xlabel('Mobility')
    plot.ylabel('Duration')

    for width in widths:
        medians, ci_up, ci_low = [], [], []
        for mobility in mobilities:
            m_d, (c_u, c_l) = utils.km_estimator(
                durations[(width, mobility)],
                censored[(width, mobility)],
            )
            medians.append(m_d)
            ci_up.append(c_u)
            ci_low.append(c_l)

        plot.plot(
            mobilities,
            medians,
            label=f'{width}x{width}',
            color=utils.WIDTH_COLORS[width],
        )
        plot.fill_between(
            mobilities,
            ci_up,
            ci_low,
            alpha=0.3,
            color=utils.WIDTH_COLORS[width],
        )

    plot.legend()

    out_path = os.path.join(args.out_path, f'durations-{args.pattern}.png')
    plot.savefig(out_path)
    plot.close()
