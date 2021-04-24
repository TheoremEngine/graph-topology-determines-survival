'''
Provides utility functions used by analysis scripts.
'''
from collections import Counter, defaultdict
import csv
from itertools import chain
from typing import Callable, Dict, List, Sequence, Tuple

import matplotlib
import matplotlib.cm
import matplotlib.pyplot as plot
import numpy as np
from PIL import Image


# Set default font size in graphs for all scripts.
plot.rc('font', size=18)


def get_colormap(num_states: int = 3, include_empty: bool = True,
                 zero_is_clear: bool = False) -> matplotlib.colors.Colormap:
    '''
    Generates viridis color map for making pretty images.

    Args:
        num_states (int): Number of system states.
        include_empty (bool): Treat the zero state as an "empty" state whose
        color is white or transperant.
        zero_is_clear (bool): Render white as transperant.

    Returns:
        A :class:`matplotlib.colors.Colormap`.
    '''
    num_states -= include_empty
    cmap = matplotlib.cm.get_cmap('viridis', num_states)
    colors = cmap(np.linspace(0, 1, num_states))
    if include_empty:
        a = 0 if zero_is_clear else 1
        colors = np.concatenate((np.array([[1, 1, 1, a]]), colors))
    return matplotlib.colors.ListedColormap(colors)


cmap = get_colormap(4, include_empty=True)
SPECIES_COLORS = {
    0: cmap(0),
    1: cmap(1),
    2: cmap(2),
    3: cmap(3),
    255: cmap(0),
}
GRAPH_COLORS = {
    'klein': 'b',
    'sphere': 'g',
    'torus': 'r',
}
cmap = get_colormap(5, include_empty=True)
WIDTH_COLORS = {
    32: 'y',
    64: 'r',
    128: 'g',
    256: 'b',
}

MOBILITY_CMAP = matplotlib.cm.ScalarMappable(
    norm=matplotlib.colors.Normalize(1e-4, 8e-4),
    cmap=matplotlib.cm.get_cmap('viridis').reversed(),
)


class Experiment(list):
    '''
    This will be a list of triples, where each triple is a (string, float,
    bool). The string will be the pattern, the float will be the duration, and
    the bool will be whether the pattern was censored, i.e. the system was
    still in that state when the experiment ended. We subclass list so we can
    attach other metadata as attributes, e.g. the random seed.
    '''
    def __init__(self, key, seed, *args):
        self.key, self.seed = key, seed
        super().__init__(*args)


def aggregate_experiments(experiments: Dict[Tuple, List[Experiment]],
                          start_at: float = 0.0):
    '''
    Calculates aggregation values for a set of experiments. Also updates the
    Experiments to attach additional attributes to them.
    '''
    def is_stable(pattern: str) -> bool:
        '''
        Convenience function for checking if a pattern is stable.
        '''
        try:
            return int(pattern[0]) == 0
        except ValueError:
            return pattern != 'transitory'

    # transitions: Table tracking transitions between different patterns. First
    # key is (graph, mobility), second key is the pattern, values is a Counter
    # of what that pattern transitions to.
    transitions = defaultdict(lambda: defaultdict(Counter))
    # pattern_durations: Tracks how long different patterns last. First key
    # will be (graph, mobility), second key is the pattern, values is a list of
    # durations.
    pattern_durations = defaultdict(lambda: defaultdict(list))
    # censored_durations: Tracks pattern durations that were censored. First
    # key will be (graph, mobility), second key is the pattern, values is a
    # list of durations.
    censored_durations = defaultdict(lambda: defaultdict(list))

    for (graph, init, mob, width), _experiments in experiments.items():
        for experiment in _experiments:
            # elapsed: How much time has passed in the current experiment.
            # stabilized: Whether the system has stabilized yet.
            # last_pattern: The last pattern we've seen.
            # partial: Whether the system has gone through partial
            # stabilization yet.
            elapsed, stabilized, last, partial = 0, False, None, False
            # Set null values
            experiment.first_stable_pattern = None
            experiment.time_to_absorption = None
            experiment.time_to_partial_absorption = None
            experiment.marching_bands = False
            experiment.absorbed = False
            experiment.last_pattern = 'transitory'

            # Go through patterns in the experiment.
            for pattern, duration, censored in experiment:
                # If system has not yet stabilized, check if it's stabilized.
                if not stabilized and is_stable(pattern):
                    stabilized = True
                    experiment.first_stable_pattern = pattern

                # Also check if it's partially absorbed.
                if (not partial) and (pattern in {'partial', 'absorbed'}):
                    experiment.time_to_partial_absorption = elapsed
                    partial = True

                # Record stable patterns
                if stabilized and (pattern != 'transitory'):
                    if pattern == '01110111':
                        experiment.marching_bands = True
                    last = pattern

                    if (elapsed + duration) >= start_at:
                        if last is not None:
                            transitions[(graph, mob, width)][last][pattern] += 1
                        if pattern != 'absorbed':
                            if censored:
                                censored_durations[(graph, mob, width)][pattern].append(
                                    duration
                                )
                                # This ensures that the pattern is present as a
                                # key in pattern_durations.
                                pattern_durations[(graph, mob, width)][pattern]
                            else:
                                pattern_durations[(graph, mob, width)][pattern].append(
                                    duration
                                )

                # Check if the system absorbed
                if pattern == 'absorbed':
                    experiment.time_to_absorption = elapsed
                    experiment.absorbed = True

                if pattern != 'transitory':
                    experiment.last_pattern = pattern

                elapsed += duration

    return pattern_durations, censored_durations, transitions


def clean_experiment(experiment: Experiment, transitory_threshold: float = 5.0,
                     removal_threshold: float = 5.0) -> Experiment:
    '''
    Cleans an experiment to make it suitable for processing. Specifically:

     * Reclassify patterns as 'transitory' if their duration falls below a
       minimum threshold.
     * Reclassify patterns as 'partial' if they are partially absorbed, i.e.,
       the 0th Betti number of a non-zero state is zero.
     * Merge any succeeding patterns of the same type.
     * Remove any transitory fluctuations of duraton below a minimum threshold.

    :param transitory_threshold: Maximum duration to consider a pattern
    transitory.
    :type transitory_threshold: float
    :param removal_threshold: Maximum duration to remove a transitory pattern
    if it is preceded and followed by the same pattern.
    :type removal_threshold: float

    :return: The cleaned :class:`Experiment`
    :rtype: :class:`Experiment`
    '''
    # First, reclassify transitory and partially absorbed patterns.
    def reclassify(row):
        pattern, duration, censored = row
        if pattern == 'absorbed':
            return ('absorbed', 0.0, True)
        elif duration < transitory_threshold:
            return ('transitory', duration, censored)
        elif (pattern[0] == '0') and ('0' in pattern[1:4]):
            return ('partial', duration, censored)
        return row

    experiment = Experiment(
        experiment.key,
        experiment.seed,
        [reclassify(row) for row in experiment],
    )

    # Now that we've done this reclassifying, collapse any neighboring patterns
    # of the same type into single patterns.
    new_experiment = Experiment(
        experiment.key,
        experiment.seed,
        experiment[0:1],
    )
    for pattern, duration, censored in experiment[1:]:
        if pattern == new_experiment[-1][0]:
            duration += new_experiment.pop(-1)[1]
        new_experiment.append((pattern, duration, censored))
    experiment = new_experiment

    # And remove any momentary transitory patterns.
    new_experiment = Experiment(
        experiment.key,
        experiment.seed,
    )
    for pattern, duration, censored in experiment:
        if (
            (len(new_experiment) >= 2) and
            (new_experiment[-1][1] <= removal_threshold) and
            (new_experiment[-2][0] == pattern) and
            (new_experiment[-1][0] == 'transitory')
        ):
            duration += new_experiment.pop(-1)[1]
            duration += new_experiment.pop(-1)[1]
        new_experiment.append((pattern, duration, censored))

    return new_experiment


def exponential_estimator(durations: Sequence[float],
                          censored: Sequence[float],
                          left_truncation: float = 0) -> float:
    '''
    Given a collection of samples from an exponential distribution, some of
    which are right-censored and all of which are left-truncated, find the MLE
    of the rate of the distribution.

    Args:
        durations (sequence of floats): Samples from the distribution.
        censored (sequence of floats): Samples from the distribution that were
        right-censored.
        left_truncation (float): The truncation value; no samples below this
        value will be observed.
    '''
    m, n = len(durations), len(censored)
    return (sum(durations) + sum(censored) - (m + n) * left_truncation) / m


def find_long_term_patterns(pattern_durations: Dict,
                            censored_durations: Dict,
                            minimum_duration: float = 10.0) -> Dict:
    '''
    Finds the long-term patterns across the experiments.

    :param pattern_durations: A dictionary mapping a (graph, diffusion rate)
    pair to a dictionary whose keys are patterns and whose values are lists of
    durations.
    :type pattern_durations: dict
    :param censored_durations: A dictionary mapping a (graph, diffusion rate)
    pair to a dictionary whose keys are patterns and whose values are lists of
    durations, where the duration is censored, that is, it ends with the end of
    the experiment and not the end of the pattern.
    :type pattern_durations: dict
    :param minimum_duration: How long to require patterns to list to be
    considered long term.
    :type minimum_duration: float

    :return: A dictionary mapping the graph type to a sorted list of patterns
    identified as long-term.
    :rtype: dict
    '''
    long_term_patterns = defaultdict(Counter)
    durations = chain(pattern_durations.items(), censored_durations.items())
    for (graph, mob), pat_dur in durations.items():
        for pattern, durations in pat_dur.items():
            if pattern.startswith('0'):
                n_long = sum(d >= minimum_duration for d in durations)
                long_term_patterns[graph][pattern] += n_long

    return defaultdict(list, {
        g: sorted([p for p, c in pattern_counts.items() if c > 32])
        for g, pattern_counts in long_term_patterns.items()
    })


def get_mean_path_length(h: int, w: int, graph_type: str):
    if graph_type == 'torus':
        x, y = np.meshgrid(np.arange(h), np.arange(w))
        return (np.minimum(x, w - x) + np.minimum(y, h - y)).mean()

    elif graph_type == 'klein':
        x, y = np.meshgrid(np.arange(h), np.arange(w))
        d = np.minimum(
            np.minimum(
                x + y,
                x + h - y,
            ),
            np.minimum(
                w - x + h - 1 - y,
                w - x + y + 1,
            )
        )
        return d.mean()

    elif graph_type == 'sphere':
        if (w % 2 == 1) or (h % 2 == 1):
            raise ValueError('Width and height must be even.')

        # We calculate the mean path length only over the lower left quadrant,
        # since this is sufficient by symmetry. Over this quadrant, the
        # distance from a vertex (x, y) to a vertex (x_p, y_p) anywhere in the
        # graph is given by this function. Note that the function is amenable
        # to accepting arrays as input for x_p, y_p.

        def d(x, y, x_p, y_p):
            x_p, y_p = x_p.reshape(-1, 1), y_p.reshape(1, -1)
            # Four possible routes to any other point.
            # Straight shot
            dist = np.abs(y - y_p) + np.abs(x - x_p)
            # Around the horizontal side
            dist = np.minimum(dist, np.abs(y - y_p) + np.abs(w - x_p + x))
            # Over the south pole
            dist = np.minimum(dist, y + 1 + y_p + np.abs(x_p - w + 1 + x))
            # Over the north pole
            dist = np.minimum(
                dist, 2 * h - 1 - y - y_p + np.abs(x_p - w + 1 + x)
            )
            return dist

        sum_of_mean_paths = 0.0

        for x in range(0, w // 2):
            for y in range(0, h // 2):
                sum_of_mean_paths += d(x, y, np.arange(w), np.arange(h)).mean()

        return sum_of_mean_paths / ((w // 2) * (h // 2))

    else:
        raise ValueError(graph_type)


def km_estimator(durations: List[float], censored: List[float]) \
        -> (float, (float, float)):
    '''
    Uses the Kaplan-Meier estimator to estimate the median survival time of a
    right-censored survival function and its 95% confidence interval.

    :param durations: The observed times till an event occurred.
    :type durations: list
    :param censored: The observed time periods after which an event did NOT
    occur.
    :type censored: list
    '''
    import lifelines

    kmf = lifelines.KaplanMeierFitter()
    observed = ([True] * len(durations)) + ([False] * len(censored))

    try:
        kmf.fit(durations + censored, observed)
        # This is the Brookmeyer and Crowley (1982) method, as found in
        # Kleinbaum and Klein, *Survival Analysis: A Self-Learning Text*, 3rd
        # Edition, p. 80. Note that this will give different bounds than the
        # lifelines package, which uses the exponential Greenwood formula for
        # the variance instead of the original Greenwood formula.
        sf = kmf.survival_function_['KM_estimate']
        m_f = kmf.event_table['observed']
        n_f = kmf.event_table['at_risk']
        left_hand = (sf - 0.5)**2
        right_hand = 3.84 * sf**2 * (m_f / (n_f * (n_f - m_f))).cumsum()
        ci_interval = sf.index[left_hand < right_hand]
        median_95th_ci = (ci_interval[0], ci_interval[-1])

        return kmf.median_survival_time_, median_95th_ci

    except (IndexError, ValueError):
        return float('nan'), (float('nan'), float('nan'))


def load_record(path: str) -> Dict[Tuple, List[Experiment]]:
    '''
    Loads a record file and returns it, after optionally cleaning up transitory
    patterns.

    :param path: Path to the record file.
    :type path: str

    :return: A dictionary containing the experiments. The keys of the
    dictionary will be tuples corresponding to the aggregation keys, while the
    values will be lists. Each entry in the list is an :class:`Experiment`.
    :rtype: dict
    '''
    experiments = defaultdict(list)
    experiment = None

    with open(path, 'r') as csv_file:
        for row in csv.DictReader(csv_file):
            # Get the aggregation key
            width = int(row['width'])
            key = (
                row['graph'],
                row['initialization'],
                float(row['mobility']),
                width
            )
            seed = int(row['seed'])

            if experiment is None:
                experiment = Experiment(key, seed)

            if (key, seed) != (experiment.key, experiment.seed):
                # If this is a new experiment, then add the experiment we've
                # been assembling to the record and instantiate a new
                # experiment to work on. First, correct the previous row to
                # reflect its duration is censored.
                experiment[-1] = (*experiment[-1][:2], True)
                experiments[experiment.key].append(experiment)
                experiment = Experiment(key, seed)

            duration = float(row['duration'])

            # Absorption is a terminal state; we don't care about anything that
            # happens after that.
            if int(row['absorbed']):
                if (not experiment) or (experiment[-1] != ('absorbed', 0)):
                    experiment.append(('absorbed', 0, True))
                continue

            experiment.append((row['pattern'], duration, False))

    experiment[-1] = (*experiment[-1][:2], True)
    experiments[experiment.key].append(experiment)

    return experiments


def print_header(message: str):
    '''
    Convenience function for pretty-printing a header.
    '''
    print()
    print('*' * (len(message) + 4))
    print('*', message, '*')
    print('*' * (len(message) + 4))
    print()
