'''
Consolidates a collection of record files from experiments into a single file.
'''
import argparse
from collections import defaultdict
import csv
import os
import sys

from tqdm import tqdm


FIELDNAMES = [
    'graph', 'pattern', 'duration', 'elapsed_time', 'censored', 'width',
    'mobility', 'absorbed', 'initialization', 'seed', 'warmup_mobility',
    'warmup_seed', 'warmup_time'
]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('target_files', type=str, nargs='+')
    parser.add_argument('output_file', type=str)
    parser.add_argument('--filter-by', type=str, nargs='+', default=())
    args = parser.parse_args()

    filter_by, key = defaultdict(set), None
    for word in args.filter_by:
        if word in FIELDNAMES:
            key = word
        elif key is None:
            print('Please specify fieldname first.')
            sys.exit(1)
        else:
            filter_by[key].add(word)

    if len(args.target_files) == 0:
        print('Please specify at least one target.')
        sys.exit(1)

    if os.path.exists(args.output_file):
        print('Output file already exists.')
        sys.exit(1)

    with open(args.output_file, 'w', newline='') as out_file:
        writer = csv.DictWriter(out_file, fieldnames=FIELDNAMES)
        writer.writeheader()

        for target in tqdm(args.target_files):
            with open(target, 'r') as in_file:
                reader = csv.DictReader(in_file)

                for row in reader:
                    if any(row[k] not in vs for k, vs in filter_by.items()):
                        continue
                    writer.writerow(row)
