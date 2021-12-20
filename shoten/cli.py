"""Command-line interface."""

import argparse
import sys

from .shoten import gen_wordlist, load_wordlist
from .filters import combined_filters



def parse_args(args):
    parser = argparse.ArgumentParser(description='Command-line interface for Shoten')
    parser.add_argument("-f", "--read-file",
                        help="name of input file",
                        type=str)
    parser.add_argument("-d", "--read-dir",
                        help="name of input directory",
                        type=str)
    parser.add_argument("-l", "--language",
                        help="languages of interest",
                        nargs='+', default=[])
    parser.add_argument('--filter-level',
                        help="Choose a level of filtering",
                        choices=['loose', 'normal', 'strict'],
                        default='normal')
    parser.add_argument('-v', '--verbose', action="store_true",
                        help="toggle verbose output",
                        #action='count', default=0,
                        #help="increase logging verbosity (-v or -vv)",
                        )
    return parser.parse_args()


def process_args(args):
    if args.read_file:
        myvocab = load_wordlist(args.read_file, args.language)
    elif args.read_dir:
        myvocab = gen_wordlist(args.read_dir, args.language)
    try:
        myvocab = combined_filters(myvocab, args.filter_level)
    except IndexError:
        myvocab = {}
    for wordform in sorted(myvocab):
        if args.verbose is True:
            print(wordform, myvocab[wordform])
        else:
            print(wordform)


def main():
    """ Run as a command-line utility. """
    args = parse_args(sys.argv[1:])
    process_args(args)


if __name__ == '__main__':
    main()
