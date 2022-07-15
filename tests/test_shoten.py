#!/usr/bin/env python

"""Tests for `shoten` package."""

import os
import re
import sys
import tempfile

from array import array
from collections import Counter
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from shoten import apply_filters, calc_timediff, calculate_bins, combine_frequencies, compute_frequencies, convert_to_numpy, dehyphen_vocab, filter_lemmaform, gen_freqlist, gen_wordlist, load_wordlist, pickle_wordinfo, putinvocab, prune_vocab, refine_frequencies, refine_vocab, store_freqlist, unpickle_wordinfo
from shoten.cli import main, parse_args, process_args
from shoten.datatypes import Entry
from shoten.filters import combined_filters, frequency_filter, headings_filter, hyphenated_filter, is_relevant_input, longtermfilter, ngram_filter, oldest_filter, read_freqlist, recognized_by_simplemma, scoring_func, shortness_filter, sources_filter, sources_freqfilter, store_results, wordlist_filter, zipf_filter


# Turns a dictionary into a class
class ToEntry(Entry):
    "Convert a dictionary to the entry class."
    def __init__(self, my_dict):
        for key in my_dict:
            setattr(self, key, my_dict[key])
        setattr(self, 'series_rel', array('f'))


def test_basics():
    """Test basic functions."""
    # dates
    assert calc_timediff('2020 A') is None
    assert calc_timediff('2020-01-01') > 1
    assert calc_timediff('2030-01-01') < 1
    # load words
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    assert len(myvocab) == 3
    assert myvocab['Other'].sources == Counter({'Source1': 1})
    # pickling and unpickling
    _, filepath = tempfile.mkstemp(suffix='.pickle', text=True)
    pickle_wordinfo(myvocab, filepath)
    myvocab2 = unpickle_wordinfo(filepath)
    assert len(myvocab2) == len(myvocab) and myvocab2['Tests'].time_series.all() == myvocab['Tests'].time_series.all()
    # generate from XML file
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir' / 'test2'), langcodes=('de'))
    assert len(myvocab) == 1 and 'Telegram' in myvocab
    # write to file
    _, temp_outputfile = tempfile.mkstemp(suffix='.tsv', text=True)
    store_results(myvocab, temp_outputfile)
    # single- vs. multi-threaded
    assert gen_wordlist(str(Path(__file__).parent / 'testdir'), langcodes=('de'), threads=1).keys() == gen_wordlist(str(Path(__file__).parent / 'testdir'), langcodes=('de'), threads=3).keys()
    # generate list
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir' / ''), langcodes=('de', 'en'))
    assert 'Messengerdienst' in myvocab
    # without language codes and with short time frame
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir'), langcodes=())
    assert 'Messengerdienst' in myvocab
    # without language codes and with short time frame
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir'), maxdiff=1)
    assert 'Messengerdienst' not in myvocab
    # with author filter
    myregex = re.compile(r'\b(|afp|apa|dpa)\b', re.I)
    myvocab = gen_wordlist(str(Path(__file__).parent / 'testdir'), authorregex=myregex)
    assert 'Messengerdienst' not in myvocab
    # test frequency calculations
    assert gen_freqlist(str(Path(__file__).parent / 'testdir' / 'test2')) == {}
    assert gen_freqlist(str(Path(__file__).parent / 'testdir'), langcodes=('en')) == {}
    result = gen_freqlist(str(Path(__file__).parent / 'testdir'), langcodes=('de', 'en'), maxdiff=10000, mindiff=0, interval=7)
    # bins present but not enough data
    assert result == {}
    # write to temp file
    mydict = {
        'Test': ToEntry({'total': 20, 'mean': 10, 'stddev': 0, 'series_rel': [10, 10]})
    }
    _, temp_outputfile = tempfile.mkstemp(suffix='.tsv', text=True)
    store_freqlist(mydict, temp_outputfile)
    mydict['Test2'] = ToEntry({'total': 10, 'mean': 5, 'stddev': 4.082, 'series_rel': [10, 5, 0]})
    store_freqlist(mydict, temp_outputfile)
    # load from file
    if os.name != "nt":  # Linux and MacOS only
        freqs = read_freqlist(temp_outputfile)
        assert freqs == {'Test2': 13.164}
        # long term filter
        myvocab = longtermfilter(deepcopy(myvocab), temp_outputfile)
        assert len(myvocab) == 1


def test_internals():
    """Test internal functions."""
    # filter known lemmata
    assert filter_lemmaform('Berge', lang='de', lemmafilter=True) is None
    assert filter_lemmaform('Berge', lang='de', lemmafilter=False) == 'Berg'

    # store in vocabulary
    myvocab = {}
    myvocab = putinvocab(myvocab, 'Berge', 5, source='Source_0', inheadings=True)
    assert 'Berge' in myvocab and myvocab['Berge'].time_series == array('H', [5]) and myvocab['Berge'].headings is True

    # de-hyphening
    myvocab = {
        'de-hyphening': ToEntry({'time_series': array('H', [1, 2, 3]), 'sources': Counter(['source1']), 'headings': True}),
        'dehyphening': ToEntry({'time_series': array('H', [3, 4]), 'sources': Counter(['source2']), 'headings': False})
    }
    # missing component
    newvocab = prune_vocab(deepcopy(myvocab), 'de-hyphen', 'dehyphen')
    assert 'dehyphen' in newvocab
    # merge operation
    newvocab = prune_vocab(deepcopy(myvocab), 'de-hyphening', 'dehyphening')
    assert len(newvocab['dehyphening'].time_series) == 5
    assert newvocab['dehyphening'].headings is True
    newvocab = dehyphen_vocab(deepcopy(myvocab))
    assert newvocab != myvocab
    assert len(newvocab['dehyphening'].time_series) == 5
    assert newvocab['dehyphening'].headings is True

    # refine vocab
    myvocab['Berge'] = ToEntry({'time_series': array('H', [5]), 'sources': Counter(['source3']), 'headings': False})
    newvocab = refine_vocab(deepcopy(myvocab), lang='de', lemmafilter=False, dehyphenation=False)
    assert 'Berg' in newvocab and 'de-hyphening' in newvocab
    newvocab = refine_vocab(deepcopy(myvocab), lang='de', lemmafilter=True, dehyphenation=True)
    assert 'Berge' not in newvocab and 'Berg' not in newvocab and 'de-hyphening' not in newvocab

    # filter levels
    newvocab = convert_to_numpy(newvocab)
    # defaults to normal
    apply_filters(deepcopy(newvocab), 'unknown')
    # chained
    apply_filters(newvocab, setting='loose')
    with pytest.raises(ZeroDivisionError):
        apply_filters(newvocab, setting='strict')

    # frequencies
    oldestday, newestday = 21, 1
    myvocab = {
        'Berg': ToEntry({'time_series': array('H', [newestday, 10, 10, oldestday])}),
        'Meer': ToEntry({'time_series': array('H', [newestday, 8, 9, 11, oldestday])}),
        'Tal': ToEntry({'time_series': array('H', [newestday, oldestday])}),
        'Zebra': ToEntry({'time_series': array('H', [10])})
    }
    bins = calculate_bins(myvocab, interval=7)
    assert bins == [14, 7]
    myvocab = refine_frequencies(myvocab, bins)
    assert 'Berg' in myvocab and 'Tal' not in myvocab and 'Zebra' not in myvocab

    # freq calculations
    myvocab, timeseries = compute_frequencies(myvocab, bins)
    assert myvocab['Berg'].total == 400000.0 and myvocab['Berg'].series_abs == array('H', [0, 2])
    assert myvocab['Meer'].total == 600000.0 and myvocab['Meer'].series_abs == array('H', [0, 3])
    assert timeseries == [0, 5]
    myvocab = combine_frequencies(myvocab, bins, timeseries)
    assert myvocab['Berg'].series_rel == array('f', [0, 400000.0])
    assert myvocab['Meer'].series_rel == array('f', [0, 600000.0])


def test_cli():
    """Basic tests for command-line interface."""
    mydir = str(Path(__file__).parent / 'testdir' / 'test')
    testargs = ['', '--read-dir', mydir, '-l', 'de', '--filter-level', 'loose']
    with patch.object(sys, 'argv', testargs):
        args = parse_args(testargs)
    process_args(args)
    myfile = str(Path(__file__).parent / 'testdir' / 'wordlist.tsv')
    testargs = ['', '--read-file', myfile, '-l', 'de', '--verbose']
    with patch.object(sys, 'argv', testargs):
        main()


def test_filters():
    "Test functions for filtering of vocabulary."
    # init
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    assert len(myvocab) == 3

    # combined
    newvocab = combined_filters(deepcopy(myvocab), 'unknown')
    # assert newvocab == myvocab

    # input
    assert is_relevant_input('testing')
    assert is_relevant_input('testing.')
    assert not is_relevant_input('')
    assert not is_relevant_input('123.')
    assert not is_relevant_input('abcde12345')
    assert not is_relevant_input('ABCDEF')
    assert not is_relevant_input(',,,,,,')

    # lemmata
    newvocab = recognized_by_simplemma(deepcopy(myvocab), lang='de')
    assert newvocab == {}

    # filters
    # sources
    newvocab = sources_filter(deepcopy(myvocab), set())
    assert len(newvocab) == 0
    newvocab = sources_filter(deepcopy(myvocab), Counter(['Source1']))
    assert len(newvocab) == 1
    # sources frequency
    newvocab = sources_freqfilter(deepcopy(myvocab), balanced=False)
    assert len(newvocab) == 2
    newvocab = sources_freqfilter(deepcopy(myvocab), balanced=True)
    assert len(newvocab) == 2
    newvocab['Tests'].sources = {'Source1': 2, 'Source2': 25}
    newvocab['Vaccines'].sources = {'Source1': 2, 'Source2': 3}
    result = sources_freqfilter(deepcopy(newvocab), balanced=True)
    assert len(result) == 1
    # morpho
    assert len(newvocab) == 2
    newvocab = shortness_filter(deepcopy(newvocab))
    assert len(newvocab) == 1
    # frequency
    newvocab = frequency_filter(deepcopy(myvocab))
    assert len(newvocab) == 2
    # list
    newvocab = wordlist_filter(deepcopy(myvocab), ['Tests', 'Other'], keep_words=False)
    assert len(newvocab) == 1
    newvocab = wordlist_filter(deepcopy(myvocab), ['Tests', 'Other'], keep_words=True)
    assert len(newvocab) == 2
    # age
    assert len(myvocab) == 3
    temp_result = oldest_filter(deepcopy(myvocab))
    assert len(temp_result) == 2

    # scoring function
    scores = {}
    scores = scoring_func(scores, 1, newvocab)
    assert scores == {'Other': 1, 'Tests': 1}
    scores = scoring_func(scores, 1, newvocab)
    assert scores == {'Other': 2, 'Tests': 2}

    # hyphens
    myvocab = {
        'de-hyphening': ToEntry({'time_series': array('H', [1, 2, 3])}),
        'hyphens-stuff': ToEntry({'time_series': array('H', [3, 3])}),
        'hyphen-stuff': ToEntry({'time_series': array('H', [3, 3, 3])}),
        'stuff': ToEntry({'time_series': array('H', [3, 3, 3, 3, 3])})
    }
    myvocab = convert_to_numpy(myvocab)
    newvocab = hyphenated_filter(myvocab, perc=0, verbose=True)
    assert list(newvocab.keys()) == ['de-hyphening', 'stuff']

    # zipf
    myvocab = {
        'short': ToEntry({'time_series': array('H', [1, 2, 3])}),
        'the-longest-word': ToEntry({'time_series': array('H', [1, 2, 3])}),
        'longer': ToEntry({'time_series': array('H', [1])}),
        'other1': ToEntry({'time_series': array('H', [1, 2, 3])}),
        'this': ToEntry({'time_series': array('H', [1, 2])}),
        'one': ToEntry({'time_series': array('H', [1, 2, 3, 4, 5])}),
    }
    myvocab = convert_to_numpy(myvocab)
    newvocab = zipf_filter(deepcopy(myvocab), freqperc=80, lenperc=20, verbose=True)
    assert len(newvocab) == 5
    newvocab = zipf_filter(deepcopy(myvocab), freqperc=20, lenperc=20, verbose=True)
    assert len(newvocab) == 4

    # headings
    myvocab = {
        'Berg': ToEntry({'time_series': array('H', [1, 2, 3]), 'headings': True}),
        'Tal': ToEntry({'time_series': array('H', [1, 2, 3]), 'headings': False}),
    }
    myvocab = headings_filter(myvocab)
    assert len(myvocab) == 1

    # n-grams
    # too large
    myvocab = dict.fromkeys([f'abc_{str(x)}' for x in range(30000)])
    newvocab = ngram_filter(myvocab, threshold=50, verbose=True)
    assert newvocab == myvocab
    # plausible test
    myvocab = dict.fromkeys(['Berg', 'Berge', 'Bergen', 'Berger', 'Bernstein', 'Tal', 'Täler'])
    newvocab = ngram_filter(deepcopy(myvocab), threshold=50, verbose=True)
    assert list(newvocab.keys()) == ['Berg', 'Berge', 'Berger', 'Bernstein', 'Tal']
    newvocab = ngram_filter(deepcopy(myvocab), threshold=10, verbose=True)
    assert list(newvocab.keys()) == ['Berge', 'Bernstein', 'Tal']



#def test_readme():
#    """Test function to verify readme examples."""
#    # ...
#
