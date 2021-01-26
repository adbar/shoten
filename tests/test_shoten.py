#!/usr/bin/env python

"""Tests for `shoten` package."""

import pytest

from pathlib import Path

from shoten import *



def test_basics():
    """Test basic functions."""
    assert calc_timediff('2020 A') is None
    assert calc_timediff('2020-01-01') > 1
    assert calc_timediff('2030-01-01') < 1
    myvocab = load_wordlist(str(Path(__file__).parent / 'inputfile.txt'))
    assert len(myvocab) == 2
    pickle_wordinfo(myvocab, str(Path(__file__).parent / 'test.pickle'))
    myvocab2 = unpickle_wordinfo(str(Path(__file__).parent / 'test.pickle'))
    assert len(myvocab2) == len(myvocab) and myvocab2['Tests'].all() == myvocab['Tests'].all()


#def test_readme():
#    """Test function to verify readme examples."""
#    # ...
#
