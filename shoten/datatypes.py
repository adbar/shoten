"""Module listing data classes, types and constants used by shoten."""



#from __future__ import annotations

from array import array
from collections import defaultdict  # Counter
from datetime import datetime


TODAY = datetime.now()


ARRAY_TYPE = 'H'
MAX_SERIES_VAL = 65535

MAX_NGRAM_VOC = 15000


class Entry:
    "Defines a class for dictionaries entries, containing metadata and stats."
    __slots__ = ['absfreq', 'headings', 'mean', 'series_abs', 'series_rel', 'sources', 'stddev', 'time_series', 'total']
    def __init__(self):
        self.absfreq: int
        self.headings: bool = False
        self.mean: float
        self.series_abs = array('f')
        self.series_rel = array('f')
        self.sources = defaultdict(int)
        self.stddev: float
        self.time_series = array(ARRAY_TYPE)
        self.total: int
