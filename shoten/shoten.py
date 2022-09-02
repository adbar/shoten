# pylint: disable=E0611
"""Main module."""


import csv
import gzip
import pickle
import re

from array import array
from collections import Counter, defaultdict
from concurrent.futures import as_completed, ProcessPoolExecutor
from datetime import datetime
from functools import lru_cache, partial
from os import cpu_count, path, walk  # cpu_count
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterator, List, Match, Optional, Pattern, Set, Tuple, Union

import numpy as np  # type: ignore[import]

from courlan import extract_domain  # type: ignore
from lxml.etree import fromstring  # type: ignore[import]
from simplemma import lemmatize, simple_tokenizer, is_known  # type: ignore

from .datatypes import dict_sum, sum_entry, flatten_series, ARRAY_TYPE, Entry, MAX_SERIES_VAL, TODAY
from .filters import combined_filters, is_relevant_input, MIN_LENGTH


THREADNUM = min(cpu_count(), 16)  # type: ignore[type-var]
NSPACE = {"tei": "http://www.tei-c.org/ns/1.0"}
DATESEARCH = re.compile(r'[0-9]{4}-[0-9]{2}-[0-9]{2}')


def find_files(dirname: str, maxdiff: int) -> Iterator[str]:
    "Search a directory for files."
    for thepath, _, files in walk(dirname):
        # check for dates in directory names
        if '-' in thepath:
            if match := DATESEARCH.search(thepath):
                thisdiff = calc_timediff(match[0])
                if thisdiff is not None and thisdiff > maxdiff:
                    continue
        # yield files with full path
        yield from (path.join(thepath, fname) for fname in files if Path(fname).suffix == '.xml')


@lru_cache(maxsize=1024)
def calc_timediff(mydate: str) -> Optional[int]:
    "Compute the difference in days between today and a date in YYYY-MM-DD format."
    try:
        thisday = datetime(int(mydate[:4]), int(mydate[5:7]), int(mydate[8:10]))
    except (TypeError, ValueError):
        return None
    return (TODAY - thisday).days


def filter_lemmaform(token: str, lang: Union[str, Tuple[str, ...], None]=('de', 'en'), lemmafilter: bool=True) -> Optional[str]:
    "Determine if the token is to be processed and try to lemmatize it."
    # potential new words only
    if lemmafilter and is_known(token, lang=lang) is True:  # type: ignore[arg-type]
        return None
    # lemmatize
    try:
        return lemmatize(token, lang=lang, greedy=True, silent=False)  # type: ignore[no-any-return]
    except ValueError:
        return token


def putinvocab_single(myvocab: Dict[str, Entry], wordform: str, timediff: int, *, source: Optional[str]=None, inheadings: bool=False) -> Dict[str, Entry]:
    "Store a single word form in the vocabulary or add a new occurrence to it."
    if wordform not in myvocab:
        myvocab[wordform] = Entry(head=inheadings)
    elif inheadings and myvocab[wordform].headings is False:
        myvocab[wordform].headings = True
    myvocab[wordform].time_series[timediff] += 1
    if source:
        # slower: myvocab[wordform].sources.update(source)
        myvocab[wordform].sources[source] += 1
    return myvocab


def putinvocab_multi(vocab: Dict[str, Entry], result: Optional[Tuple[Dict[str, int], int, Optional[str], Set[Union[Match[str], str]]]]) -> Dict[str, Entry]:
    "Store a series of word forms in the vocabulary."
    if not result:
        return vocab
    tokens, timediff, source, headwords = result
    for token in tokens:
        if token not in vocab:
            vocab[token] = Entry(head=token in headwords)
        elif token in headwords and vocab[token].headings is False:
            vocab[token].headings = True
        vocab[token].time_series[timediff] += tokens[token]
        if source:
            # slower: myvocab[wordform].sources.update(source)
            vocab[token].sources[source] += 1
    return vocab


def prune_vocab(vocab: Dict[str, Entry], first: str, second: str) -> Dict[str, Entry]:
    "Append characteristics of wordform to be deleted to an other one."
    if first not in vocab:
        vocab[first] = Entry()
    if second not in vocab:
        vocab[second] = Entry()
    # sum up series (faster)
    vocab[second].time_series = dict_sum(vocab[first].time_series, vocab[second].time_series)
    # sum up sources (faster)
    vocab[second].sources = dict_sum(vocab[first].sources, vocab[second].sources)
    # set heading boolean
    if vocab[first].headings is True:
        vocab[second].headings = True
    return vocab


def dehyphen_vocab(vocab: Dict[str, Entry]) -> Dict[str, Entry]:
    "Remove hyphens in words if a variant without hyphens exists."
    deletions = []
    for wordform in [w for w in vocab if '-' in w]:
        candidate = wordform.replace('-', '').lower()
        if wordform[0].isupper():
            candidate = candidate.capitalize()
        # fusion occurrence lists and schedule for deletion
        if candidate in vocab:
            vocab = prune_vocab(vocab, wordform, candidate)
            deletions.append(wordform)
    for word in deletions:
        del vocab[word]
    return vocab


def refine_vocab(myvocab: Dict[str, Entry], lang: Union[str, Tuple[str, ...], None]=None, lemmafilter: bool=False, dehyphenation: bool=True) -> Dict[str, Entry]:
    """Refine the word list, currently: lemmatize, regroup forms with/without hyphens,
       and convert time series to numpy array."""
    if lang is not None:
        changes, deletions = [], []
        for token in myvocab:
            lemma = filter_lemmaform(token, lang=lang, lemmafilter=lemmafilter)
            #if is_relevant_input(lemma) is True:
            if lemma is None or len(lemma) < MIN_LENGTH:
                deletions.append(token)
            # register lemma and add frequencies
            elif lemma != token:
                changes.append((token, lemma))
                deletions.append(token)
        for token, lemma in changes:
            myvocab = prune_vocab(myvocab, token, lemma)
        for token in deletions:
            del myvocab[token]
    # dehyphen
    if dehyphenation:
        myvocab = dehyphen_vocab(myvocab)
    return myvocab


def read_file(filepath: str, *, maxdiff: int=1000, mindiff: int=0, authorregex: Optional[Pattern[str]]=None, details: bool=True) -> Optional[Tuple[Dict[str, int], int, Optional[str], Set[Union[Match[str], str]]]]:
    "Extract word forms from a XML TEI file generated by Trafilatura."
    # read data
    with open(filepath, 'r', encoding='utf-8') as filehandle:
        mytree = fromstring(filehandle.read())
    # todo: XML-TEI + XML
    # XML-TEI: compute difference in days
    timediff = calc_timediff(mytree.findtext(".//tei:date", namespaces=NSPACE))
    if timediff is None or not mindiff < timediff <= maxdiff:
        return None
    # XML-TEI: filter author
    # todo: add authorship flag instead?
    if authorregex is not None:
        author = mytree.findtext('.//tei:author', namespaces=NSPACE)
        if author is not None and authorregex.search(author):
            return None
        # no author string in the document, log?
    # source: extract domain from URL first
    source = None
    if details:
        url_elem = mytree.find('.//tei:ptr[@type="URL"][@target]', namespaces=NSPACE)
        if url_elem is not None:
            source = extract_domain(url_elem.get('target'), fast=True)
        # use TEI publisher info
        else:
            source = mytree.findtext('.//tei:publisher', namespaces=NSPACE)
    # headings
    headwords = set()
    if details:
        bow = [' '.join(h.itertext()) for h in mytree.xpath('.//tei:fw|.//tei:head', namespaces=NSPACE)]
        headwords = {t for t in simple_tokenizer(' '.join(bow)) if is_relevant_input(t)}
    # process
    tokens: DefaultDict[str, int] = defaultdict(int)
    for token in simple_tokenizer(' '.join(mytree.find('.//tei:text', namespaces=NSPACE).itertext())):
        # form and regex-based filter
        if is_relevant_input(token) is True:
            tokens[token] += 1  # type: ignore[index]
            # return tuple
            #yield token, timediff, source, token in headwords  # type: ignore[misc]
    return tokens, timediff, source, headwords


def parallel_reads(readfunc: Any, batch: Any) -> List[Any]:
    "Read batches of files (useful for multiprocessing)."
    return [readfunc(f) for f in batch]


def gen_wordlist(mydir: str, *, langcodes: Union[str, Tuple[str, ...], None]=None, maxdiff: int=1000, mindiff: int=0, authorregex: Optional[Pattern[str]]=None, lemmafilter: bool=False, details: bool=True, threads: Optional[int]=THREADNUM) -> Any:  # ArrayLike
    """Generate a list of occurrences (tokens or lemmatas) from an input directory
       containing XML-TEI files."""
    # init
    myvocab: Dict[str, Entry] = {}
    readfunc = partial(read_file, maxdiff=maxdiff, mindiff=mindiff, authorregex=authorregex, details=details)
    if langcodes is None:
        langcodes = ()
    # read files
    # legacy code
    if threads == 1:
        for filepath in find_files(mydir, maxdiff):
            myvocab = putinvocab_multi(myvocab, readfunc(filepath))
    # multi-threaded code
    else:
        with ProcessPoolExecutor(max_workers=threads) as executor:
            batches, tasks = [], []
            # process
            for filepath in find_files(mydir, maxdiff):
                tasks.append(filepath)
                if len(tasks) >= 50:
                    batches.append(executor.submit(parallel_reads, readfunc, tasks))
                    tasks = []
            # clean up
            if tasks:
                batches.append(executor.submit(parallel_reads, readfunc, tasks))
            # finish
            for future in as_completed(batches):
                for result in future.result():
                    myvocab = putinvocab_multi(myvocab, result)
    # post-processing
    myvocab = refine_vocab(myvocab, lang=langcodes, lemmafilter=lemmafilter)
    return myvocab


def load_wordlist(myfile: str, langcodes: Union[str, Tuple[str, ...], None]=None, maxdiff: int=1000) -> Any:
    """Load a pre-generated list of occurrences in TSV-format:
       token/lemma + TAB + date in YYYY-MM-DD format + TAB + source (optional)."""
    # init
    filepath = str(Path(__file__).parent / myfile)
    myvocab: Dict[str, Entry] = {}
    if langcodes is None:
        langcodes = ()
    with open(filepath, 'r', encoding='utf-8') as filehandle:
        for line in filehandle:
            columns = line.strip().split('\t')
            if len(columns) == 2:
                token, date, source = columns[0], columns[1], None
            elif len(columns) == 3:
                token, date, source = columns[0], columns[1], columns[2]
            else:
                print('invalid line:', line)
                continue
            # compute difference in days
            timediff = calc_timediff(date)
            if timediff is None or timediff > maxdiff:
                continue
            # skipping this: if is_relevant_input(token) is True
            myvocab = putinvocab_single(myvocab, token, timediff, source=source)
    # post-processing
    myvocab = refine_vocab(myvocab, langcodes)
    return myvocab


def pickle_wordinfo(mydict: Dict[str, Entry], filepath: str) -> None:
    "Store the frequency dict in a compressed format."
    with gzip.open(filepath, 'w') as filehandle:
        pickle.dump(mydict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_wordinfo(filepath: str) -> Any:
    "Open the compressed pickle file and load the frequency dict."
    with gzip.open(filepath) as filehandle:
        return pickle.load(filehandle)


def apply_filters(myvocab: Dict[str, Entry], setting: str='normal') -> None:
    "Default setting of chained filters for trend detection."
    if setting not in ('loose', 'normal', 'strict'):
        print('invalid setting:', setting)
        setting = 'normal'
    for wordform in sorted(combined_filters(myvocab, setting)):
        print(wordform)


def calculate_bins(vocab: Dict[str, Entry], interval: int=7, maxdiff: int=1000) -> List[int]:
    "Calculate time frame bins to fit the data (usually weeks)."
    # init
    oldest, newest = 0, maxdiff
    # iterate over vocabulary to find bounds
    for entry in vocab.values():
        oldest = max(oldest, max(entry.time_series))
        newest = min(newest, min(entry.time_series))
    # return bins corresponding to boundaries and interval
    return [d for d in range(oldest, newest, -1) if d % interval == 0 and oldest - d >= interval]


def refine_frequencies(vocab: Dict[str, Entry], bins: List[int]) -> Dict[str, Entry]:
    "Adjust the frequencies to a time frame and remove superfluous words."
    deletions = []
    # remove occurrences that are out of bounds: no complete week
    for word in vocab:
        new_series = [d for d in flatten_series(vocab[word]) if bins[-1] <= d < bins[0]]
        if len(new_series) <= 1:
            deletions.append(word)
        else:
            vocab[word].time_series = Counter(new_series)
    # remove words with too little data
    for word in deletions:
        del vocab[word]
    return vocab


def compute_frequencies(vocab: Dict[str, Entry], bins: List[int]) -> Tuple[Dict[str, Entry], List[int]]:
    "Compute absolute frequencies of words."
    timeseries = [0] * len(bins)
    # necessary for old code to work
    oldestday = max(bins) + 6
    newestday = min(bins) - 6
    # frequency computations
    freqsum = sum(sum_entry(vocab[l]) for l in vocab)
    for wordform in vocab:
        freqseries = []
        # parts per million
        ppm = (sum_entry(vocab[wordform]) / freqsum)*1000000
        vocab[wordform].total = float(f'{ppm:.3f}')
        ## OLD code
        mysum = 0
        mydays = vocab[wordform].time_series
        for day in range(oldestday, newestday, -1):
            if day in mydays:
                mysum += mydays[day]
            if day % 7 == 0:
                # prevent OverflowError according to array type
                freqseries.append(min(MAX_SERIES_VAL, mysum))
                mysum = 0
        vocab[wordform].series_abs = array(ARRAY_TYPE, freqseries)
        for i in range(len(bins)):
            timeseries[i] += vocab[wordform].series_abs[i]  # type: ignore[call-overload]
        ## NEW code: problem here
        #days = Counter(vocab[wordform].time_series)
        #for i, split in enumerate(bins):
        #    if i != 0:
        #        total = sum(days[d] for d in days if bins[i-1] < d <= split)
        #    else:
        #        total = sum(days[d] for d in days if d <= split)
        #    # prevent OverflowError according to array type
        #    total = min(MAX_SERIES_VAL, total)
        #    freqseries.append(total)
        #    timeseries[i] += total
        #vocab[wordform].series_abs = array(ARRAY_TYPE, reversed(freqseries))
        ## WRAP UP
        # spare memory
        del vocab[wordform].time_series
    return vocab, timeseries  # new code: list(reversed(timeseries)) ??


def combine_frequencies(vocab: Dict[str, Entry], bins: List[int], timeseries: List[int]) -> Dict[str, Entry]:
    "Compute relative frequencies and word statistics."
    deletions = []
    for wordform in vocab:
        for i in range(len(bins)):
            try:
                vocab[wordform].series_rel.append((vocab[wordform].series_abs[i] / timeseries[i])*1000000)
            except ZeroDivisionError:
                vocab[wordform].series_rel.append(0.0)
        # take non-zero values and perform calculations
        series = np.array([f for f in vocab[wordform].series_rel if f != 0.0], dtype="float32")
        # todo: skip if series too short
        # delete rare words to prevent unreliable figures
        #if len(series) < len(bins) / 2:
        if len(series) < 3 <= len(bins):
            deletions.append(wordform)
            continue
        vocab[wordform].stddev = float(f'{np.std(series):.3f}')
        vocab[wordform].mean = float(f'{np.mean(series):.3f}')
        # spare memory
        del vocab[wordform].series_abs
    for word in deletions:
        del vocab[word]
    return vocab


def gen_freqlist(mydir: str, *, langcodes: Union[str, Tuple[str, ...], None]=None, maxdiff: int=1000, mindiff: int=0, interval: int=7, threads: Optional[int]=THREADNUM) -> Dict[str, Entry]:
    "Compute long-term frequency info out of a directory containing text files."
    # read files
    myvocab = gen_wordlist(mydir, langcodes=langcodes, maxdiff=maxdiff, mindiff=mindiff, authorregex=None, lemmafilter=False, details=False, threads=threads)

    # determine bins
    bins = calculate_bins(myvocab, interval=interval, maxdiff=maxdiff)
    if not bins:
        print('Not enough days to compute frequencies')
        return {}

    # clean and refine the data
    myvocab = refine_frequencies(myvocab, bins)

    # frequency computations
    myvocab, timeseries = compute_frequencies(myvocab, bins)

    # sum up frequencies
    myvocab = combine_frequencies(myvocab, bins, timeseries)

    return myvocab  # type: ignore[no-any-return]


def store_freqlist(freqs: Dict[str, Entry], filename: str, thres_a: float=1, thres_b: float=0.2) -> None:
    "Write relevant (defined by frequency) long-term occurrences info to a file."
    with open(filename, 'w', encoding='utf-8') as outfile:
        tsvwriter = csv.writer(outfile, delimiter='\t')
        tsvwriter.writerow(['word', 'total', 'mean', 'stddev', 'relfreqs'])
        for entry in sorted(freqs):
            # only store statistically significant entries
            if freqs[entry].stddev == 0:
                continue
            if freqs[entry].mean > thres_a or \
                (
                    freqs[entry].mean > thres_b and \
                    freqs[entry].stddev < freqs[entry].mean/2
                ):
                tsvwriter.writerow(
                    [entry, freqs[entry].total, freqs[entry].mean,
                     freqs[entry].stddev, freqs[entry].series_rel]
                )


if __name__ == '__main__':
    print('Shoten.')
