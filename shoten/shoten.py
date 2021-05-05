"""Main module."""


import csv
import gzip
import pickle

from collections import Counter, defaultdict
# from concurrent.futures import as_completed, ThreadPoolExecutor
from datetime import datetime
from os import path, walk # cpu_count
from pathlib import Path

import numpy as np

from simplemma import load_data, lemmatize, simple_tokenizer, is_known
from htmldate.utils import load_html #, sanitize

import _pickle as cpickle

from .filters import combined_filters, is_relevant_input


TODAY = datetime.today()


def find_files(dirname):
    "Search a directory for files."
    for thepath, _, files in walk(dirname):
        yield from (path.join(thepath, fname) for fname in files if Path(fname).suffix == '.xml')


def calc_timediff(mydate):
    "Compute the difference in days between today and a date in YYYY-MM-DD format."
    try:
        thisday = datetime.strptime(mydate, '%Y-%m-%d')
    except (TypeError, ValueError):
        return None
    return (TODAY - thisday).days


def filter_lemmaform(token, lemmadata, lemmafilter=True):
    "Determine if the token is to be processed and try to lemmatize it."
    # potential new words only
    if lemmafilter is True and is_known(token, lemmadata) is True:
        return None
    # lemmatize
    try:
        return lemmatize(token, lemmadata, greedy=True, silent=False)
    except ValueError:
        return token


def putinvocab(myvocab, wordform, timediff, source, inheadings=False):
    "Store the word form in the vocabulary or add a new occurrence to it."
    # form and regex-based filter
    if is_relevant_input(wordform) is False:
        return myvocab
    # store info
    if wordform not in myvocab:
        myvocab[wordform] = dict(time_series=[], sources=[], headings=False)
    myvocab[wordform]['time_series'].append(timediff)
    if source is not None and len(source) > 0:
        myvocab[wordform]['sources'].append(source)
    if inheadings is True:
        myvocab[wordform]['headings'] = True
    return myvocab


def append_to_vocab(myvocab, first, second):
    "Append characteristics of wordform to be deleted to an other one."
    if second not in myvocab:
        myvocab[second] = dict(time_series=[], sources=[], headings=False)
    myvocab[second]['time_series'] = myvocab[second]['time_series'] + myvocab[first]['time_series']
    try:
        myvocab[second]['sources'] = myvocab[second]['sources'] + myvocab[first]['sources']
        if myvocab[first]['headings'] is True:
            myvocab[second]['headings'] = True
    # additional info potentially not present
    except KeyError:
        pass
    return myvocab


def dehyphen_vocab(vocab):
    "Remove hyphens in words if a variant without hyphens exists."
    deletions = []
    for wordform in [w for w in vocab if '-' in w]:
        splitted = wordform.split('-')
        candidate = ''.join([t.lower() for t in splitted])
        if wordform[0].isupper():
            candidate = candidate.capitalize()
        # fusion occurrence lists and schedule for deletion
        if candidate in vocab:
            vocab = append_to_vocab(vocab, wordform, candidate)
            deletions.append(wordform)
    for word in deletions:
        del vocab[word]
    return vocab


def refine_vocab(myvocab, lemmadata, lemmafilter=False, dehyphenation=True):
    """Refine the word list, currently: lemmatize, regroup forms with/without hyphens,
       and convert time series to numpy array."""
    if len(lemmadata) > 0:
        changes, deletions = [], []
        for token in myvocab:
            lemma = filter_lemmaform(token, lemmadata, lemmafilter)
            ##if is_relevant_input(lemma) is True:
            if lemma is None:
                deletions.append(token)
            # register lemma and add frequencies
            elif lemma != token:
                changes.append((token, lemma))
                deletions.append(token)
        for token, lemma in changes:
            myvocab = append_to_vocab(myvocab, token, lemma)
        for token in deletions:
            del myvocab[token]
    # dehyphen
    if dehyphenation is True:
        myvocab = dehyphen_vocab(myvocab)
    return myvocab


def convert_to_array(myvocab):
    "Convert time series to numpy array."
    for wordform in myvocab:
        myvocab[wordform]['time_series'] = np.array(myvocab[wordform]['time_series'])
    return myvocab


def read_file(filepath, maxdiff=1000, authorregex=None):
    "Extract word forms from a XML TEI file generated by Trafilatura."
    # read data
    with open(filepath, 'rb') as filehandle:
        mytree = load_html(filehandle.read())
    # todo: XML-TEI + XML
    # ...
    # XML-TEI: filter author
    if authorregex is not None:
        try:
            author = mytree.xpath('//author')[0].text
            if authorregex.search(author):
                return
        # no author string in the document, log?
        except IndexError:
            pass
    # XML-TEI: compute difference in days
    timediff = calc_timediff(mytree.xpath('//date')[0].text)
    if timediff is None or timediff >= maxdiff:
        return
    # process
    source = mytree.xpath('//publisher')[0].text
    # headings
    headwords = set()
    for heading in mytree.xpath('//fw'):
        if heading.text_content() is not None:
            # print(heading.text_content())
            for token in simple_tokenizer(heading.text_content()):
                headwords.add(token)
    # process
    for token in simple_tokenizer(' '.join(mytree.xpath('//text')[0].itertext())):
        inheadings = False
        if token in headwords:
            inheadings = True
        # return tuple
        yield token, timediff, source, inheadings


def gen_wordlist(mydir, langcodes=[], maxdiff=1000, authorregex=None, lemmafilter=False):
    """Generate a list of occurrences (tokens or lemmatas) from an input directory
       containing XML-TEI files."""
    # init
    myvocab = dict()
    # load language data
    lemmadata = load_data(*langcodes)
    # read files
    #with ThreadPoolExecutor(max_workers=1) as executor:  # min(cpu_count()*2, 16)
    #    futures = {executor.submit(read_file, f, 15): f for f in find_files(mydir)}
    #    for future in as_completed(futures):
    #        for token, timediff in future.result():
    #            myvocab[token] = np.append(myvocab[token], timediff)
    for filepath in find_files(mydir):
        for token, timediff, source, inheadings in read_file(filepath, maxdiff, authorregex):
            myvocab = putinvocab(myvocab, token, timediff, source, inheadings)
    # post-processing
    myvocab = refine_vocab(myvocab, lemmadata, lemmafilter)
    return convert_to_array(myvocab)


def load_wordlist(myfile, langcodes=[], maxdiff=1000):
    """Load a pre-generated list of occurrences in TSV-format:
       token/lemma + TAB + date in YYYY-MM-DD format + TAB + source (optional)."""
    filepath = str(Path(__file__).parent / myfile)
    myvocab = defaultdict(list)
    # load language data
    lemmadata = load_data(*langcodes)
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
            myvocab = putinvocab(myvocab, token, timediff, source)
    # post-processing
    myvocab = refine_vocab(myvocab, lemmadata)
    return convert_to_array(myvocab)


def pickle_wordinfo(mydict, filepath):
    "Store the frequency dict in a compressed format."
    with gzip.open(filepath, 'w') as filehandle:
        cpickle.dump(mydict, filehandle, protocol=pickle.HIGHEST_PROTOCOL)


def unpickle_wordinfo(filepath):
    "Open the compressed pickle file and load the frequency dict."
    with gzip.open(filepath) as filehandle:
        return cpickle.load(filehandle)


def apply_filters(myvocab, setting='normal'):
    "Default setting of chained filters for trend detection."
    if setting not in ('loose', 'normal', 'strict'):
        print('invalid setting:', setting)
        setting = 'normal'
    for wordform in sorted(combined_filters(myvocab, setting)):
        print(wordform)


def gen_freqlist(mydir, langcodes=[], maxdiff=1000, mindiff=0):
    "Compute long-term frequency info out of a directory containing text files."
    # init
    myvocab, freqs, oldestday, newestday = dict(), dict(), 0, maxdiff
    # load language data
    lemmadata = load_data(*langcodes)
    # read files
    for filepath in find_files(mydir):
        # read data
        with open(filepath, 'rb') as filehandle:
            mytree = load_html(filehandle.read())
            # XML-TEI: compute difference in days
            timediff = calc_timediff(mytree.xpath('//date')[0].text)
            if timediff is None or not mindiff < timediff < maxdiff:
                continue
            if timediff > oldestday:
                oldestday = timediff
            elif timediff < newestday:
                newestday = timediff
            # extract
            for token in simple_tokenizer(' '.join(mytree.xpath('//text')[0].itertext())):
                if is_relevant_input(token) is True:
                    if token not in myvocab:
                        myvocab[token] = dict(time_series=[])
                    myvocab[token]['time_series'].append(timediff)
    # lemmatize and dehyphen
    myvocab = refine_vocab(myvocab, lemmadata)
    # determine bins
    bins = [i for i in range(oldestday, newestday, -1) if oldestday - i >= 7 and i % 7 == 0]
    if len(bins) == 0:
        print('Not enough days to compute frequencies')
        return freqs
    timeseries = [0] * len(bins)
    #print(oldestday, newestday)
    # remove occurrences that are out of bounds: no complete week
    for item in myvocab:
        myvocab[item]['time_series'] = [d for d in myvocab[item]['time_series'] if not d < bins[-1] and not d > bins[0]]
    # remove hapaxes
    deletions = [w for w in myvocab if len(myvocab[w]['time_series']) <= 1]
    for item in deletions:
        del myvocab[item]
    # frequency computations
    freqsum = sum([len(myvocab[l]['time_series']) for l in myvocab])
    for wordform in myvocab:
        # parts per million
        myvocab[wordform]['total'] = float('{0:.3f}'.format((len(myvocab[wordform]['time_series']) / freqsum)*1000000))
        counter = 0
        freqseries = [0] * len(bins)
        mydays = Counter(myvocab[wordform]['time_series'])
        for day in range(oldestday, newestday, -1):
            if day in mydays:
                counter += mydays[day]
            if day % 7 == 0:
                try:
                    freqseries[bins.index(day)] = counter
                    counter = 0
                except ValueError:
                    pass
        myvocab[wordform]['series_abs'] = freqseries
        # spare memory
        myvocab[wordform]['time_series'] = []
        for i in range(len(bins)):
            timeseries[i] += myvocab[wordform]['series_abs'][i]
    # sum up frequencies
    for wordform in myvocab:
        myvocab[wordform]['series_rel'] = [0] * len(bins)
        for i in range(len(bins)):
            try:
                myvocab[wordform]['series_rel'][i] = (myvocab[wordform]['series_abs'][i] / timeseries[i])*1000000
            except ZeroDivisionError:
                pass
        # take non-zero values and perform calculations
        series = [f for f in myvocab[wordform]['series_rel'] if f != 0]
        # todo: skip if series too short
        myvocab[wordform]['stddev'] = float('{0:.3f}'.format(np.std(series)))
        myvocab[wordform]['mean'] = float('{0:.3f}'.format(np.mean(series)))
        # spare memory
        myvocab[wordform]['series_abs'] = []
    return myvocab


def store_freqlist(freqs, filename, thres_a=1, thres_b=0.2):
    "Write relevant (defined by frequency) long-term occurrences info to a file."
    with open(filename, 'w') as outfile:
        tsvwriter = csv.writer(outfile, delimiter='\t')
        tsvwriter.writerow(['word', 'total', 'mean', 'stddev', 'relfreqs'])
        for entry in sorted(freqs):
            # only store statistically significant entries
            if freqs[entry]['stddev'] == 0:
                continue
            if freqs[entry]['mean'] > thres_a or \
                (
                    freqs[entry]['mean'] > thres_b and \
                    freqs[entry]['stddev'] < freqs[entry]['mean']/2
                ):
                tsvwriter.writerow(
                    [entry, freqs[entry]['total'], freqs[entry]['mean'],
                     freqs[entry]['stddev'], freqs[entry]['series_rel']]
                )


if __name__ == '__main__':
    print('Shoten.')
