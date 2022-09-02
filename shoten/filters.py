"""Filters meant to reduce noise."""


import csv
import re
import string

from collections import Counter
#from copy import deepcopy
from functools import lru_cache
from math import ceil
from typing import Any, Dict, List, Set, Tuple, Union

import numpy as np  # type: ignore[import]

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer  # type: ignore[import]
from sklearn.metrics.pairwise import cosine_similarity  # type: ignore[import]

from simplemma import is_known, lemmatize  # type: ignore[attr-defined]

from .datatypes import flatten_series, sum_entry, sum_entries, Entry, MAX_NGRAM_VOC, MAX_SERIES_VAL


UNSUITABLE_PUNCT = set(string.punctuation) - {'-', '_'}

MIN_LENGTH = 6
MAX_LENGTH = 40



def print_changes(phase: str, old_len: int, new_len: int) -> None:
    'Report on absolute and relative changes.'
    coeff = 100 - (new_len/old_len)*100
    print(f'{phase} removed {old_len - new_len} {coeff:.1f} %')


def scoring_func(scores: Dict[str, int], value: int, newvocab: Dict[str, Entry]) -> Dict[str, int]:
    'Defines scores for each word and add values.'
    for wordform in newvocab:
        if wordform not in scores:
            scores[wordform] = 0
        scores[wordform] += value
    return scores


def store_results(vocab: Dict[str, Entry], filename: str) -> None:
    'Write vocabulary with essential data to file.'
    with open(filename, 'w', encoding='utf-8') as outfile:
        tsvwriter = csv.writer(outfile, delimiter='\t')
        tsvwriter.writerow(['word', 'sources', 'time series'])
        # for token in sorted(vocab, key=locale.strxfrm):
        for entry in vocab:
            tsvwriter.writerow([entry, ','.join(vocab[entry].sources), str(sorted(flatten_series(vocab[entry])))])


def combined_filters(vocab: Dict[str, Entry], setting: str) -> Dict[str, Entry]:
    '''Apply a combination of filters based on the chosen setting.'''
    if setting == 'loose':
        return freshness_filter(frequency_filter(hapax_filter(vocab)))
    if setting == 'normal':
        return freshness_filter(oldest_filter(frequency_filter(hapax_filter(vocab))))
    if setting == 'strict':
        return freshness_filter(oldest_filter(
               frequency_filter(shortness_filter(ngram_filter(hapax_filter(vocab))), min_perc=10)
               ))
    return vocab


@lru_cache(maxsize=1048576)
def is_relevant_input(token: str) -> bool:
    'Determine if the given token is to be considered relevant for further processing.'
    # apply filters first
    if not MIN_LENGTH <= len(token) <= MAX_LENGTH:
        return False
    if token.endswith('-'):
        return False
    token = token.rstrip(string.punctuation)
    if not token or token.isnumeric():
        return False
    num_upper, num_digit = 0, 0
    for char in token:
        if char.isupper():
            num_upper += 1
            if num_upper > 4:
                return False
        elif char.isdigit():
            num_digit += 1
            if num_digit > 3:
                return False
        elif char in UNSUITABLE_PUNCT:
            return False
    return True


def hapax_filter(vocab: Dict[str, Entry], freqcount: int=2) -> Dict[str, Entry]:
    '''Eliminate hapax legomena and delete same date only.'''
    old_len = len(vocab)
    for token in [t for t in vocab if len(vocab[t].time_series) == 1 and sum_entry(vocab[t]) <= freqcount]:
        del vocab[token]
    print_changes('sameness/hapax', old_len, len(vocab))
    return vocab


def different_days_filter(vocab: Dict[str, Entry], threshold: int=2) -> Dict[str, Entry]:
    '''Delete words which have not been seen on at least n days.'''
    old_len = len(vocab)
    for token in [t for t in vocab if len(set(vocab[t].time_series)) < threshold]:
        del vocab[token]
    print_changes('different days', old_len, len(vocab))
    return vocab


def regex_filter(vocab: Dict[str, Entry], regex_str: str) -> Dict[str, Entry]:
    "Delete words based on a custom regular expression."
    old_len = len(vocab)
    regex = re.compile(fr'{regex_str}', re.I)
    for token in [t for t in vocab if regex.search(t)]:
        del vocab[token]
    print_changes('custom regex:', old_len, len(vocab))
    return vocab


def recognized_by_simplemma(vocab: Dict[str, Entry], lang: Union[str, Tuple[str], None]=None) -> Dict[str, Entry]:
    'Run the simplemma lemmatizer to check if input is recognized.'
    old_len = len(vocab)
    for token in [t for t in vocab if is_known(t, lang=lang)]:
        del vocab[token]
    print_changes('known by simplemma', old_len, len(vocab))
    deletions = []
    for word in vocab:
        try:
            lemmatize(word, lang=lang, greedy=True, silent=False)
        except ValueError:
            deletions.append(word)
    for token in deletions:
        del vocab[token]
    print_changes('reduced by simplemma', old_len, len(vocab))
    return vocab


def compound_filter(vocab: Dict[str, Entry], lang: Union[str, Tuple[str], None]=None, threshold: float=10) -> Dict[str, Entry]:
    "Split hyphenated compounds and delete already known or too frequent components."
    old_len = len(vocab)
    freqs = sum_entries(vocab)
    threshold = np.percentile(freqs, 10)  # type: ignore[assignment]
    for token in [t for t in vocab if '-' in t]:
        mylist = token.split('-')
        if len(mylist) == 2:
            first, second = mylist[0], mylist[1]
            #if firstpart in vocab and sum_entry(vocab[firstpart]) < threshold:
            if is_known(first, lang=lang) and is_known(second, lang=lang):
                del vocab[token]
            elif is_known(first, lang=lang) and second in vocab and sum_entry(vocab[second]) > threshold:
                del vocab[token]
            elif is_known(second, lang=lang) and first in vocab and sum_entry(vocab[first]) > threshold:
                del vocab[token]
    print_changes('compound filter:', old_len, len(vocab))
    return vocab


def shortness_filter(vocab: Dict[str, Entry], threshold: float=20) -> Dict[str, Entry]:
    '''Eliminate short words'''
    old_len = len(vocab)
    lengths = np.array([len(l) for l in vocab])
    lenthreshold = np.percentile(lengths, threshold)
    for token in [t for t in vocab if len(t) < lenthreshold]:
        del vocab[token]
    print_changes('short words', old_len, len(vocab))
    return vocab


def frequency_filter(vocab: Dict[str, Entry], max_perc: float=50, min_perc: float=.001) -> Dict[str, Entry]:  # 50 / 0.01
    '''Reduce dict size by stripping least and most frequent items.'''
    old_len = len(vocab)
    myfreq = np.array(sum_entries(vocab))
    min_thres, max_thres = np.percentile(myfreq, min_perc), np.percentile(myfreq, max_perc)
    deletions = []
    for word, values in vocab.items():
        occurrences = sum_entry(values)
        if occurrences < min_thres or occurrences > max_thres:
            deletions.append(word)
    for token in deletions:
        del vocab[token]
    print_changes('most/less frequent', old_len, len(vocab))
    return vocab


def hyphenated_filter(vocab: Dict[str, Entry], perc: float=50, verbose: bool=False) -> Dict[str, Entry]:    # threshold in percent
    '''Reduce dict size by deleting hyphenated tokens when the parts are frequent.'''
    deletions, old_len = [], len(vocab)
    myfreqs = np.array(sum_entries(vocab))
    threshold = np.percentile(myfreqs, perc)
    for word in [w for w in vocab if '-' in w]:
        mylist = word.split('-')
        firstpart, secondpart = mylist[0], mylist[1]
        if firstpart in vocab and sum_entry(vocab[firstpart]) > threshold or \
           secondpart in vocab and sum_entry(vocab[secondpart]) > threshold:
            deletions.append(word)
    if verbose:
        print(sorted(deletions))
    for item in deletions:
        del vocab[item]
    print_changes('hyphenated', old_len, len(vocab))
    return vocab


def oldest_filter(vocab: Dict[str, Entry], threshold: float=50) -> Dict[str, Entry]:
    '''Reduce number of candidate by stripping the oldest.'''
    # todo: what about cases like [365, 1, 1, 1] ?
    old_len = len(vocab)
    ratios, values = {}, []
    for key, entry in vocab.items():
        # todo: check these lines
        # ratio = sum(entry.time_series.keys()) / sum_entry(entry)
        ratio = sum(flatten_series(entry))/sum_entry(entry)
        ratios[key] = ratio
        values.append(ratio)
    threshold = np.percentile(np.array(values), threshold)
    for word, ratio in ratios.items():
        if ratio > threshold:
            del vocab[word]
    print_changes('oldest', old_len, len(vocab))
    return vocab


def zipf_filter(vocab: Dict[str, Entry], freqperc: float=65, lenperc: float=35, verbose: bool=False) -> Dict[str, Entry]:
    '''Filter candidates based on a approximation of Zipf's law.'''
    # todo: count length without hyphen or punctuation: len(l) - l.count('-')
    old_len = len(vocab)
    freqs = np.array(sum_entries(vocab))
    freqthreshold = np.percentile(freqs, freqperc)
    lengths = np.array([len(l) for l in vocab])
    lenthreshold = np.percentile(lengths, lenperc)
    deletions = []
    for token in [t for t in vocab if sum_entry(vocab[t]) >= freqthreshold and len(t) <= lenthreshold]:
        deletions.append(token)
        # if verbose is True:
            # print(token, len(token), vocab[token].time_series.shape[0])
        del vocab[token]
    if verbose:
        print(sorted(deletions))
    print_changes('zipf frequency', old_len, len(vocab))
    return vocab


def freshness_filter(vocab: Dict[str, Entry], percentage: float=10) -> Dict[str, Entry]:
    '''Define a freshness threshold to model series of token occurrences in time.'''
    old_len = len(vocab)
    datethreshold = np.percentile(sum_entries(vocab), percentage)
    deletions = []
    for token in vocab:
        # re-order
        series = [-i for i in sorted(flatten_series(vocab[token]))]
        # thresholds
        thresh = len(series)*(percentage/100)
        freshnessindex = sum(series[-ceil(thresh):])
        oldnessindex = sum(series[:ceil(thresh)])
        if oldnessindex < datethreshold and freshnessindex < np.percentile(
            series, percentage
        ):
            deletions.append(token)
    for item in deletions:
        del vocab[item]
    print_changes('freshness', old_len, len(vocab))
    return vocab


def sources_freqfilter(vocab: Dict[str, Entry], threshold: int=2, balanced: bool=True) -> Dict[str, Entry]:
    '''Filter words based on source diversity.'''
    deletions = []
    i, j = 0, 0
    for word, entry in vocab.items():
        sources = entry.sources
        if len(sources) == 0:
            continue
        # absolute number
        if len(sources) < threshold:
            deletions.append(word)
            i += 1
            continue
        # distribution of sources
        if balanced:
            values = [t[1] for t in Counter(sources).most_common()]
            # first value too present compared to the rest
            # (sum(values)/len(values))
            if values[0] >= 4*values[1]:
                deletions.append(word)
                j += 1
    old_len = len(vocab)
    for item in deletions:
        del vocab[item]
    print_changes('sources freq', old_len, old_len-i)
    print_changes('sources balance', old_len, old_len-j)
    return vocab


def sources_filter(vocab: Dict[str, Entry], myset: Set[str]) -> Dict[str, Entry]:
    '''Only keep the words for which the source contains at least
       one string listed in the input set.'''
    deletions = []
    for word in vocab:
        deletion_flag = True
        if len(vocab[word].sources) > 0:
            for source in vocab[word].sources:
                # for / else construct
                for mystring in myset:
                    if mystring in source:
                        deletion_flag = False
                        break
                else:
                    continue
                # inner loop was broken, break the outer
                break
        # record deletion
        if deletion_flag:
            deletions.append(word)
    old_len = len(vocab)
    for item in deletions:
        del vocab[item]
    print_changes('sources list', old_len, len(vocab))
    return vocab


def wordlist_filter(vocab: Dict[str, Entry], mylist: List[str], keep_words: bool=False) -> Dict[str, Entry]:
    '''Keep or discard words present in the input list.'''
    intersection = set(vocab) & set(mylist)
    deletions = (
        [w for w in vocab if w not in intersection]
        if keep_words
        else list(intersection)
    )

    old_len = len(vocab)
    for word in deletions:
        del vocab[word]
    print_changes('word list', old_len, len(vocab))
    return vocab


def headings_filter(vocab: Dict[str, Entry]) -> Dict[str, Entry]:
    '''Filter words based on their presence in headings.'''
    deletions = [word for word in vocab if vocab[word].headings is False]
    old_len = len(vocab)
    for item in deletions:
        del vocab[item]
    print_changes('headings', old_len, len(vocab))
    return vocab


def read_freqlist(filename: str) -> Dict[str, float]:
    'Read frequency info from a TSV file.'
    freqlimits = {}
    with open(filename, 'r', encoding='utf-8') as csvfile:
        tsvreader = csv.reader(csvfile, delimiter='\t')
        # skip headline
        next(tsvreader)
        for row in tsvreader:
            # unpack
            word, mean, stddev = row[0], float(row[2]), float(row[3])
            # limit
            freqlimits[word] = float(f'{mean + 2*stddev:.3f}')
    return freqlimits


def longtermfilter(vocab: Dict[str, Entry], filename: str, mustexist: bool=False, startday: int=1, interval: int=7) -> Dict[str, Entry]:
    'Discard words which are not significantly above a mean long-term frequency.'
    freqlimits = read_freqlist(filename)
    oldestday = startday + interval - 1
    allfreqs = 0
    for word in vocab:
        mydays = vocab[word].time_series
        occurrences = sum(
            mydays[day]
            for day in range(oldestday, startday - 1, -1)
            if day in mydays
        )
        # compare with maximum possible value
        # todo: check this line!
        occurrences = min(MAX_SERIES_VAL, occurrences)
        # compute totals
        vocab[word].absfreq = occurrences
        allfreqs += occurrences
    # safeguard
    if allfreqs == 0:
        return vocab
    # relative frequency
    deletions = []
    intersection = set(vocab) & set(freqlimits)
    for word in intersection:
        relfreq = (vocab[word].absfreq / allfreqs)*1000000
        # threshold defined by long-term frequencies
        if relfreq < freqlimits[word]:
            #print(word, relfreq, freqlimits[word])
            deletions.append(word)
    if mustexist:
        deletions += [w for w in vocab if w not in freqlimits]
    old_len = len(vocab)
    for item in deletions:
        del vocab[item]
    print_changes('long-term frequency threshold', old_len, len(vocab))
    return vocab


def ngram_filter(vocab: Dict[str, Entry], threshold: float=90, verbose: bool=False) -> Dict[str, Entry]:
    '''Find dissimilar tokens based on character n-gram occurrences.'''
    lengths = np.array([len(l) for l in vocab])
    minlengthreshold = np.percentile(lengths, 1)
    for i in (70, 65, 60, 55, 50, 45, 40):
        maxlengthreshold = np.percentile(lengths, i)
        mytokens = [t for t in vocab if minlengthreshold <= len(t) <= maxlengthreshold]
        #print(i, len(mytokens))
        if len(mytokens) <= MAX_NGRAM_VOC:
            break
    if len(mytokens) > MAX_NGRAM_VOC:
        print('Vocabulary size too large, skipping n-gram filtering')
        return vocab
    old_len = len(vocab)
    # token cosine similarity
    max_exp = 21
    vectorizer = CountVectorizer(analyzer='char', max_features=2 ** max_exp, ngram_range=(1,4), strip_accents=None, lowercase=True, max_df=1.0)
    firstset = set(compute_deletions(mytokens, vectorizer, threshold))
    vectorizer = TfidfVectorizer(analyzer='char', max_features=2 ** max_exp, ngram_range=(1,4), strip_accents=None, lowercase=True, max_df=1.0, sublinear_tf=True, binary=True)
    secondset = set(compute_deletions(mytokens, vectorizer, threshold))
    for token in firstset.intersection(secondset):
        del vocab[token]
    if verbose:
        print(sorted(firstset.intersection(secondset)))
    print_changes('ngrams', old_len, len(vocab))
    return vocab


def compute_deletions(mytokens: List[str], vectorizer: Any, threshold: float) -> List[str]:
    '''Compute deletion list based on n-gram dissimilarities.'''
    count_matrix = vectorizer.fit_transform(mytokens)
    cosine_similarities = cosine_similarity(count_matrix)  # linear_kernel, laplacian_kernel, rbf_kernel, sigmoid_kernel
    myscores = {
        # np.median(cosine_similarities[:, rownum]) * (np.log(len(mytokens[rownum]))/len(mytokens[rownum]))
        mytokens[rownum]: 1 - np.mean(cosine_similarities[:, rownum])
        for rownum, _ in enumerate(mytokens)
    }

    # print info
    #if verbose is True:
    #    resultsize = 20 # :resultsize*2
    #    for k, v in sorted(myscores.items(), key=lambda item: item[1], reverse=True)[:resultsize]:
    #        print(k, v)
    #    #print()
    #    for k, v in sorted(myscores.items(), key=lambda item: item[1])[:resultsize]:
    #        print(k, v)
    # process
    mylist = np.array([myscores[s] for s in myscores])
    return [s for s in myscores if myscores[s] >= np.percentile(mylist, threshold)]
