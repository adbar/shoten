"""Filters meant to reduce noise."""


import csv
import re
import string

from collections import Counter
from math import ceil

import numpy as np

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from simplemma import is_known, lemmatize


MAX_NGRAM_VOC = 15000

RE_FILTER = re.compile(r'[^\W\d\.-]')



def print_changes(phase, old_len, new_len):
    'Report on absolute and relative changes.'
    coeff = 100 - (new_len/old_len)*100
    print(phase, 'removed', old_len - new_len, '%.1f' % coeff, '%')


def scoring_func(scores, value, newvocab):
    for wordform in set(newvocab):
        if wordform not in scores:
            scores[wordform] = 0
        scores[wordform] += value
    return scores


def store_results(myvocab, filename):
    with open(filename, 'w') as outfile:
        tsvwriter = csv.writer(outfile, delimiter='\t')
        tsvwriter.writerow(['word', 'sources', 'time series'])
        # for token in sorted(myvocab, key=locale.strxfrm):
        for entry in myvocab:
            tsvwriter.writerow([entry, ','.join(myvocab[entry]['sources']), str(myvocab[entry]['time_series'].tolist())])


def combined_filters(myvocab, setting):
    '''Apply a combination of filters based on the chosen setting.'''
    if setting == 'loose':
        return freshness_filter(frequency_filter(hapax_filter(myvocab)))
    if setting == 'normal':
        return freshness_filter(oldest_filter(frequency_filter(hapax_filter(myvocab))))
    if setting == 'strict':
        return freshness_filter(oldest_filter(
               frequency_filter(shortness_filter(ngram_filter(hapax_filter(myvocab))), min_perc=10)
               ))
    return myvocab


def is_relevant_input(token):
    # apply filter first
    if not 5 <= len(token) <= 50 or token.startswith('@') or token.startswith('#') or token.endswith('-'):
        return False
    token = token.rstrip(string.punctuation)
    if len(token) == 0 or token.isnumeric() or not RE_FILTER.search(token):
        return False
    if sum(map(str.isupper, token)) > 4 or sum(map(str.isdigit, token)) > 4:
        return False
    return True


def hapax_filter(myvocab, freqcount=2):
    '''Eliminate hapax legomena and delete same date only.'''
    old_len = len(myvocab)
    for token in [t for t in myvocab if np.unique(myvocab[t]['time_series']).shape[0] <= freqcount]:
        del myvocab[token]
    print_changes('sameness/hapax', old_len, len(myvocab))
    return myvocab


def recognized_by_simplemma(myvocab, lemmadata):
    old_len = len(myvocab)
    for token in [t for t in myvocab if is_known(t, lemmadata)]:
        del myvocab[token]
    print_changes('known by simplemma', old_len, len(myvocab))
    deletions = []
    for word in myvocab:
        try:
            lemmatize(word, lemmadata, greedy=True, silent=False)
        except ValueError:
            deletions.append(word)
    for token in deletions:
        del myvocab[token]
    print_changes('reduced by simplemma', old_len, len(myvocab))
    return myvocab


def shortness_filter(myvocab, threshold=20):
    '''Eliminate short words'''
    old_len = len(myvocab)
    lengths = np.array([len(l) for l in myvocab])
    lenthreshold = np.percentile(lengths, threshold)
    for token in [t for t in myvocab if len(t) < lenthreshold]:
        del myvocab[token]
    print_changes('short words', old_len, len(myvocab))
    return myvocab


def frequency_filter(myvocab, max_perc=50, min_perc=.001): # 50 / 0.01
    '''Reduce dict size by stripping least and most frequent items.'''
    old_len = len(myvocab)
    myfreq = np.array([myvocab[l]['time_series'].shape[0] for l in myvocab])
    min_thres, max_thres = np.percentile(myfreq, min_perc), np.percentile(myfreq, max_perc)
    for token in [t for t in myvocab if
                  myvocab[t]['time_series'].shape[0] < min_thres or myvocab[t]['time_series'].shape[0] > max_thres
                 ]:
        del myvocab[token]
    print_changes('most/less frequent', old_len, len(myvocab))
    return myvocab


def hyphenated_filter(myvocab, perc=50, verbose=False): # threshold in percent
    '''Reduce dict size by deleting hyphenated tokens when the parts are frequent.'''
    deletions, old_len = [], len(myvocab)
    myfreqs = np.array([myvocab[l]['time_series'].shape[0] for l in myvocab])
    threshold = np.percentile(myfreqs, perc)
    for word in [w for w in myvocab if '-' in w]:
        mylist = word.split('-')
        firstpart, secondpart = mylist[0], mylist[1]
        if firstpart in myvocab and myvocab[firstpart]['time_series'].shape[0] > threshold or \
           secondpart in myvocab and myvocab[secondpart]['time_series'].shape[0] > threshold:
            deletions.append(word)
    if verbose is True:
        print(sorted(deletions))
    for item in deletions:
        del myvocab[item]
    print_changes('hyphenated', old_len, len(myvocab))
    return myvocab


def oldest_filter(myvocab, threshold=50):
    '''Reduce number of candidate by stripping the oldest.'''
    # todo: what about cases like [365, 1, 1, 1] ?
    old_len = len(myvocab)
    myratios = np.array([np.sum(myvocab[l]['time_series'])/myvocab[l]['time_series'].shape[0] for l in myvocab])
    threshold = np.percentile(myratios, threshold)
    for token in [t for t in myvocab if np.sum(myvocab[t]['time_series'])/myvocab[t]['time_series'].shape[0] > threshold]:
        del myvocab[token]
    print_changes('oldest', old_len, len(myvocab))
    return myvocab


def zipf_filter(myvocab, freqperc=65, lenperc=35, verbose=False):
    '''Filter candidates based on a approximation of Zipf's law.'''
    # todo: count length without hyphen or punctuation: len(l) - l.count('-')
    old_len = len(myvocab)
    freqs = np.array([myvocab[l]['time_series'].shape[0] for l in myvocab])
    freqthreshold = np.percentile(freqs, freqperc)
    lengths = np.array([len(l) for l in myvocab])
    lenthreshold = np.percentile(lengths, lenperc)
    deletions = []
    for token in [t for t in myvocab if myvocab[t]['time_series'].shape[0] >= freqthreshold and len(t) <= lenthreshold]:
        deletions.append(token)
        # if verbose is True:
            # print(token, len(token), myvocab[token]['time_series'].shape[0])
        del myvocab[token]
    if verbose is True:
        print(sorted(deletions))
    print_changes('zipf frequency', old_len, len(myvocab))
    return myvocab


def freshness_filter(myvocab, percentage=10):
    '''Define a freshness threshold to model series of token occurrences in time.'''
    old_len = len(myvocab)
    mysums = np.array([np.sum(myvocab[l]['time_series']) for l in myvocab])
    datethreshold = np.percentile(mysums, percentage)
    deletions = []
    for token in myvocab:
        # re-order
        myvocab[token]['time_series'] = -np.sort(-myvocab[token]['time_series'])
        # thresholds
        thresh = myvocab[token]['time_series'].shape[0]*(percentage/100)
        freshnessindex = np.sum(myvocab[token]['time_series'][-ceil(thresh):])
        oldnessindex = np.sum(myvocab[token]['time_series'][:ceil(thresh)])
        if oldnessindex < datethreshold:
            #if oldnessindex < np.percentile(myvocab[token]['time_series'], percentage):
            #    continue
            thresh = myvocab[token]['time_series'].shape[0]*(percentage/100)
            freshnessindex = np.sum(myvocab[token]['time_series'][-ceil(thresh):])
            if freshnessindex < np.percentile(myvocab[token]['time_series'], percentage):
                deletions.append(token)
            # print(myvocab[token], freshnessindex, oldnessindex, token)
    for item in deletions:
        del myvocab[item]
    print_changes('freshness', old_len, len(myvocab))
    return myvocab


def sources_freqfilter(myvocab, threshold=2, balanced=True):
    '''Filter words based on source diversity.'''
    deletions = []
    i, j = 0, 0
    for word in myvocab:
        if len(myvocab[word]['sources']) == 0:
            continue
        # absolute number
        if len(myvocab[word]['sources']) < threshold:
            deletions.append(word)
            i += 1
            continue
        # distribution of sources
        if balanced is True:
            values = [t[1] for t in myvocab[word]['sources'].most_common()]
            # first value too present compared to the rest
            if values[0] >= 4*values[1]: # (sum(values)/len(values)):
                deletions.append(word)
                j += 1
                continue
    old_len = len(myvocab)
    for item in deletions:
        del myvocab[item]
    print_changes('sources freq', old_len, old_len-i)
    print_changes('sources balance', old_len, old_len-j)
    return myvocab


def sources_filter(myvocab, myset):
    '''Only keep the words for which the source contains at least
       one string listed in the input set.'''
    deletions = []
    for word in myvocab:
        deletion_flag = True
        if len(myvocab[word]['sources']) > 0:
            for source in myvocab[word]['sources']:
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
    old_len = len(myvocab)
    for item in deletions:
        del myvocab[item]
    print_changes('sources list', old_len, len(myvocab))
    return myvocab


def wordlist_filter(myvocab, mylist, keep_words=False):
    '''Keep or discard words present in the input list.'''
    mylist = set(mylist)
    intersection = [w for w in myvocab if w in mylist]
    if keep_words is False:
        deletions = intersection
    else:
        intersection = set(intersection)
        deletions = [w for w in myvocab if w not in intersection]
    old_len = len(myvocab)
    for word in deletions:
        del myvocab[word]
    print_changes('word list', old_len, len(myvocab))
    return myvocab


def headings_filter(myvocab):
    '''Filter words based on their presence in headings.'''
    deletions = [word for word in myvocab if myvocab[word]['headings'] is False]
    old_len = len(myvocab)
    for item in deletions:
        del myvocab[item]
    print_changes('headings', old_len, len(myvocab))
    return myvocab


def read_freqlist(filename):
    freqlimits = {}
    with open(filename, 'r') as csvfile:
        tsvreader = csv.reader(csvfile, delimiter='\t')
        # skip headline
        next(tsvreader)
        for row in tsvreader:
            # unpack
            word, mean, stddev = row[0], float(row[2]), float(row[3])
            # limit
            freqlimits[word] = float('{0:.3f}'.format(mean + 2*stddev))
    return freqlimits


def longtermfilter(myvocab, filename, mustexist=False, startday=1):
    freqlimits = read_freqlist(filename)
    oldestday = startday + 6
    allfreqs = 0
    for word in myvocab:
        mydays = Counter(myvocab[word]['time_series'])
        occurrences = sum(
            mydays[day]
            for day in range(oldestday, startday - 1, -1)
            if day in mydays
        )

        myvocab[word]['absfreq'] = occurrences
        allfreqs += occurrences
    # relative frequency
    deletions = []
    intersection = [w for w in myvocab if w in freqlimits]
    for word in intersection:
        relfreq = (myvocab[word]['absfreq'] / allfreqs)*1000000
        # threshold defined by long-term frequencies
        if relfreq < freqlimits[word]:
            #print(word, relfreq, freqlimits[word])
            deletions.append(word)
    if mustexist is True:
        exclusion = [w for w in myvocab if w not in freqlimits]
        deletions += exclusion
    old_len = len(myvocab)
    for item in deletions:
        del myvocab[item]
    print_changes('long-term frequency threshold', old_len, len(myvocab))
    return myvocab


def ngram_filter(myvocab, threshold=90, verbose=False):
    '''Find dissimilar tokens based on character n-gram occurrences.'''
    lengths = np.array([len(l) for l in myvocab])
    minlengthreshold = np.percentile(lengths, 1)
    for i in (70, 65, 60, 55, 50, 45, 40):
        maxlengthreshold = np.percentile(lengths, i)
        mytokens = [t for t in myvocab if minlengthreshold <= len(t) <= maxlengthreshold]
        #print(i, len(mytokens))
        if len(mytokens) <= MAX_NGRAM_VOC:
            break
    if len(mytokens) > MAX_NGRAM_VOC:
        print('Vocabulary size too large, skipping n-gram filtering')
        return myvocab
    old_len = len(myvocab)
    # token cosine similarity
    max_exp = 21
    vectorizer = CountVectorizer(analyzer='char', max_features=2 ** max_exp, ngram_range=(1,4), strip_accents=None, lowercase=True, max_df=1.0)
    firstset = set(compute_deletions(mytokens, vectorizer, threshold))
    vectorizer = TfidfVectorizer(analyzer='char', max_features=2 ** max_exp, ngram_range=(1,4), strip_accents=None, lowercase=True, max_df=1.0, sublinear_tf=True, binary=True)
    secondset = set(compute_deletions(mytokens, vectorizer, threshold))
    for token in firstset.intersection(secondset):
        del myvocab[token]
    if verbose is True:
        print(sorted(firstset.intersection(secondset)))
    print_changes('ngrams', old_len, len(myvocab))
    return myvocab


def compute_deletions(mytokens, vectorizer, threshold):
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
