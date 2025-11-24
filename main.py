# review of tokenization using moby dick
## Andrew Richard
## 2025-11-24

import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
#
from nltk import PorterStemmer
from nltk.tokenize import sent_tokenize, word_tokenize, regexp_tokenize
from nltk.corpus import stopwords
#
from nltk.stem import PorterStemmer, WordNetLemmatizer
from wordcloud import WordCloud

# run once at first
# nltk.download("stopwords")
# nltk.download("punkt_tab")
# nltk.download("wordnet")


def main():

    with open("./data/moby_dick.txt") as f:
        moby = f.read()

    print(f"First 1000 words: {moby[:1000]}")

# split into sentences
    sentences = sent_tokenize(moby)
    print(sentences[1000])

# split into tokens where each word is a token
# this counts punctuation as tokens.
    i = 1000
    print(f"Sentences: {sentences[i]}\n")
    word_tokens = word_tokenize(sentences[i])
    print(f"word_tokens: {word_tokens}\n")

# regexp tokenize allows us to specify a regulat expression pattern to define a token
    regexp_tokens = regexp_tokenize(sentences[1000], r'\w+')
    print(f"regexp_tokens: {regexp_tokens}\n")


# add more characters and count the most frequent tokens
    moby_counter = Counter(regexp_tokenize(moby, r'[-\'\w]+'))
    print(f"Counter - whale:{moby_counter['whale']}\n")
    print(f"Counter - Ishmael: {moby_counter['Ishmael']}\n")

    print(f"Length of moby_counters: {len(moby_counter)} unique words\n")
    print(f"Total words of moby_counters: {sum(moby_counter.values())} words\n")
    print(f"Most common words: {moby_counter.most_common()[:10]}\n")



# The most common words include lots of articles and "stopwords" which can be removed
    stop_words = set(stopwords.words("english"))
    print(f"===Stop words===\n\n{stop_words}\n")


# modify counter
    moby_counter = Counter([x.lower() for x in regexp_tokenize(moby, r'[-\'\w]+') if x.lower() not in stop_words])
    print(f"New moby_counter length: {len(moby_counter)}\n")
    print(f"New moby_counter most_common: {moby_counter.most_common()[:10]}\n")


# now look at stemming and lemmatization.

# PorterStemmer
    porter = PorterStemmer()

# gets the stem of a words
    print(porter.stem('whales'))
    print(porter.stem('numerical'))

# update moby_counter again
    moby_counter = Counter([porter.stem(x.lower()) for x in regexp_tokenize(moby, r'[-\'\w]+') if x.lower() not in stop_words])
    print(f"porterstemmer length of moby_counter: {len(moby_counter)}\n")
    print(f"PorterStemmer moby_counter most_common: {moby_counter.most_common()[:10]}\n")

    wnl = WordNetLemmatizer()
    print(f"Lemmatizer: {wnl.lemmatize("whales")}\n")

    moby_counter = Counter(wnl.lemmatize(x.lower()) for x in regexp_tokenize(moby, r'[-\'\w]+') if x.lower() not in stop_words)
    print(f"Lemmatized moby_counter: {len(moby_counter)}\n")
    print(f"Lemmatizer moby_counter most_common: {moby_counter.most_common()[:10]}\n")

# visualize distribution of words
    print(f"Type of moby_counter.values(): {type(moby_counter.values())}\n")
    plt.figure()
    plt.hist(pd.Series(moby_counter.values()), bins = 50, edgecolor="#00CC96")
    plt.title("Distribution of words -- lemmatized")
    plt.xlabel("Number of appearances")
    plt.ylabel("Number of words")
    describe_text = str(pd.Series(moby_counter.values()).describe())
    plt.text(s=describe_text, x=800, y=10000)
    plt.show()

    wc = WordCloud()

    wc.generate_from_frequencies(moby_counter)
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
