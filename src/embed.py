# Demonstration using word embeddings to perform semantic searches
#
# Running `make demo` will execute all of the mini demos with demo(). Semantic
# search will use whatever is specified in config/embed.py for its search query.
#
# Script can be run by passing a cli flag ('-s', '--search') which will only
# run the semantic search demo for the input string provided.
# This can be run using `python demo.py --search "<SEARCH_QUERY>"` or  `make
# search_query=<SEARCH_QUERY>`.

import getopt
import sys
from textwrap import fill as wrap
import yaml

from sentence_transformers import SentenceTransformer, util


with open("../config/embed.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


def print_title(
    text: str,
    char: str = "-",
    line_lim: int = config["lineLim"],
    double: bool = False,
    colourise=True,
):
    """
    Helper function to print text as a title nicely to stdout
    """
    if colourise:
        text = f"\033[1m{text}\033[0m"

    if double:  # Underline above & below text
        print(line_lim * char + f"\n{text}\n" + line_lim * char)
    else:
        print(f"\n\n{text}\n" + line_lim * char)


def print_wrap(text: str, line_lim: int = config["lineLim"]):
    """
    Helper function to wrap text to line_lim number of characters when
    printing to stdout
    """
    print(wrap(text, line_lim) + "\n")


def demo(search_query: str = None):
    """
    Script outlining a demo of word embeddings and semantic searching. If a
    search_query is provided, bypasses word embedding demos and only runs
    semantic search for the input query provided, else runs everything and
    defaults to the query specified in config/embed.py.
    """
    # Initiate model and create embeddings for sentence corpus
    print_title(config["text"]["title"], char="=", double=True)
    model = SentenceTransformer("all-MiniLM-L6-v2")

    sentences = [
        "The cat sits outside",
        "A man is playing guitar",
        "I love animals",
        "The new movie is awesome",
        "The cat plays in the garden",
        "A woman watches TV",
        "The new movie is so great",
        "Do you like pizza?",
    ]

    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)

    # Only run sentence embedding demo's if no custom search query provided
    if search_query == None:
        # Embedding a Sentence
        print_title(config["text"]["embedding"]["title"])
        print_wrap(config["text"]["embedding"]["description"])

        print(f"\nSentence: {sentences[0]}\n")
        print(f"Embedding:\n{sentence_embeddings[0]}\n")

        cosine_scores = util.cos_sim(sentence_embeddings, sentence_embeddings)

        pairs = []
        for i in range(len(cosine_scores) - 1):
            for j in range(i + 1, len(cosine_scores)):
                pairs.append({"index": [i, j], "score": cosine_scores[i][j]})

        pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)

        # Cosine Similarity
        print_title(config["text"]["similarity"]["title"])
        print_wrap(config["text"]["similarity"]["description"])

        print("Input Sentences")
        for s in sentences:
            print(f"\t - {s}")

        header = "\nSentence 1\t Sentence 2\t Similarity Score"
        print_title(header.expandtabs(config["tabWidth"]), colourise=False)

        for pair in pairs[0:10]:
            i, j = pair["index"]
            row = f"{sentences[i]}\t {sentences[j]}\t Score: {pair['score']:.4f}"
            print(row.expandtabs(config["tabWidth"]))

        # Semantic Search
        print_title(config["text"]["semantic"]["title"])
        print_wrap(config["text"]["semantic"]["description"])

        search_query = config["semanticQuery"]

    search_query_embedding = model.encode(search_query, convert_to_tensor=True)

    search_results = []
    for (s, e) in zip(sentences, sentence_embeddings):
        cosine_score = util.cos_sim(search_query_embedding, e)
        search_results.append({"sentence": s, "score": cosine_score})

    search_results = sorted(search_results, key=lambda x: x["score"], reverse=True)

    print(f"Search Query: \033[1m{search_query}\033[0m")
    print(f"\nResults by relevance:")

    header = "Sentence:\t\t Score:"
    print_title(header.expandtabs(config["tabWidth"]), colourise=False)
    for hit in search_results:
        row = f"{hit['sentence']}\t\t {float(hit['score']):.4f}"
        print(row.expandtabs(config["tabWidth"]))


if __name__ == "__main__":
    search_query = None

    try:  # PARSE CLI FLAGS
        opts, args = getopt.getopt(sys.argv[1:], "s:", ["search="])
    except getopt.GetoptError:
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-s", "--search"):
            search_query = arg

    demo(search_query=search_query)
