# embed.py
from textwrap import fill as wrap
import yaml

from sentence_transformers import SentenceTransformer, util


with open("../config/embed.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


# #####################################################################
# Helper Print Functions
# #####################################################################
def print_title(
    text: str,
    char: str = "-",
    line_lim: int = config["lineLim"],
    double: bool = False,
    colourise=True,
):
    if colourise:
        text = f"\033[1m{text}\033[0m"

    if double:  # Underline above & below text
        print(line_lim * char + f"\n{text}\n" + line_lim * char)
    else:
        print(f"\n\n{text}\n" + line_lim * char)


def print_wrap(text: str, line_lim: int = config["lineLim"]):
    print(wrap(text, line_lim) + "\n")


# #####################################################################
# Initiate model and create embeddings for sentence corpus
# #####################################################################
model = SentenceTransformer("all-MiniLM-L6-v2")

print_title(config["text"]["title"], char="=", double=True)

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


# #####################################################################
# Embedding a Sentence
# #####################################################################
print_title(config["text"]["embedding"]["title"])
print_wrap(config["text"]["embedding"]["description"])

print(f"\nSentence: {sentences[0]}\n")
print(f"Embedding:\n{sentence_embeddings[0]}\n")

cosine_scores = util.cos_sim(sentence_embeddings, sentence_embeddings)

# Find the pairs with the highest cosine similarity scores
pairs = []
for i in range(len(cosine_scores) - 1):
    for j in range(i + 1, len(cosine_scores)):
        pairs.append({"index": [i, j], "score": cosine_scores[i][j]})

pairs = sorted(pairs, key=lambda x: x["score"], reverse=True)


# #####################################################################
# Cosine Similarity
# #####################################################################
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


# #####################################################################
# Semantic Search
# #####################################################################
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
    print(
        (f"{hit['sentence']}\t\t {float(hit['score']):.4f}").expandtabs(
            config["tabWidth"]
        )
    )
