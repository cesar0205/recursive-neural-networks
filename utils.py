import numpy as np


class Node():
    def __init__(self, word):
        self.left = None;
        self.right = None
        self.word = word
        self.rating = None;

    def __repr__(self):
        if (self.left is not None and self.right is not None):
            return str(self.left.word) + "-" + str(self.right.word)
        else:
            return str(self.word)


def get_word(string, index):
    word = ""
    while (string[index] not in ["(", ")"]):
        word += string[index]
        index += 1
    return word, index


def create_tree(string2, binary_classification=True):
    i = 0
    nodes = []
    ratings = []
    while (i < len(string2)):
        # print("Testing element", string[i])
        if (string2[i] == " "):
            # print("Found empty")
            i += 1
        elif (string2[i] == "("):
            # print("Found (")
            ratings.append(string2[i + 1])
            # print("Updated ratings", ratings)
            i += 3
        elif (string2[i] == ")"):
            # print("Found )")
            node_right = nodes.pop()
            node_left = nodes.pop()
            rating = ratings.pop()

            node = Node(None)
            node.left = node_left;
            node.right = node_right;
            if (binary_classification):
                if (int(rating) <= 2):
                    node.rating = 0;
                else:
                    node.rating = 1;
            # print("creating double node", node.left, node.right)

            nodes.append(node)
            # print("After update", nodes, ratings)
            i += 1
        else:
            # print("Found characters")
            word, last_index = get_word(string2, i)
            # print("Word obtained", word, i)

            rating = ratings.pop()
            node = Node(word.lower())
            if (binary_classification):
                if (int(rating) <= 2):
                    node.rating = 0;
                else:
                    node.rating = 1;
            nodes.append(node)
            # print("After update", nodes, ratings)
            i = last_index
            i += 1

    return nodes[0]


def print_sentence(node, depth=0):
    if (node is None):
        return;
    left_name = print_sentence(node.left, depth);
    right_name = print_sentence(node.right, depth);
    if (node.word is not None):
        # print(node.word)
        return node.word
    else:
        return left_name + " " + right_name


def load_data():
    print('Loading word vectors...')
    embeddings = np.zeros((400000, 50), dtype=np.float32)
    word2idx = {}
    counter = 0
    with open('./glove.6B/glove.6B.50d.txt', encoding='utf-8') as f:
        # is just a space-separated text file in the format:
        # word vec[0] vec[1] vec[2] ...
        for line in f:
            values = line.split()
            word = values[0]
            vec = np.asarray(values[1:], dtype='float32')
            embeddings[counter] = vec
            word2idx[word] = counter;
            counter += 1
        print('Found %s word vectors.' % len(word2idx))

    train_sentences = []
    with open("./trees/train.txt") as f:
        for line in f:
            root = create_tree(line[:-1])
            # print_sentence(root, 0)
            train_sentences.append(root)

    test_sentences = []
    with open("./trees/test.txt") as f:
        for line in f:
            root = create_tree(line[:-1])
            # print_sentence(root, 0)
            test_sentences.append(root)

    return embeddings, word2idx, train_sentences, test_sentences


# Recursive function to transform trees to lists [indexes, relations, parents, words, ratings]
def tree_to_list(node, word2idx, last_idx, indexes=[], left_children=[], right_children=[], words=[], ratings=[]):
    if (node is None):
        return -1;

    left_idx = tree_to_list(node.left, word2idx, last_idx, indexes, left_children, right_children, words, ratings)
    right_idx = tree_to_list(node.right, word2idx, left_idx, indexes, left_children, right_children, words, ratings)

    ratings.append(node.rating)
    left_children.append(left_idx);
    right_children.append(right_idx);
    if (right_idx != -1):
        curr_idx = right_idx + 1;
    else:
        curr_idx = last_idx + 1

    indexes.append(curr_idx)

    if (node.word is not None):
        if (node.word in word2idx):
            word_idx = word2idx[node.word]
            # complete_words.append(node.word)
        else:
            word_idx = word2idx['unk']
            # complete_words.append('unk')
    else:
        word_idx = -1
        # complete_words.append('NODE')

    words.append(word_idx)
    return curr_idx;

#Main function to convert tree to a list
def convert_tree_to_list(word2idx, node):
    indexes = []
    left_children = []
    right_children = []
    words = []
    ratings = []

    tree_to_list(node, word2idx, -1, indexes, left_children, right_children, words, ratings);
    # print(indexes, left_children, right_children, words, ratings)
    return np.array([left_children, right_children, words, ratings]).T.astype(int)


def transform_tree_set_to_list(word2idx, sentences):
    transformations = []
    for i, sentence in enumerate(sentences):
        transformations.append(convert_tree_to_list(word2idx, sentence))
    return transformations;

