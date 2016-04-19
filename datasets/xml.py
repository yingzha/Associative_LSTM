from fuel.datasets import Dataset
from fuel.streams import DataStream
import numpy
import string

def xml_generate(num_seq, max_nest=4, max_iter=50,
                 min_char=1, max_char=10):
    alphabet = list(string.ascii_lowercase)
    #alphabet + < + / + >
    one_hot_encoding = numpy.eye(29)
    tags = []
    tags_encoding = []
    tags_mask = []

    def get_tag_encoding(tag):
        encoding = []
        for i in tag:
            if i == '<':
                encoding.append(one_hot_encoding[-3])
            elif i == '/':
                encoding.append(one_hot_encoding[-2])
            elif i == '>':
                encoding.append(one_hot_encoding[-1])
            else:
                encoding.append(one_hot_encoding[alphabet.index(i)])
        return encoding

    def get_tag():
        num = numpy.random.randint(low=min_char, high=max_char)
        indices = numpy.random.randint(low=0, high=len(alphabet), size=num)
        tag = ''
        for i in indices:
            tag += alphabet[i]
        return tag

    def padding(batch, mask):
        max_length = max([i.shape[0] for i in batch])
        padded_data = numpy.zeros((max_length, len(batch), 29))
        padded_mask = numpy.zeros((max_length, len(batch)))
        for i, seq in enumerate(batch):
            padded_data[:seq.shape[0], i, :] = seq

        for i, seq in enumerate(mask):
            padded_mask[:seq.shape[0], i] = 1

        return padded_data, padded_mask

    for i in xrange(num_seq):
        each_tag = []
        each_tag_encoding = []
        each_tag_mask = []
        stack_tags = []
        iters = 0
        pop = 0
        push = 0
        while iters < max_iter:
            if not stack_tags:
                init_tag = get_tag()
                stack_tags.append('<' + init_tag + '>')
                each_tag_encoding.extend(get_tag_encoding('<' + init_tag + '>'))
                each_tag.append(stack_tags[-1])
                each_tag_mask.extend([1] + [0] * (len(init_tag) + 1))
                iters += 1
                push += 1

            pop_prob = numpy.random.binomial(1, .5)

            if pop_prob:
                pop_tag = stack_tags.pop()
                each_tag.append(pop_tag[0] + '/' + pop_tag[1:])
                each_tag_encoding.extend(get_tag_encoding(
                    pop_tag[0] + '/' + pop_tag[1:]))
                each_tag_mask.extend([1] * (len(pop_tag) + 1))
                iters += 1
                pop += 1

            else:
                tag = get_tag()
                stack_tags.append('<' + tag + '>')
                each_tag_encoding.extend(get_tag_encoding('<' + tag + '>'))
                each_tag_mask.extend([1] + [0] * (len(tag) + 1))
                each_tag.append(stack_tags[-1])
                iters += 1
                push += 1

            if push - pop >= max_nest:
                break

        if stack_tags:
            for i in xrange(len(stack_tags)):
                pop_tag = stack_tags.pop()
                each_tag.append(pop_tag[0] + '/' + pop_tag[1:])
                each_tag_encoding.extend(get_tag_encoding(
                    pop_tag[0] + '/' + pop_tag[1:]))
                each_tag_mask.extend([1] * (len(pop_tag) + 1))

        tags.append(''.join(each_tag))
        tags_encoding.append(numpy.asarray(each_tag_encoding))
        tags_mask.append(numpy.asarray(each_tag_mask))

    tags_encoding, tags_mask = padding(tags_encoding, tags_mask)
    return tags, tags_encoding, tags_mask

class XML(Dataset):
    provides_sources = ('features', 'masks')
    example_iteration_scheme = None

    def __init__(self, num_seq, max_nest=4, max_iter=50,
                 min_char=1, max_char=10):
        self.num_seq = num_seq
        self.max_nest = max_nest
        self.max_iter = max_iter
        self.min_char = min_char
        self.max_char = max_char
        super(XML, self).__init__()

    def get_data(self, state=None, request=None):
        tags, data, masks = xml_generate(self.num_seq, self.max_nest,
                                         self.max_iter, self.min_char,
                                         self.max_char)
        return (data, masks)

if __name__ == "__main__":
    dataset = XML(5)
    stream = DataStream(dataset)
    data = dataset.get_data()
