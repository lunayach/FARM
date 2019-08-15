# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes."""
from __future__ import absolute_import, division, print_function, unicode_literals

import collections
import logging
from io import open
import os
import unicodedata
import six
import json
from google.protobuf.json_format import MessageToJson

import farm.modeling.sentencepiece_pb2 as sentencepiece_pb2

from pytorch_transformers.tokenization_bert import WordpieceTokenizer, load_vocab
from pytorch_transformers.tokenization_bert import BertTokenizer as BertTokenizerHF
from pytorch_transformers.tokenization_bert import BasicTokenizer as BasicTokenizerHF

from pytorch_transformers.tokenization_xlnet import XLNetTokenizer as XLNetTokenizerHF

logger = logging.getLogger(__name__)

SPIECE_UNDERLINE = u'▁'



class XLNetTokenizer(XLNetTokenizerHF):
    def __init__(self, vocab_file, max_len=None,
                 do_lower_case=False, remove_space=True, keep_accents=False,
                 bos_token="<s>", eos_token="</s>", unk_token="<unk>", sep_token="<sep>",
                 pad_token="<pad>", cls_token="<cls>", mask_token="<mask>",
                 additional_special_tokens=["<eop>", "<eod>"], **kwargs):
        super().__init__(vocab_file, max_len,
                 do_lower_case, remove_space, keep_accents,
                 bos_token, eos_token, unk_token, sep_token,
                 pad_token, cls_token, mask_token,
                 additional_special_tokens, **kwargs)
        self.spt = sentencepiece_pb2.SentencePieceText()


    def _tokenize(self, text, return_unicode=True, sample=False):
        """
        Tokenize a string. Copied from parent class but with support for token to text alignment
        """

        if not sample:
            pieces, self.offsets = self.tokenize_with_offsets(text)
            self.initial_mask = word_initial_mask(pieces)

        else:
            raise NotImplementedError
            # pieces = self.sp_model.SampleEncodeAsPieces(text, 64, 0.1)
        new_pieces = []
        for piece in pieces:
            if len(piece) > 1 and piece[-1] == ',' and piece[-2].isdigit():
                cur_pieces = self.sp_model.EncodeAsPieces(
                    piece[:-1].replace(SPIECE_UNDERLINE, ''))
                if piece[0] != SPIECE_UNDERLINE and cur_pieces[0][0] == SPIECE_UNDERLINE:
                    if len(cur_pieces[0]) == 1:
                        cur_pieces = cur_pieces[1:]
                    else:
                        cur_pieces[0] = cur_pieces[0][1:]
                cur_pieces.append(piece[-1])
                new_pieces.extend(cur_pieces)
            else:
                new_pieces.append(piece)

        return new_pieces

    def tokenize_with_offsets(self, text):
        serialized_proto = self.sp_model.encode_as_serialized_proto(text)
        self.spt.ParseFromString(serialized_proto)
        pieces_and_spans = json.loads(MessageToJson(self.spt))["pieces"]
        pieces = []
        alignment = []
        for ps in pieces_and_spans:
            piece = ps["piece"]
            begin = ps["begin"]
            end = ps["end"]
            pieces.append(piece)
            alignment.append((begin, end))
        offsets = [x[0] for x in alignment]
        return pieces, offsets


class BasicTokenizer(BasicTokenizerHF):
    def __init__(self, do_lower_case=True, never_split=None, never_split_chars=None, tokenize_chinese_chars=True):
        """ Constructs a BasicTokenizer.

        Args:
            **do_lower_case**: Whether to lower case the input.
            **never_split**: (`optional`) list of str
                Kept for backward compatibility purposes.
                Now implemented directly at the base class level (see :func:`PreTrainedTokenizer.tokenize`)
                List of token not to split.
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be desactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = never_split
        self.tokenize_chinese_chars = tokenize_chinese_chars
        self.never_split_chars = never_split_chars

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char, excluded_chars=self.never_split_chars):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]


class BertTokenizer(BertTokenizerHF):

    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None, never_split_chars=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True, **kwargs):
        """Constructs a BertTokenizer.

        Args:
            **vocab_file**: Path to a one-wordpiece-per-line vocabulary file
            **do_lower_case**: (`optional`) boolean (default True)
                Whether to lower case the input
                Only has an effect when do_basic_tokenize=True
            **do_basic_tokenize**: (`optional`) boolean (default True)
                Whether to do basic tokenization before wordpiece.
            **never_split**: (`optional`) list of string
                List of tokens which will never be split during tokenization.
                Only has an effect when do_basic_tokenize=True
            **tokenize_chinese_chars**: (`optional`) boolean (default True)
                Whether to tokenize Chinese characters.
                This should likely be desactivated for Japanese:
                see: https://github.com/huggingface/pytorch-pretrained-BERT/issues/328
        """
        super(BertTokenizer, self).__init__(vocab_file, do_lower_case=True, do_basic_tokenize=True, never_split=None, never_split_chars=None,
                 unk_token="[UNK]", sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]",
                 mask_token="[MASK]", tokenize_chinese_chars=True, **kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                  never_split=never_split,
                                                  never_split_chars=never_split_chars,
                                                  tokenize_chinese_chars=tokenize_chinese_chars)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)


    def add_custom_vocab(self, custom_vocab_file):
        self.vocab = self._load_custom_vocab(custom_vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    def _load_custom_vocab(self, custom_vocab_file):
        custom_vocab = {}
        unique_custom_tokens = set()
        idx = 0
        with open(custom_vocab_file, "r", encoding="utf-8") as reader:
            while True:
                token = reader.readline()
                if not token:
                    break

                if token not in unique_custom_tokens:
                    if token not in self.vocab.keys():
                        key = "[unused{}]".format(idx)
                        custom_vocab[key] = token.strip()
                        idx += 1
                        unique_custom_tokens.add(token)
                    else:
                        logger.info("Dropped custom token (already in original vocab): {}".format(token))
                else:
                    logger.info("Dropped custom token (duplicate): {}".format(token))
        # merge vocabs
        update_count = 0
        updated_vocab = []
        for k,v in self.vocab.items():
            if k in custom_vocab.keys():
                updated_vocab.append((custom_vocab[k], v))
                update_count += 1
            else:
                updated_vocab.append((k, v))
        self.vocab = collections.OrderedDict(updated_vocab)

        if update_count < len(custom_vocab):
            logger.warning("Updated vocabulary only with {} out of {} tokens from supplied custom vocabulary. The original vocab might not have enough unused tokens.".format(update_count, len(custom_vocab)))
        else:
            logger.info("Updated vocabulary with {} out of {} tokens from custom vocabulary.".format(update_count, len(custom_vocab)))
        return self.vocab

def word_initial_mask(tokens):
    # TODO: Deal with punctuation like "."
    style = "wordpiece"
    for t in tokens:
        if t[0] == "▁":
            style = "sentencepiece"
            break
    if style == "wordpiece":
        mask = wordpiece_initial_mask(tokens)
    elif style == "sentencepiece":
        mask = sentencepiece_initial_mask(tokens)
    return mask

def wordpiece_initial_mask(tokens):
    mask = []
    for t in tokens:
        if t[:2] == "##":
            mask.append(False)
        elif t[0].isalpha():
            mask.append(True)
        else:
            mask.append(False)
    return mask

def sentencepiece_initial_mask(tokens):
    mask = []
    for i, t in enumerate(tokens):
        try:
            if t[0] == "▁" and len(t) > 1:
                mask.append(True)
            elif t[0] != "▁" and t[0].isalpha() and tokens[i - 1] == "▁":
                mask.append(True)
            else:
                mask.append(False)
        except IndexError:
            mask.append(False)

    return mask


def tokenize_with_metadata(text, tokenizer, max_seq_len):
    # split text into "words" (here: simple whitespace tokenizer)
    tokens = None
    offsets = None
    start_of_word = None

    if type(tokenizer) == BertTokenizer:
        words = text.split(" ")
        word_offsets = []
        cumulated = 0
        for idx, word in enumerate(words):
            word_offsets.append(cumulated)
            cumulated += len(word) + 1  # 1 because we so far have whitespace tokenizer
            # TODO: This is wrong if there are multiple spaces in a row

        # split "words"into "subword tokens"
        tokens, offsets, start_of_word = _words_to_tokens(
            words, word_offsets, tokenizer, max_seq_len
        )
    elif type(tokenizer) == XLNetTokenizer:
        tokens = tokenizer.tokenize(text)[:max_seq_len - 2]
        offsets = tokenizer.offsets[:max_seq_len - 2]
        start_of_word = tokenizer.initial_mask[:max_seq_len - 2]

    return{"tokens": tokens, "offsets": offsets, "start_of_word": start_of_word}


def _words_to_tokens(words, word_offsets, tokenizer, max_seq_len):
    tokens = []
    token_offsets = []
    start_of_word = []
    for w, w_off in zip(words, word_offsets):
        # Get tokens of single word
        tokens_word = tokenizer.tokenize(w)

        # Sometimes the tokenizer returns no tokens
        if len(tokens_word) == 0:
            continue
        tokens += tokens_word

        # get gloabl offset for each token in word + save marker for first tokens of a word
        first_tok = True
        for tok in tokens_word:
            token_offsets.append(w_off)
            w_off += len(tok.replace("##", ""))
            if first_tok:
                start_of_word.append(True)
                first_tok = False
            else:
                start_of_word.append(False)

    # Clip at max_seq_length. The "-2" is for CLS and SEP token
    tokens = tokens[: max_seq_len - 2]
    token_offsets = token_offsets[: max_seq_len - 2]
    start_of_word = start_of_word[: max_seq_len - 2]

    assert len(tokens) == len(token_offsets) == len(start_of_word)
    return tokens, token_offsets, start_of_word

def _is_punctuation(char, excluded_chars=None):
    """Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if excluded_chars:
        if char in excluded_chars:
            return False

    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False