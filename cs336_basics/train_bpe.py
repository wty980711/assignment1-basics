from __future__ import annotations
from collections import Counter, defaultdict
from pathlib import Path
import heapq
import regex as re

PRETOKENIZATION_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"

def _pretoken_counts(text: str, special_tokens: list[str]) -> Counter[tuple[bytes, ...]]:
    """Return weighted UTF-8-byte pre-tokens, excluding special-token spans."""
    if special_tokens:
        # Prefer a longer special token when two tokens share a prefix.
        # re.escape: let reg treat special token as content rather than special operation symbols
        special_pattern = re.compile("|".join(re.escape(token) for token in sorted(special_tokens, key=len, reverse=True)))
        text_spans = special_pattern.split(text)
    else:
        text_spans = [text]

    counts: Counter[tuple[bytes, ...]] = Counter()
    pretoken_pattern = re.compile(PRETOKENIZATION_PATTERN)
    for span in text_spans:
        for match in pretoken_pattern.finditer(span):
            encoded = match.group().encode("utf-8")
            counts[tuple(bytes([byte]) for byte in encoded)] += 1
    return counts

def _merge_pair_in_word(word: tuple[bytes,...], pair: tuple[bytes, bytes]) -> tuple[bytes,...]:
    new_word = []
    i = 0
    while i < len(word):
        if (word[i] == pair[0] and i < len(word) - 1 and word[i+1] == pair[1]):
            new_word.append(word[i] + word[i+1])
            i += 2
        else:
            new_word.append(word[i])
            i += 1
    return tuple(new_word)

def _merge(
    words: Counter[tuple[bytes, ...]], 
    vocab: dict[int, bytes], 
    vocab_size: int
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """"Merge byte pairs to new token"""
    merge_history: list[tuple[bytes, bytes]] = []

    pair_counts: Counter[tuple[bytes, bytes]] = Counter()
    pair_to_words: dict[tuple[bytes, bytes], set[tuple[bytes,...]]] = defaultdict(set)
    # Record the frequency of words from pre-tokenization
    for word, freq in words.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += freq
            pair_to_words[pair].add(word)

    version: Counter[tuple[bytes, bytes]] = Counter() # version of the pair -> lazy cleanup
    heap: list[tuple[int, tuple[bytes, bytes], int]] = [] # max heap to find the most frequent pair

    for pair, freq in pair_counts.items():
        version[pair] += 1
        heapq.heappush(heap, (-freq, pair, version[pair]))

    while len(vocab) < vocab_size and heap:
        candidates: list[tuple[int, tuple[bytes, bytes], int]] = []

        freq = heap[0][0]  
        while (heap and heap[0][0] == freq):
            element = heapq.heappop(heap)
            if (element[2] != version[element[1]]):
                continue
            candidates.append(element)

        if not candidates:
            continue

        candidates.sort(key=lambda x: x[1], reverse=True)

        pair_to_merge = candidates[0][1]

        for other in candidates[1:]:
            heapq.heappush(heap, other)

        merged_pair = pair_to_merge[0] + pair_to_merge[1]

        merge_history.append(pair_to_merge)
        vocab[len(vocab)] = merged_pair

        words_to_update = pair_to_words[pair_to_merge].copy()
        changed_pairs = set()

        for word in words_to_update:
            freq = words.pop(word)
            # remove old pairs
            checked_pairs = set()
            for old_pair in zip(word, word[1:]):
                pair_counts[old_pair] -= freq
                changed_pairs.add(old_pair)
                if pair_counts[old_pair] == 0:
                    pair_counts.pop(old_pair)

                if old_pair not in checked_pairs:
                    pair_to_words[old_pair].remove(word)
                    checked_pairs.add(old_pair)
                    if len(pair_to_words[old_pair]) == 0:
                        pair_to_words.pop(old_pair)
            # merge pairs in word
            new_word = _merge_pair_in_word(word, pair_to_merge)
            words[new_word] += freq
            # add new pairs
            for new_pair in zip(new_word, new_word[1:]):
                changed_pairs.add(new_pair)
                pair_counts[new_pair] += freq
                pair_to_words[new_pair].add(new_word)

        for pair in changed_pairs:
            version[pair] += 1
            if pair_counts[pair] > 0:
                heapq.heappush(heap, (-pair_counts[pair], pair, version[pair]))

    return (vocab, merge_history)


def train_bpe(
    input_path: str | Path,
    vocab_size: int,
    special_tokens: list[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    """Train a byte-level BPE tokenizer using the assignment's GPT-2-style pre-tokenizer."""
    if vocab_size < 256 + len(special_tokens):
        raise ValueError("vocab_size must accommodate the byte vocabulary and all special tokens")
    if any(not token for token in special_tokens):
        raise ValueError("special tokens must be non-empty")
    if len(set(special_tokens)) != len(special_tokens):
        raise ValueError("special tokens must be unique")

    special_token_bytes = [token.encode("utf-8") for token in special_tokens]
    initial_tokens = special_token_bytes + [bytes([byte]) for byte in range(256)]
    vocab = dict(enumerate(initial_tokens))

    text = Path(input_path).read_text(encoding="utf-8")
    words = _pretoken_counts(text, special_tokens)

    return _merge(words, vocab, vocab_size)