from os import path
import logging
from collections import Counter
from transformers import BertTokenizerFast #对文本进行tokenizer，得到文本的tokens，token_span，token_ids，token_mask等


class Tokenizer(BertTokenizerFast):
    @classmethod
    def loads(cls, vocab_path, do_lower_case=True):
        return cls.from_pretrained(
            path.dirname(vocab_path),
            vocab_files_names={'vocab_file': path.basename(vocab_path)},
            do_lower_case=do_lower_case
        )

    def transform(self, texts, is_tokenized=False, max_length=None, return_tensors=None, return_texts=False, return_tokens=False, do_lower_case=None, return_offsets=False):
        assert return_tensors in ("pt", "np", "tf", None)
        if do_lower_case:
            if is_tokenized:
                precased_texts = [[x.lower() for x in words]
                                  for words in texts]
            else:
                precased_texts = [x.lower() for x in texts]
        else:
            precased_texts = texts
        result = self(
            precased_texts,
            add_special_tokens=True,
            return_length=False,
            return_special_tokens_mask=False,
            return_attention_mask=True,
            return_overflowing_tokens=True,
            return_offsets_mapping=return_offsets,
            max_length=max_length,
            # is_split_into_words=is_tokenized,
            padding='max_length' if max_length else 'do_not_pad',
            truncation=True if max_length else 'do_not_truncate',
            return_tensors=return_tensors
        )
        if return_texts:
            result['texts'] = texts
        if max_length:
            overflow_mapping = result['overflow_to_sample_mapping']
            if return_tensors == 'tf':
                overflow_mapping = overflow_mapping.numpy()
            elif return_tensors == 'pt':
                overflow_mapping = overflow_mapping.tolist()
            seg_counts = Counter(overflow_mapping)
            not_overflowed_indexes = []
            for it, text in enumerate(texts):
                seg_count = seg_counts[overflow_mapping[it]]
                if seg_count > 1:
                    logging.warning(f'Ignored overflowed text: "{text}"')
                else:
                    not_overflowed_indexes.append(it)
            for k, v in result.items():
                if isinstance(v, list):
                    result[k] = [x for ix, x in enumerate(v) if ix in not_overflowed_indexes]
                elif return_tensors == 'tf':
                    import tensorflow as tf
                    result[k] = tf.gather(v, indices=not_overflowed_indexes)
                else:
                    result[k] = v[not_overflowed_indexes]
        del result['overflow_to_sample_mapping']
        if return_tokens:
            result['tokens'] = [
                self.convert_ids_to_tokens(x, skip_special_tokens=False)
                for x in result['input_ids']
            ]
        return result


if __name__ == '__main__':
    # tokenizer = Tokenizer.loads('./data/biobert/vocab.txt', do_lower_case=False)
    tokenizer = Tokenizer.loads('../../data/datasets/biobert/biobert-base-cased-v1.1/vocab.txt', do_lower_case=False)
    result = tokenizer.transform(
        [
            'propagation.',
            'Propagation',
            'P53 protein, is a tumor-related protein.'
        ], 
        max_length=10, 
        return_texts=True, 
        return_tokens=True, 
        return_tensors=None, 
        do_lower_case=False
    )

    print(tokenizer.all_special_tokens)
    print(result)
    # text = ['i', 'am', 'a', 'happy', 'boy', '.']
    text = 'i am a happy boy.'
    x = tokenizer.tokenize(text)
    print(x)

