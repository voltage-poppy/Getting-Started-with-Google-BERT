from transformers import BertModel, BertTokenizer

def main():
    model = BertModel.from_pretrained('bert-base-uncased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    sentence = 'I love Paris'
    tokens = tokenizer.tokenize(sentence)
    tokens = ['[CLS]'] + tokens + ['[SEP]']
    print(tokens)
    assert tokens == ['[CLS]', 'i', 'love', 'paris', '[SEP]']

    tokens = tokens + ['[PAD]'] + ['[PAD]']
    print(tokens)
    assert tokens == ['[CLS]', 'i', 'love', 'paris', '[SEP]', '[PAD]', '[PAD]']

    attention_mask = [1 if i!= '[PAD]' else 0 for i in tokens]
    print(attention_mask)
    assert attention_mask == [1, 1, 1, 1, 1, 0, 0]

    token_ids = tokenizer.convert_tokens_to_ids(tokens)


if __name__ == "__main__":
    main()
