from typing import List
from transformers import BertTokenizer, BertModel


def extract_bert_embeddings(tokenizer, model, texts: List[str]):
    """

    Args:
        texts: Batch of Texts e.g questions/anwers or both
        batch_size: len(texts)
        em_size = 768
        seq_len = (max_seq_len_of_batch + 2) , 2= [CLS, SEP]

    Returns: Bert Embeddings from last hidden layers of shape seq_len * batch_size * em_size

    """

    """
    tokenizer returns a dictionary of tensors of size  seq_len * batch_size
    input_ids: ids of each word in bert_base_uncased_vocab
    token_ids: segment id, whether the words belong to different segments
    attention_masks: tensor of 1s and 0s. 1 if word is not pad and vice versa
    """
    inputs = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)

    """
    final_layer_embedding_op: batch_size * seq_len * em_size
    pooled_op: batch_size * 1 * em_size Flattened vector: 1 Sentence embedding
    all_hidden_layers: Tuple of n_heads+1 (Input Embeddings) batch_size * seq_len * em_size embeddings
    """
    final_layer_embedding_op, pooled_op, all_hidden_layers = model(**inputs, output_hidden_states=True)
    return final_layer_embedding_op


if __name__ == '__main__':
    sample_batch = ['Hello, How are you?', 'Good. Thanks. How are you?']
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    embeddings = extract_bert_embeddings(tokenizer, model, texts=sample_batch)
   