from tensorflow.keras import layers
import tensorflow_hub as hub
import tensorflow_text as text

ENCODER = {
    "bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3",
    "bert_en_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3",
    "bert_multi_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_multi_cased_L-12_H-768_A-12/3",
    "small_bert/bert_en_uncased_L-2_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-128_A-2/1",
    "small_bert/bert_en_uncased_L-2_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-256_A-4/1",
    "small_bert/bert_en_uncased_L-2_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-512_A-8/1",
    "small_bert/bert_en_uncased_L-2_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-2_H-768_A-12/1",
    "small_bert/bert_en_uncased_L-4_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-128_A-2/1",
    "small_bert/bert_en_uncased_L-4_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-256_A-4/1",
    "small_bert/bert_en_uncased_L-4_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-512_A-8/1",
    "small_bert/bert_en_uncased_L-4_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-4_H-768_A-12/1",
    "small_bert/bert_en_uncased_L-6_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-128_A-2/1",
    "small_bert/bert_en_uncased_L-6_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-256_A-4/1",
    "small_bert/bert_en_uncased_L-6_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-512_A-8/1",
    "small_bert/bert_en_uncased_L-6_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-6_H-768_A-12/1",
    "small_bert/bert_en_uncased_L-8_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-128_A-2/1",
    "small_bert/bert_en_uncased_L-8_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-256_A-4/1",
    "small_bert/bert_en_uncased_L-8_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-512_A-8/1",
    "small_bert/bert_en_uncased_L-8_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-8_H-768_A-12/1",
    "small_bert/bert_en_uncased_L-10_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-128_A-2/1",
    "small_bert/bert_en_uncased_L-10_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-256_A-4/1",
    "small_bert/bert_en_uncased_L-10_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-512_A-8/1",
    "small_bert/bert_en_uncased_L-10_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-10_H-768_A-12/1",
    "small_bert/bert_en_uncased_L-12_H-128_A-2": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-128_A-2/1",
    "small_bert/bert_en_uncased_L-12_H-256_A-4": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-256_A-4/1",
    "small_bert/bert_en_uncased_L-12_H-512_A-8": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-512_A-8/1",
    "small_bert/bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/small_bert/bert_en_uncased_L-12_H-768_A-12/1",
    "albert_en_base": "https://tfhub.dev/tensorflow/albert_en_base/2",
    "electra_small": "https://tfhub.dev/google/electra_small/2",
    "electra_base": "https://tfhub.dev/google/electra_base/2",
    "experts_pubmed": "https://tfhub.dev/google/experts/bert/pubmed/2",
    "experts_wiki_books": "https://tfhub.dev/google/experts/bert/wiki_books/2",
    "talking-heads_base": "https://tfhub.dev/tensorflow/talkheads_ggelu_bert_en_base/1",
}

PREPROCESSOR = {
    "bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "bert_en_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3",
    "small_bert/bert_en_uncased_L-2_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-2_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-2_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-2_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-4_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-4_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-4_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-4_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-6_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-6_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-6_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-6_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-8_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-8_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-8_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-8_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-10_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-10_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-10_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-10_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-12_H-128_A-2": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-12_H-256_A-4": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-12_H-512_A-8": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "small_bert/bert_en_uncased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "bert_multi_cased_L-12_H-768_A-12": "https://tfhub.dev/tensorflow/bert_multi_cased_preprocess/3",
    "albert_en_base": "https://tfhub.dev/tensorflow/albert_en_preprocess/2",
    "electra_small": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "electra_base": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "experts_pubmed": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "experts_wiki_books": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
    "talking-heads_base": "https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3",
}


class BERTEncoder(layers.Layer):
    """A keras layer to learn embeddings from raw text using BERT and finetune it

    Args:
        bert_model_name (string): bert model to be used, default: small_bert/bert_en_uncased_L-4_H-512_A-8

    All Models:
        experts_pubmed
        small_bert/bert_en_uncased_L-2_H-256_A-4
        small_bert/bert_en_uncased_L-2_H-512_A-8
        small_bert/bert_en_uncased_L-2_H-128_A-2
        talking-heads_base
        small_bert/bert_en_uncased_L-12_H-256_A-4
        small_bert/bert_en_uncased_L-6_H-768_A-12
        small_bert/bert_en_uncased_L-4_H-128_A-2
        small_bert/bert_en_uncased_L-2_H-768_A-12
        small_bert/bert_en_uncased_L-4_H-768_A-12
        bert_en_cased_L-12_H-768_A-12
        albert_en_base
        experts_wiki_books
        small_bert/bert_en_uncased_L-8_H-256_A-4
        small_bert/bert_en_uncased_L-12_H-128_A-2
        small_bert/bert_en_uncased_L-4_H-256_A-4
        small_bert/bert_en_uncased_L-10_H-768_A-12
        electra_small
        electra_base
        bert_multi_cased_L-12_H-768_A-12
        small_bert/bert_en_uncased_L-12_H-768_A-12
        small_bert/bert_en_uncased_L-6_H-256_A-4
        small_bert/bert_en_uncased_L-10_H-128_A-2
        bert_en_uncased_L-12_H-768_A-12
        small_bert/bert_en_uncased_L-8_H-512_A-8
        small_bert/bert_en_uncased_L-4_H-512_A-8
        small_bert/bert_en_uncased_L-6_H-128_A-2
        small_bert/bert_en_uncased_L-8_H-768_A-12
        small_bert/bert_en_uncased_L-10_H-256_A-4
        small_bert/bert_en_uncased_L-6_H-512_A-8
        small_bert/bert_en_uncased_L-12_H-512_A-8
        small_bert/bert_en_uncased_L-8_H-128_A-2
        small_bert/bert_en_uncased_L-10_H-512_A-8
    """

    def __init__(self, bert_model_name="small_bert/bert_en_uncased_L-4_H-512_A-8"):
        super().__init__()
        self.tfhub_handle_encoder = ENCODER[bert_model_name]
        self.tfhub_handle_preprocess = PREPROCESSOR[bert_model_name]

    def __call__(self, text_input):
        encoder_inputs = hub.KerasLayer(
            self.tfhub_handle_preprocess, name="preprocessing"
        )(text_input)
        outputs = hub.KerasLayer(
            self.tfhub_handle_encoder, trainable=True, name="BERT_encoder"
        )(encoder_inputs)
        return outputs["pooled_output"]
