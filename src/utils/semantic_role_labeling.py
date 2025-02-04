from allennlp.common import JsonDict
# Semantic Role Labeling with BERT : https://github.com/Riccorl/transformer-srl
from transformer_srl import dataset_readers, models, predictors


def sentence_to_srl(sentence: str) -> JsonDict:
    """
    Extracts Semantic Roles from a sentence.

    :param sentence: sentence from which to extract semantic roles labels.
    :return: semantic_roles as PropBank English SRLs.
    """

    # Pre-trained model with BERT fine-tuned to predict PropBank SRLs on CoNLL 2012 dataset.
    predictor = predictors.SrlTransformersPredictor.from_path(
        "../data/pre-trained-transformer-srl/srl_bert_base_conll2012.tar.gz", "transformer_srl")

    # More documentation: https://docs.allennlp.org/models/main/models/structured_prediction/predictors/srl/
    semantic_roles: JsonDict = predictor.predict(sentence)
    return semantic_roles


def get_predictor():
    """
    Return SRL transformer predictor
    :return:
    """
    return predictors.SrlTransformersPredictor.from_path(
        "../data/pre-trained-transformer-srl/srl_bert_base_conll2012.tar.gz", "transformer_srl")


def run_batch_predictor(batch_data, predictor):
    """
    Run SRL extraction from a batch of sentences
    :param batch_data:
    :param predictor:
    :return:
    """
    if len(batch_data) == 1:
        result = predictor.predict_json(batch_data[0])
        results = [result]
    else:
        results = predictor.predict_batch_json(batch_data)
    return results

