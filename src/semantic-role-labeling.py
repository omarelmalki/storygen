from allennlp.common import JsonDict
from transformer_srl import dataset_readers, models, predictors

s = 'Jenny lived in Florida.'


def get_srl(sentence: str) -> JsonDict:
    """
    Extracts Semantic Roles from a sentence.

    :param sentence: sentence from which to extract semantic roles labels.
    :return: semantic_roles as PropBank English SRLs.
    """
    predictor = predictors.SrlTransformersPredictor.from_path(
        "../data/pre-trained-transformer-srl/srl_bert_base_conll2012.tar.gz", "transformer_srl")
    semantic_roles: JsonDict = predictor.predict(sentence)
    return semantic_roles

