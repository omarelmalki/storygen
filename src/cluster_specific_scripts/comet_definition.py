import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from utils.comet_utils import use_task_specific_params, trim_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i: i + n]


class Comet0:
    def __init__(self, model_path):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.model = torch.nn.DataParallel(AutoModelForSeq2SeqLM.from_pretrained(model_path)).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 256
        self.decoder_start_token_id = None

    def generate(
            self,
            queries,
            decode_method="beam",
            num_generate=5,
    ):
        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):
                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(
                    self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                _model = self.model.module if hasattr(self.model, 'module') else self.model
                if decode_method =="beam":
                    summaries = _model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_start_token_id=self.decoder_start_token_id,
                        num_beams=num_generate,
                        num_return_sequences=num_generate,
                    )
                elif decode_method=="top_k":
                    summaries = _model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        decoder_start_token_id=self.decoder_start_token_id,
                        do_sample=True,
                        top_k=50,
                        top_p=0.9,
                        num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                decs += dec
                # decs.append(dec)

            n = num_generate
            return [decs[i:i + n] for i in range(0, len(decs), n)]


all_relations = [
    "AtLocation",
    "CapableOf",
    "Causes",
    "CausesDesire",
    "CreatedBy",
    "DefinedAs",
    "DesireOf",
    "Desires",
    "HasA",
    "HasFirstSubevent",
    "HasLastSubevent",
    "HasPainCharacter",
    "HasPainIntensity",
    "HasPrerequisite",
    "HasProperty",
    "HasSubEvent",
    "HasSubevent",
    "HinderedBy",
    "InheritsFrom",
    "InstanceOf",
    "IsA",
    "LocatedNear",
    "LocationOfAction",
    "MadeOf",
    "MadeUpOf",
    "MotivatedByGoal",
    "NotCapableOf",
    "NotDesires",
    "NotHasA",
    "NotHasProperty",
    "NotIsA",
    "NotMadeOf",
    "ObjectUse",
    "PartOf",
    "ReceivesAction",
    "RelatedTo",
    "SymbolOf",
    "UsedFor",
    "isAfter",
    "isBefore",
    "isFilledBy",
    "oEffect",
    "oReact",
    "oWant",
    "xAttr",
    "xEffect",
    "xIntent",
    "xNeed",
    "xReact",
    "xReason",
    "xWant",
]


class Comet1:
    def __init__(self, model_path):
        self.device = "cuda:1" if torch.cuda.is_available() else "cpu"
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model = torch.nn.DataParallel(model)
        self.model = model.to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 256
        self.decoder_start_token_id = None

    def generate(
            self,
            queries,
            decode_method="beam",
            num_generate=5,
    ):
        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):
                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(
                    self.device)
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True,
                                                  clean_up_tokenization_spaces=False)
                decs += dec
                # decs.append(dec)

            n = num_generate
            return [decs[i:i + n] for i in range(0, len(decs), n)]
