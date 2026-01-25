import datasets
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.retrieval import AbsTaskRetrieval
import json

def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)

class ColieeTask1(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    def load_data(self, **kwargs):
        if self.data_loaded:
            return
        queries = load_json("datasets/coliee_task1/task1_test_queries_2025.json")
        corpus = load_json("datasets/coliee_task1/task1_test_corpus_2025.json")
        qrels = {}
        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}
        self.data_loaded = True
