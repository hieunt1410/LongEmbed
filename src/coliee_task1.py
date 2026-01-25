import json
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.retrieval import AbsTaskRetrieval


def load_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


class ColieeTask1(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"
    metadata = TaskMetadata(
        name="coliee_task1",
        dataset={
            "path": "local",
            "revision": "1.0.0",
        },
        reference="https://sites.ualberta.ca/~rabelo/COLIEE2025/",
        description="COLIEE Task 1: Legal Case Retrieval - Given a query case, retrieve relevant cases from the corpus.",
        type="Retrieval",
        category="t2t",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2020-01-01", "2025-12-31"),
        domains=["Legal"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="expert-annotated",
        dialect=[],
        sample_creation="found",
        bibtex_citation="",
    )

    def load_data(self, **kwargs):
        if self.data_loaded:
            return

        query_list = load_json("datasets/coliee_task1/task1_test_queries_2025.json")
        corpus_list = load_json("datasets/coliee_task1/task1_test_corpus_2025.json")

        # Convert to MTEB v1 dict format (will be auto-converted to v2)
        # queries: {qid: text}, corpus: {doc_id: {"_id": doc_id, "text": text}}, qrels: {qid: {doc_id: score}}
        queries = {row["qid"]: row["query"] for row in query_list}
        corpus = {row["doc_id"]: {"_id": row["doc_id"], "text": row["text"]} for row in corpus_list}

        # Create dummy qrels for test set (needed for MTEB to process queries)
        # Each query gets a dummy relevance to the first corpus doc
        first_doc_id = corpus_list[0]["doc_id"] if corpus_list else None
        qrels = {row["qid"]: {first_doc_id: 1} for row in query_list} if first_doc_id else {}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}
        self.data_loaded = True
