import datasets
from mteb.abstasks.task_metadata import TaskMetadata
from mteb.abstasks.retrieval import AbsTaskRetrieval

class LEMBNeedleRetrieval(AbsTaskRetrieval):
    _EVAL_SPLIT = "test"

    metadata = TaskMetadata(
        name="LEMBNeedleRetrieval",
        dataset={
            "path": "dwzhu/LongEmbed",
            "revision": "6e346642246bfb4928c560ee08640dc84d074e8c",
            "name": "needle",
        },
        reference="https://huggingface.co/datasets/dwzhu/LongEmbed",
        description=("needle subset of dwzhu/LongEmbed dataset."),
        type="Retrieval",
        category="t2t",
        eval_splits=[_EVAL_SPLIT],
        eval_langs=["eng-Latn"],
        main_score="ndcg_at_10",
        date=("2000-01-01", "2023-12-31"),
        domains=["Academic", "Blog"],
        task_subtypes=["Article retrieval"],
        license="not specified",
        annotations_creators="derived",
        dialect=[],
        sample_creation="found",
        bibtex_citation=None,
    )

    def __init__(self, context_length=None, **kwargs):
        super().__init__(**kwargs)
        self._context_length = context_length

    def load_data(self, num_proc=1, **kwargs):
        if self.data_loaded:
            return

        # Use stored context_length or get from kwargs if provided
        context_length = kwargs.get("context_length", self._context_length)
        if context_length is None:
            raise ValueError("Need to specify context_length")

        query_list = datasets.load_dataset(**self.metadata.dataset)[
            "queries"
        ]
        query_list = query_list.filter(lambda x: x["context_length"] == context_length)
        queries = {row["qid"]: row["text"] for row in query_list}

        corpus_list = datasets.load_dataset(**self.metadata.dataset)[
            "corpus"
        ]
        corpus_list = corpus_list.filter(
            lambda x: x["context_length"] == context_length
        )
        corpus = {row["doc_id"]: {"text": row["text"]} for row in corpus_list}

        qrels_list = datasets.load_dataset(**self.metadata.dataset)[
            "qrels"
        ]
        qrels_list = qrels_list.filter(lambda x: x["context_length"] == context_length)
        qrels = {row["qid"]: {row["doc_id"]: 1} for row in qrels_list}

        self.corpus = {self._EVAL_SPLIT: corpus}
        self.queries = {self._EVAL_SPLIT: queries}
        self.relevant_docs = {self._EVAL_SPLIT: qrels}
        self.data_loaded = True
