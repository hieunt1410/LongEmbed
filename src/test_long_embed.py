import os
import json
import logging

# Monkey-patch MTEB RetrievalEvaluator BEFORE importing MTEB
# This adds k=50 to all metrics
import mteb.evaluation.evaluators

# _original_re_init = mteb.evaluation.evaluators.RetrievalEvaluator.__init__


# def _patched_re_init(
#     self, retriever=None, k_values=[1, 3, 5, 10, 20, 50, 100, 1000], **kwargs
# ):
#     _original_re_init(self, retriever=retriever, k_values=k_values, **kwargs)


# mteb.evaluation.evaluators.RetrievalEvaluator.__init__ = _patched_re_init

from mteb import MTEB

from utils import logger, get_args
from encoder_model import RetrievalModel

logging.getLogger().setLevel(logging.INFO)


def main():
    args = get_args()
    model = RetrievalModel(args)

    model_name = os.path.basename(os.path.normpath(args.model_name_or_path))
    mteb_output_dir = os.path.join(args.output_dir, model_name)

    chunking_mode: str = os.getenv("CHUNKING_MODE")
    if chunking_mode != "no_chunk":
        chunk_max_len = os.getenv("MAX_TOKEN_NUM", "0")
        mteb_output_dir += f"_{chunking_mode}-{chunk_max_len}"
    if args.pos_mode != "original":
        mteb_output_dir += f"_{args.pos_mode}"
    if args.use_self_extend == True:
        mteb_output_dir += f"_se_{model.encode_max_length}"
    if args.rope_theta != 10000:
        mteb_output_dir += f"_theta{args.rope_theta}_{model.encode_max_length}"
    if args.rotary_scaling_factor != None:
        mteb_output_dir += f"_rsf{args.rotary_scaling_factor}"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(mteb_output_dir, exist_ok=True)

    retrieval_task_list = []
    needle_passkey_task_list = []
    output_dict = dict()
    needle_passkey_score_list = list()

    for task in [
        "LEMBSummScreenFDRetrieval",
        "LEMBQMSumRetrieval",
        "LEMBWikimQARetrieval",
        "LEMBNarrativeQARetrieval",
    ]:
        if task in args.task_list:
            retrieval_task_list.append(task)

    for task in ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]:
        if task in args.task_list:
            needle_passkey_task_list.append(task)

    # evaluating needle and passkey retrieval tasks
    if needle_passkey_task_list != []:
        context_length_list = list(args.window_length_list)
        context_length_list.sort()

        evaluation = MTEB(tasks=needle_passkey_task_list)
        results = evaluation.run(
            model,
            output_folder=mteb_output_dir,
            overwrite_results=False,
            batch_size=args.batch_size,
            verbosity=0,
            save_predictions=True,
        )
        for key, value in results.items():
            needle_passkey_score_list = []
            for ctx_len in context_length_list:
                needle_passkey_score_list.append(
                    [ctx_len, value[f"test_{ctx_len}"]["ndcg_at_1"]]
                )
            needle_passkey_score_list.append(
                [
                    "avg",
                    sum([x[1] for x in needle_passkey_score_list])
                    / len(context_length_list),
                ]
            )
            output_dict[key] = {item[0]: item[1] for item in needle_passkey_score_list}

    # evaluating retrieval tasks
    if retrieval_task_list != []:
        results = mteb.evaluate(
            model,
            tasks=retrieval_task_list,
            overwrite_results="only-missing",
            prediction_folder=mteb_output_dir,
            encode_kwargs={"batch_size": args.batch_size}
        )

        for key, value in results.items():
            split = "test" if "test" in value else "validation"
            output_dict[key] = {
                "ndcg@1": value[split]["ndcg_at_1"],
                "ndcg@10": value[split]["ndcg_at_10"],
            }

    logger.info(output_dict)

    if len(args.task_list) == 6:
        with open(os.path.join(mteb_output_dir, "overall_results.json"), "w") as f:
            json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    main()
