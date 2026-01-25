import os
import json
import logging

import mteb

from utils import logger, get_args
from encoder_model import RetrievalModel

# Import custom LEMB tasks
from LEMBNeedleRetrieval import LEMBNeedleRetrieval
from LEMBPasskeyRetrieval import LEMBPasskeyRetrieval
from LEMBNarrativeQARetrieval import LEMBNarrativeQARetrieval
from LEMBQMSumRetrieval import LEMBQMSumRetrieval
from LEMBSummScreenFDRetrieval import LEMBSummScreenFDRetrieval
from LEMBWikimQARetrieval import LEMBWikimQARetrieval

# Map task names to task classes for MTEB v2
CUSTOM_TASKS = {
    "LEMBNeedleRetrieval": LEMBNeedleRetrieval,
    "LEMBPasskeyRetrieval": LEMBPasskeyRetrieval,
    "LEMBNarrativeQARetrieval": LEMBNarrativeQARetrieval,
    "LEMBQMSumRetrieval": LEMBQMSumRetrieval,
    "LEMBSummScreenFDRetrieval": LEMBSummScreenFDRetrieval,
    "LEMBWikimQARetrieval": LEMBWikimQARetrieval,
}

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
    if args.use_self_extend:
        mteb_output_dir += f"_se_{model.encode_max_length}"
    if args.rope_theta != 10000:
        mteb_output_dir += f"_theta{args.rope_theta}_{model.encode_max_length}"
    if args.rotary_scaling_factor:
        mteb_output_dir += f"_rsf{args.rotary_scaling_factor}"

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(mteb_output_dir, exist_ok=True)

    retrieval_task_list = []
    needle_passkey_task_list = []
    output_dict = dict()

    for task in [
        "LEMBSummScreenFDRetrieval",
        "LEMBQMSumRetrieval",
        "LEMBWikimQARetrieval",
        "LEMBNarrativeQARetrieval",
        "coliee_task1"
    ]:
        if task in args.task_list:
            retrieval_task_list.append(task)

    for task in ["LEMBNeedleRetrieval", "LEMBPasskeyRetrieval"]:
        if task in args.task_list:
            needle_passkey_task_list.append(task)

    # evaluating needle and passkey retrieval tasks
    if needle_passkey_task_list:
        # Use the model's encode_max_length as the context length
        context_length = model.encode_max_length

        # Get task instances for needle and passkey tasks with context_length
        tasks = [CUSTOM_TASKS[task_name](context_length=context_length) for task_name in needle_passkey_task_list]
        results = mteb.evaluate(
            model,
            tasks,
            prediction_folder=mteb_output_dir,
            overwrite_strategy="only-missing",
            encode_kwargs={"batch_size": args.batch_size},
        )
        # MTEB v2 returns ModelResult object with task_results list
        for task_result in results.task_results:
            split = "test"
            if split in task_result.scores and task_result.scores[split]:
                scores = task_result.scores[split][0]
                output_dict[task_result.task_name] = {
                    "ndcg@1": scores.get("ndcg_at_1"),
                    "ndcg@10": scores.get("ndcg_at_10"),
                }

    # evaluating retrieval tasks
    if retrieval_task_list:
        # Get task instances for retrieval tasks
        tasks = [CUSTOM_TASKS[task_name]() for task_name in retrieval_task_list]
        results = mteb.evaluate(
            model,
            tasks,
            prediction_folder=mteb_output_dir,
            overwrite_strategy="only-missing",
            encode_kwargs={"batch_size": args.batch_size},
        )

        # MTEB v2 returns ModelResult object with task_results list
        for task_result in results.task_results:
            split = "test" if "test" in task_result.scores else "validation"
            if split in task_result.scores and task_result.scores[split]:
                scores = task_result.scores[split][0]
                output_dict[task_result.task_name] = {
                    "ndcg@1": scores.get("ndcg_at_1"),
                    "ndcg@10": scores.get("ndcg_at_10"),
                }

    logger.info(output_dict)

    if len(args.task_list) == 6:
        with open(os.path.join(mteb_output_dir, "overall_results.json"), "w") as f:
            json.dump(output_dict, f, indent=4)


if __name__ == "__main__":
    main()
