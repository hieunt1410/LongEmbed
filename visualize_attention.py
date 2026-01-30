"""Visualize attention heatmaps for the custom E5-Mistral model under the TP plan."""

import sys
sys.path.insert(0, "./Attention-Viewer")

import torch
from transformers import AutoTokenizer
from attn_viewer.plot_utils import plot_heatmap
from src.custom_models.custom_e5_mistral import MistralModel, MistralConfig


def main():
    model_path = "intfloat/e5-mistral-7b-instruct"

    # Which layers to visualize (set to None for all layers — but will use lots of memory)
    LAYERS_TO_VISUALIZE = [0, 1, 3, 6, 15, 31]  # adjust as needed

    # Load tokenizer and add <PST> token (becomes id 32000)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.add_tokens(["<PST>"])

    # Load custom MistralModel with eager attention (required to return attention weights)
    config = MistralConfig.from_pretrained(model_path)
    config._attn_implementation = "eager"
    model = MistralModel.from_pretrained(
        model_path,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.resize_token_embeddings(len(tokenizer))
    model.output_attentions_layers = LAYERS_TO_VISUALIZE
    model.eval()

    # Prompt must contain <PST> (token 32000) — the TP plan requires it
    prompt = "Instruct: Retrieve relevant passages\nQuery: <PST> What is attention in transformers?"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs["input_ids"]

    # Sanity check
    ids = input_ids[0].tolist()
    assert 32000 in ids, f"Token 32000 (<PST>) not found: {ids}"
    assert 1 in ids, f"Token 1 (BOS) not found: {ids}"

    tokens_list = [t.replace("▁", "") for t in tokenizer.convert_ids_to_tokens(ids)]
    print(f"Tokens ({len(tokens_list)}): {tokens_list}")
    print(f"Visualizing layers: {LAYERS_TO_VISUALIZE}")

    # Forward pass — only selected layers output attention weights
    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=inputs.get("attention_mask"),
            output_attentions=True,
        )

    attention_scores = [attn.cpu() for attn in outputs.attentions]
    print(f"Got attention from {len(attention_scores)} layers")

    plot_heatmap(
        attention_scores,
        model_id="e5-mistral-tp",
        plot_figs_per_head=True,
        save_fig_path="./attention_figs",
        tokens_list=tokens_list,
        ignore_first_token=True,
        num_figs_per_row=8,
    )
    print("Done. Heatmaps saved to ./attention_figs/")


if __name__ == "__main__":
    main()
