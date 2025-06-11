import argparse
import os
import time
import torch
import wandb
from main_space.model import MyTransformerLM
from tokenizers import Tokenizer


def parse_args():

    projectName = "self-distill-research"
    entity = "luka_newbie"
    inputText = "Default input text "

    alias_tag = "100M_teacher:run_9ttfmtet_step_49994"
    artifactName = alias_tag.split(':')[0]
    alias = alias_tag.split(':')[1]

    parser = argparse.ArgumentParser(description="Run inference on a W&B checkpointed model.")
    parser.add_argument("--entity", type=str, default=entity, help="W&B entity (user or org)")
    parser.add_argument("--project", type=str, default=projectName, help="W&B project name")
    parser.add_argument("--artifact-name", type=str, default=artifactName, help="Base artifact name")
    parser.add_argument("--alias", type=str, default=alias, help="Artifact alias to load (default: latest)")
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., cpu or cuda). Auto-detect if not set.")
    parser.add_argument("--input-text", type=str, default=inputText, help="Text to run inference on")
    return parser.parse_args()

@torch.no_grad()
def generate_greedy(
    model,
    tokenizer,
    prompt: str,
    device: torch.device,
    max_new_tokens: int = 50,
    print_each: bool = True,
) -> str:
    """
    Autoregressively generates text from `prompt` using greedy decoding.

    Args:
        model: your MyTransformerLM, already in eval() mode on `device`.
        tokenizer: HF or tokenizers tokenizer with .encode(...).ids and .decode(...) methods.
        prompt: the initial text to condition on.
        device: torch.device("cpu") or torch.device("cuda").
        max_new_tokens: maximum number of tokens to append.
        print_each: if True, prints each new token as it’s generated.

    Returns:
        The full generated string: prompt + generated continuation.
    """
    # 1) Encode prompt
    encoding = tokenizer.encode(prompt)
    token_ids = encoding.ids  # list[int]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    generated = token_ids.copy()

    print(f"[INFO] Starting generation: prompt length = {len(token_ids)} tokens")
    start_time = time.time()

    for step in range(max_new_tokens):
        # a) Forward pass
        logits = model(input_ids)               # [1, seq_len, vocab_size]
        next_logits = logits[0, -1, :]          # [vocab_size]

        # b) Greedy pick
        next_id = int(torch.argmax(next_logits, dim=-1))
        generated.append(next_id)

        # c) Optionally print the decoded token
        if print_each:
            next_token = tokenizer.decode([next_id])
            print(f"[GENERATED {step+1:03d}] {next_id}: \"{next_token}\"")

        # d) Prepare for next step
        input_ids = torch.tensor([generated], dtype=torch.long, device=device)

    elapsed = time.time() - start_time
    print(f"[INFO] Generation of {max_new_tokens} tokens took {elapsed:.3f}s")

    # Decode full sequence
    full_text = tokenizer.decode(generated)
    return full_text

def main():
    args = parse_args()

    tokenizer_path =  os.path.join(os.getcwd(), "dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")

    # Set up device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")

    # Initialize W&B API
    api = wandb.Api()
    artifact_path = f"{args.entity}/{args.project}/{args.artifact_name}:{args.alias}"
    print(f"[INFO] Loading artifact '{artifact_path}'")
    artifact = api.artifact(artifact_path, type="model-checkpoint")
    download_dir = artifact.download()
    print(f"[INFO] Artifact downloaded to: {download_dir}")

    # 1) Grab the run that produced this artifact:
    creator = artifact.logged_by()  # a <wandb.apis.public.Run> stub
    run_ref = f"{creator.entity}/{creator.project}/{creator.id}"
    print(f"[INFO] Loading run that produced artifact: {run_ref}")
    run = api.run(run_ref)

    # 2) Extract and print the run.config
    config = run.config or {}
    print("[INFO] Loaded run.config:")
    for k, v in sorted(config.items()):
        print(f"  - {k}: {v!r}")

    # Read metadata for model hyperparameters
    metadata = artifact.metadata or {}
    print("[INFO] Loaded artifact metadata:")
    for k,v in metadata.items():
        print(f"  - {k}: {v}")

    # Extract required hyperparameters (with defaults or errors)
    try:
        vocab_size = int(config["vocab_size"])
        d_model     = int(config["d_model"])
        n_heads     = int(config["num_heads"])
        n_layers    = int(config["num_layers"])
        ctx_size    = int(config["ctx_len"])
        p_dropout   = float(config["dropout_rate"])
    except KeyError as e:
        print(f"[ERROR] Missing hyperparameter in metadata: {e}")
        return

    # Instantiate model
    print("[INFO] Instantiating model with loaded hyperparameters...")
    model = MyTransformerLM(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        ctx_size=ctx_size,
        p_dropout=p_dropout,
    )
    model.to(device)
    model.eval()
    print("[INFO] Model architecture:")
    print(model)

    # Load checkpoint
    checkpoint_file = os.path.join(download_dir, "checkpoint.pt")
    if not os.path.exists(checkpoint_file):
        print(f"[ERROR] checkpoint.pt not found in {download_dir}")
        return

    print(f"[INFO] Loading state dict from {checkpoint_file}")
    checkpoint = torch.load(checkpoint_file, map_location=device)
    result = model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    print(f"[INFO] Missing keys: {result.missing_keys}")
    print(f"[INFO] Unexpected keys: {result.unexpected_keys}")
    print("[INFO] Model weights loaded successfully.")
    print(f"[INFO] Tokenizing input: \"{args.input_text}\"")


    tokenizer = Tokenizer.from_file(tokenizer_path)

    # ... after loading model, tokenizer, and moving model to device ...

    prompt = args.input_text
    generated = generate_greedy(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        device=device,
        max_new_tokens=100,    # or whatever you like
        print_each=True,
    )

    print("[RESULT] Full generated text:")
    print(generated)


if __name__ == "__main__":
    main()
