import argparse
import os
import re
import time
import torch
import wandb
from main_space.model import MyTransformerLM
from tokenizers import Tokenizer
import torch.nn.functional as F

inputText = """ = Croatia = """

skip_loading = False


def parse_args():
    projectName = "self-distill-research"
    entity = "luka_newbie"

    alias_tag = "distilled_model_dummies:run_0hctfkyc_step_9999"
    artifactName = alias_tag.split(':')[0]
    alias = alias_tag.split(':')[1]

    parser = argparse.ArgumentParser(description="Run inference on a W&B checkpointed model.")
    parser.add_argument("--entity", type=str, default=entity, help="W&B entity (user or org)")
    parser.add_argument("--project", type=str, default=projectName, help="W&B project name")
    parser.add_argument("--artifact-name", type=str, default=artifactName, help="Base artifact name")
    parser.add_argument("--alias", type=str, default=alias, help="Artifact alias to load (default: latest)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to run on (e.g., cpu or cuda). Auto-detect if not set.")
    parser.add_argument("--input-text", type=str, default=inputText, help="Text to run inference on")
    return parser.parse_args()


def clean_decoded_text(text: str) -> str:
    text = re.sub(r"\s*@-@\s*", "-", text)
    text = re.sub(r"\s*@,@\s*", ",", text)  # "2 @,@ 500" → "2,500"
    text = re.sub(r"\s*@\.@\s*", ".", text)
    return text.strip()


@torch.no_grad()
def generate_greedy(
        model,
        tokenizer,
        prompt: str,
        device: torch.device,
        ctx_size: int,
        max_new_tokens: int = 256,
        print_each: bool = False,
        never_stop: bool = True,  # doesn't stop generation on EOS token
        temperature: float = 0.8,
        repetition_penalty: float = 2.0,
        top_k: int = 10,
        top_p: float = 0.90  # Vrijednosti blizu 1.0 su manje restriktivne, a blizu 0 su više.
) -> str:

    if not ( 0.1 < temperature < 5.0):
        raise ValueError("Temperature should be between 0.1 and 5.0")

    if not (0.01 < top_p < 1.0):
        raise ValueError("Top P should be between 0.01 and 1.0")


    token_ids = [2, 3] + tokenizer.encode(prompt).ids # Add [CLS][SEP], which represent eos and bos
    org_tokens_len = len(token_ids)
    generated_ids = token_ids.copy()

    print(f"[INFO] Starting generation: prompt_len={len(token_ids)}, ctx_size={ctx_size}")
    start_time = time.time()

    for step in range(max_new_tokens):
        input_for_model = torch.tensor([generated_ids[-ctx_size:]], device=device)
        logits = model(input_for_model)  # Calling the regular forward pass

        next_logits = logits[0, -1, :]

        if step > 0:
            for token_id in set(generated_ids):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty


        next_logits = next_logits / temperature

        sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        next_logits[indices_to_remove] = float('-inf')

        all_probs = F.softmax(next_logits, dim=-1)

        # --- 2. PERFORM TOP-K SAMPLING ---
        top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)
        top_k_relative_probs = F.softmax(top_k_logits, dim=-1)
        sampled_relative_index = torch.multinomial(top_k_relative_probs, num_samples=1)
        next_id = top_k_indices[sampled_relative_index.item()].item()
        generated_ids.append(next_id)
        chosen_token_abs_prob = all_probs[next_id].item()
        top_k_abs_probs = all_probs[top_k_indices]

        if print_each:
            print_verbose(chosen_token_abs_prob,
                          generated_ids,
                          next_id,
                          org_tokens_len,
                          tokenizer,
                          top_k,
                          top_k_abs_probs,
                          top_k_indices,
                          top_k_relative_probs)
        else:
            print(tokenizer.decode([next_id], skip_special_tokens=False), end="", flush=True)


        if next_id == 2:  # Check for an end-of-sequence token
            print("<EOT>", end="")
            if not never_stop:
                break

    print("DONE WITH STEPS")
    elapsed = time.time() - start_time
    tps = (step + 1) / elapsed
    print(f"\n[INFO] Generation of {step + 1} new tokens took {elapsed:.3f}s ({tps:.2f} tokens/sec)")

    full_text = clean_decoded_text(tokenizer.decode(generated_ids, skip_special_tokens=False))
    return full_text


def print_verbose(chosen_token_abs_prob, generated_ids, next_id, org_tokens_len, tokenizer, top_k, top_k_abs_probs,
                  top_k_indices, top_k_relative_probs):
    a = f"Chosen: '{tokenizer.decode([next_id])}' (Absolute Probability: {chosen_token_abs_prob:.2%})"
    b = ["" for _ in range(top_k)]
    for i in range(top_k):
        token_id = top_k_indices[i].item()
        token_str = tokenizer.decode([token_id])
        abs_prob = top_k_abs_probs[i].item()
        rel_prob = top_k_relative_probs[i].item()
        b[i] = f"  - Top {i + 1}: '{token_str}' (Abs Prob: {abs_prob:.2%}, Rel Prob: {rel_prob:.2%})"
    print("\n--- Token Generation Step ---")
    print(f"Input [{clean_decoded_text(tokenizer.decode(generated_ids[org_tokens_len:]))}]")
    print(a)
    print(f"Top {top_k} Candidates:")
    for line in b:
        print(line)
    print("---------------------------\n")


def main():
    args = parse_args()

    tokenizer_path = os.path.join(os.getcwd(), "dataset_creation/tokenizer/1_raw_wikitext103_bpe_vocab_5000.json")

    # Set up device
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"[INFO] Using device: {device}")
    if not skip_loading:
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
        for k, v in metadata.items():
            print(f"  - {k}: {v}")

        # Extract required hyperparameters (with defaults or errors)
        try:
            vocab_size = int(config["vocab_size"])
            d_model = int(config["d_model"])
            n_heads = int(config["num_heads"])
            n_layers = int(config["num_layers"])
            ctx_size = int(config["ctx_len"])
            p_dropout = float(config["dropout_rate"])
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

    if skip_loading:
        # Instantiate model

        ctx_size = 512
        print("[INFO] Instantiating model with loaded hyperparameters...")
        model = MyTransformerLM(
            vocab_size=5000,
            d_model=512,
            n_heads=4,
            n_layers=2,
            ctx_size=ctx_size,
            p_dropout=0.1,
        )
        model.to(device)
        model.eval()
        print("[INFO] Model architecture:")
        print(model)

    tokenizer = Tokenizer.from_file(tokenizer_path)

    # ... after loading model, tokenizer, and moving model to device ...

    if not skip_loading:
        prompt = args.input_text
        generated = generate_greedy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            ctx_size=ctx_size,
            device=device,
        )

    if skip_loading:
        prompt = args.input_text
        generated = generate_greedy(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            ctx_size=ctx_size,
            device=device,
        )

    print("[RESULT] Full generated text:")
    print(generated)

    print()
    print("-"*60)
    print(f"End of script...")
    print("-"*60)


if __name__ == "__main__":
    main()
