import argparse
import os
import time
import torch
import wandb
from main_space.model import MyTransformerLM
from tokenizers import Tokenizer
import torch.nn.functional as F



inputText = """
user: What is 2 + 2?
agent: 2 + 2 = 4
user: What is 5 + 5?
agent: 5 + 5 = 10
user: What is 3 + 8?
agent: 3 + 8 = 11
user: What is 6 + 2?
agent: 6 + 2 = 8
user: What is 3 + 2?
agent: 
"""

skip_loading = False


def parse_args():
    projectName = "self-distill-research"
    entity = "luka_newbie"

    alias_tag = "100M_teacher:run_9ttfmtet_step_49994"
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


@torch.no_grad()
def generate_greedy(
        model,
        tokenizer,
        prompt: str,
        device: torch.device,
        ctx_size: int,
        max_new_tokens: int = 1000,
        print_each: bool = True,
) -> str:
    disable_cache = True

    """
    Generates text using a KV cache with a nested structure and a sliding window.
    This function is specifically tailored for the provided model's .inference methods.
    """
    # 1. Encode prompt and prepare initial state
    token_ids = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    org_tokens_len = len(token_ids)
    generated_ids = token_ids.copy()

    print(f"[INFO] Starting generation: prompt_len={len(token_ids)}, ctx_size={ctx_size}")
    start_time = time.time()

    # 2. Initialize the cache. It starts as None.
    kv_cache = None

    # 3. The main generation loop
    for step in range(max_new_tokens):

        if not disable_cache:
            # --- A. SLIDING WINDOW LOGIC FOR THE KV CACHE ---
            if kv_cache is not None:
                # First, determine the current length of the cache.
                # We can reliably check the first key tensor of the first head of the first layer.
                # kv_cache -> [layer0_cache, layer1_cache, ...]
                # layer0_cache -> [head0_cache, head1_cache, ...]
                # head0_cache -> (k_tensor, v_tensor)
                # k_tensor -> shape [B, seq_len, head_size]
                first_key_tensor = kv_cache[0][0][0]
                cached_seq_len = first_key_tensor.size(1)

                # If the cache is at or over the context limit, we must slide it.
                if cached_seq_len >= ctx_size:
                    # To "slide", we rebuild the entire cache structure, but with trimmed tensors.
                    new_kv_cache_after_sliding = []
                    for layer_cache in kv_cache:  # This is a list of head_caches
                        new_layer_cache = []
                        for head_k, head_v in layer_cache:  # This is a (k, v) tensor tuple
                            # Trim the oldest token (at index 0) from the sequence dimension (dim 1)
                            k_trimmed = head_k[:, 1:, :]
                            v_trimmed = head_v[:, 1:, :]
                            new_layer_cache.append((k_trimmed, v_trimmed))
                        new_kv_cache_after_sliding.append(new_layer_cache)
                    kv_cache = new_kv_cache_after_sliding

            # --- B. MODEL FORWARD PASS ---
            # On the first step, input_ids is the prompt and kv_cache is None.
            # On subsequent steps, input_ids is just the newest token, and we pass the (potentially slid) cache.
            logits, kv_cache = model.inference_step(input_ids, kv_cache)
            if disable_cache:
                kv_cache = None

        else:
            input_for_model = torch.tensor([generated_ids[-ctx_size:]], device=device)
            logits = model(input_for_model)  # Calling the regular forward pass

        # --- C. SAMPLE NEXT TOKEN ---

        # Change start
        next_logits = logits[0, -1, :]

        # --- 1. CALCULATE PROBABILITIES ---
        # Calculate absolute probabilities over the entire vocabulary by applying softmax
        all_probs = F.softmax(next_logits, dim=-1)

        # --- 2. PERFORM TOP-K SAMPLING ---
        top_k = 10
        # Get the top k tokens, their scores (logits), and their indices (IDs)
        top_k_logits, top_k_indices = torch.topk(next_logits, top_k, dim=-1)

        # Calculate the RELATIVE probabilities for just the top k tokens
        top_k_relative_probs = F.softmax(top_k_logits, dim=-1)

        # Sample from the top k based on their relative probabilities
        sampled_relative_index = torch.multinomial(top_k_relative_probs, num_samples=1)

        # Get the actual token ID that was chosen from the sampling
        next_id = top_k_indices[sampled_relative_index.item()].item()
        generated_ids.append(next_id)

        # --- 3. GATHER DATA FOR PRINTING ---
        # Get the absolute probability of the single token that was chosen
        chosen_token_abs_prob = all_probs[next_id].item()

        # Get the absolute probabilities for each of the top_k tokens
        top_k_abs_probs = all_probs[top_k_indices]


        # --- 4. FORMAT AND PRINT OUTPUT ---
        # Format the string for the final chosen token
        a = f"Chosen: '{tokenizer.decode([next_id])}' (Absolute Probability: {chosen_token_abs_prob:.2%})"

        # Prepare the list of top candidates with their detailed probabilities
        b = ["" for _ in range(top_k)]
        for i in range(top_k):
            token_id = top_k_indices[i].item()
            token_str = tokenizer.decode([token_id])
            abs_prob = top_k_abs_probs[i].item()
            rel_prob = top_k_relative_probs[i].item()
            b[i] = f"  - Top {i + 1}: '{token_str}' (Abs Prob: {abs_prob:.2%}, Rel Prob: {rel_prob:.2%})"

        # Print the formatted results
        print("\n--- Token Generation Step ---")
        print(f"Input [{tokenizer.decode(generated_ids[org_tokens_len:])}]")
        print(a)
        print("Top 3 Candidates:")
        for line in b:
            print(line)
        print("---------------------------\n")

        continue
        # change end

        # --- E. PRINT AND CHECK FOR STOP TOKEN ---
        if print_each:
            print(tokenizer.decode([next_id]), end="", flush=True)

        # You should replace "2" with your actual EOS token ID if you have one
        if next_id == 2:  # Example: Check for an end-of-sequence token
            print("<EOT>", end="", flush=True)
        # print("\n[INFO] End-of-sequence token generated.")
        # break

    # 4. Final cleanup and return
    print()  # Final newline
    elapsed = time.time() - start_time
    tps = (step + 1) / elapsed if elapsed > 0 else float('inf')
    print(f"\n[INFO] Generation of {step + 1} new tokens took {elapsed:.3f}s ({tps:.2f} tokens/sec)")

    full_text = tokenizer.decode(generated_ids)
    return full_text


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


if __name__ == "__main__":
    main()
