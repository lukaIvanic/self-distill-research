import argparse
import os
import time
import torch
import wandb
from main_space.model import MyTransformerLM
from tokenizers import Tokenizer
import torch.nn.functional as F
import re






inputText = """
<|im_start|>system
You are a helpful AI assistant.<|im_end|>
<|im_start|>user
Write a short story about a robot who discovers music.<|im_end|>
<|im_start|>assistant
"""

skip_loading = False

def clean_decoded_text(text: str) -> str:
    # Zamijeni "@-@" s "-" i očisti višestruke razmake
    text = re.sub(r"\s*@-@\s*", "-", text)
    text = re.sub(r"\s*@,@\s*", ",", text)       # "2 @,@ 500" → "2,500"
    text = re.sub(r"\s*@\.@\s*", ".", text)
    return text.strip()


def parse_args():
    projectName = "self-distill-research"
    entity = "luka_newbie"

    alias_tag = "ctx_len_1M_exp:latest"
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
        max_new_tokens: int = 100,
        print_each: bool = True,
        temperature: float = 0.4,
        repetition_penalty: float = 1.30,
        top_k: int = 50,
        top_p: float = 0.8  # Vrijednosti blizu 1.0 su manje restriktivne, a blizu 0 su više.
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

        # ### NOVI KOD ZA SMANJENJE PONAVLJANJA - START ###
        # Primijeni kaznu za ponavljanje (repetition penalty) na logite.
        # Ovo smanjuje vjerojatnost tokena koji su se već pojavili u generiranom tekstu.

        if step > 0:
            for token_id in set(generated_ids):
                if next_logits[token_id] > 0:
                    next_logits[token_id] /= repetition_penalty
                else:
                    next_logits[token_id] *= repetition_penalty

            # Primijeni temperaturu. Niže vrijednosti čine tekst fokusiranijim, a više kreativnijim.
        if temperature > 0:
            next_logits = next_logits / temperature

        # ### NOVI KOD ZA Top-P UZORKOVANJE - START ###
        # Top-P filtriranje se primjenjuje nakon temperature i kazne.
        # Ono dinamički uklanja tokene s niskom vjerojatnošću.
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Ukloni tokene koji su izvan praga `top_p`
            sorted_indices_to_remove = cumulative_probs > top_p
            # Pomičemo udesno da zadržimo prvi token koji je prešao prag
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            next_logits[indices_to_remove] = float('-inf')
        # ### NOVI KOD ZA Top-P UZORKOVANJE - KRAJ ###

        # --- 1. CALCULATE PROBABILITIES ---
        # Calculate absolute probabilities over the entire vocabulary by applying softmax
        all_probs = F.softmax(next_logits, dim=-1)

        # --- 2. PERFORM TOP-K SAMPLING ---
        #top_k = 10
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
        print(f"Input [{clean_decoded_text(tokenizer.decode(generated_ids[org_tokens_len:]))}]")
        print(a)
        print(f"Top {top_k} Candidates:")
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

# @torch.no_grad()
# def generate_greedy(
#         model,
#         tokenizer,
#         prompt: str,
#         device: torch.device,
#         ctx_size: int,
#         max_new_tokens: int = 100,
#         print_each: bool = True,
#         temperature: float = 0.5,
#         repetition_penalty: float = 1.25,
#         top_k: int = 50,
#         top_p: float = 0.8  # Vrijednosti blizu 1.0 su manje restriktivne, a blizu 0 su više.
# ) -> str:
#     # NOTE: Using the KV cache is much more efficient.
#     # Set this to False to enable your KV cache implementation.
#     disable_cache = True
#
#     """
#     Generates text. This version has been simplified to use greedy decoding
#     to produce more stable and coherent output.
#     """
#     # 1. Encode prompt and prepare initial state
#     token_ids = tokenizer.encode(prompt).ids
#     input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
#     org_tokens_len = len(token_ids)
#     generated_ids = token_ids.copy()
#
#     print(f"\n[INFO] Starting generation: prompt_len={len(token_ids)}, ctx_size={ctx_size}")
#     print(f"[PROMPT] {prompt}", end="")
#     start_time = time.time()
#
#     # 2. Initialize the cache. It starts as None.
#     kv_cache = None
#
#     # 3. The main generation loop
#     for step in range(max_new_tokens):
#         if not disable_cache:
#             # --- A. SLIDING WINDOW LOGIC FOR THE KV CACHE ---
#             if kv_cache is not None:
#                 first_key_tensor = kv_cache[0][0][0]
#                 cached_seq_len = first_key_tensor.size(1)
#                 if cached_seq_len >= ctx_size:
#                     new_kv_cache_after_sliding = []
#                     for layer_cache in kv_cache:
#                         new_layer_cache = []
#                         for head_k, head_v in layer_cache:
#                             k_trimmed, v_trimmed = head_k[:, 1:, :], head_v[:, 1:, :]
#                             new_layer_cache.append((k_trimmed, v_trimmed))
#                         new_kv_cache_after_sliding.append(new_layer_cache)
#                     kv_cache = new_kv_cache_after_sliding
#
#             # --- B. MODEL FORWARD PASS ---
#             logits, kv_cache = model.inference_step(input_ids, kv_cache)
#             # On subsequent steps, input_ids is just the new token
#             input_ids = None
#
#         else: # This path is taken if cache is disabled
#             input_for_model = torch.tensor([generated_ids[-ctx_size:]], device=device)
#             logits = model(input_for_model)
#
#         # --- C. SAMPLE NEXT TOKEN (MODIFIED FOR COHERENCE) ---
#         # Get the logits for the very last token in the sequence
#         next_logits = logits[0, -1, :]
#
#         # Optional: Apply repetition penalty. This is still useful with greedy decoding.
#         if repetition_penalty != 1.0 and step > 0:
#             for token_id in set(generated_ids[-ctx_size:]): # Check recent tokens
#                 if next_logits[token_id] > 0:
#                     next_logits[token_id] /= repetition_penalty
#                 else:
#                     next_logits[token_id] *= repetition_penalty
#
#         # Perform greedy decoding: pick the single most likely token.
#         # This is much more stable than sampling and prevents topic drift.
#         next_id = torch.argmax(next_logits, dim=-1).item()
#
#         # Update generated sequence
#         generated_ids.append(next_id)
#         if not disable_cache:
#              input_ids = torch.tensor([[next_id]], dtype=torch.long, device=device)
#
#         # --- D. PRINT AND CHECK FOR STOP TOKEN ---
#         if print_each:
#             # We decode and clean just the new token for streaming output
#             print(clean_decoded_text(tokenizer.decode([next_id])), end="", flush=True)
#
#         # Check for an end-of-sequence token (e.g., ID 2 for EOS)
#         if next_id == 2:
#             print("<EOT>", end="", flush=True)
#             #break
#
#     # 4. Final cleanup and return
#     print()  # Final newline
#     elapsed = time.time() - start_time
#     tps = (step + 1) / elapsed if elapsed > 0 else float('inf')
#     print(f"\n[INFO] Generation of {step + 1} new tokens took {elapsed:.3f}s ({tps:.2f} tokens/sec)")
#
#     full_text = tokenizer.decode(generated_ids)
#     return full_text


@torch.no_grad()
def generate_beam_search_robust(
        model,
        tokenizer,
        prompt: str,
        device: torch.device,
        max_new_tokens: int = 100,
        min_new_tokens: int = 10,
        beam_size: int = 5,
        repetition_penalty: float = 1.2,
        length_penalty_alpha: float = 0.7,
        no_repeat_ngram_size: int = 3,
        top_k_filter: int = 0,  # Onemogući po defaultu u korist Top-P
        top_p_filter: float = 0.92  # Omogući Top-P po defaultu
) -> str:
    """
    Generira tekst koristeći robusnu Beam Search verziju.
    Primjenjuje Top-K/Top-P filtriranje PRIJE kazni kako bi se spriječila degeneracija modela.
    """
    model.eval()
    eos_token_id = tokenizer.eos_token_id if hasattr(tokenizer,
                                                     'eos_token_id') and tokenizer.eos_token_id is not None else 2

    print(
        f"[INFO] Starting Robust Beam Search: beam_size={beam_size}, top_k={top_k_filter}, top_p={top_p_filter}, ngram={no_repeat_ngram_size}")
    start_time = time.time()

    prompt_ids = tokenizer.encode(prompt).ids
    prompt_len = len(prompt_ids)
    initial_input = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    logits, kv_caches = model.inference_step(initial_input, None)
    next_logits = logits[0, -1, :]

    log_probs = F.log_softmax(next_logits, dim=-1)
    top_log_probs, top_indices = torch.topk(log_probs, beam_size)

    live_hypotheses = torch.cat([initial_input.repeat(beam_size, 1), top_indices.view(-1, 1)], dim=1)
    live_scores = top_log_probs

    expanded_kv_caches = []
    for layer_cache in kv_caches:
        new_layer_cache = []
        for head_k, head_v in layer_cache:
            head_k_expanded = head_k.repeat(beam_size, 1, 1)
            head_v_expanded = head_v.repeat(beam_size, 1, 1)
            new_layer_cache.append((head_k_expanded, head_v_expanded))
        expanded_kv_caches.append(new_layer_cache)
    kv_caches = expanded_kv_caches

    completed_hypotheses = []

    for step in range(1, max_new_tokens):
        if not live_hypotheses.numel(): break

        print(f"\n===== BEAM STEP {step + 1}/{max_new_tokens} =====")
        input_ids = live_hypotheses[:, -1].view(-1, 1)

        logits, kv_caches = model.inference_step(input_ids, kv_caches)
        next_logits = logits[:, -1, :]

        # --- REVIDIRANI REDOSLIJED ---

        # 1. KAZNE - primjenjuju se PRVO kako bi se modificirala cijela distribucija logita
        if repetition_penalty != 1.0:
            for i in range(live_hypotheses.shape[0]):
                generated_tokens = live_hypotheses[i][prompt_len:]
                if generated_tokens.numel() > 0:
                    for token_id in set(generated_tokens.tolist()):
                        if next_logits[i, token_id] > 0:
                            next_logits[i, token_id] /= repetition_penalty
                        else:
                            next_logits[i, token_id] *= repetition_penalty

        if no_repeat_ngram_size > 0:
            for i in range(live_hypotheses.shape[0]):
                current_sequence = live_hypotheses[i].tolist()
                if len(current_sequence) < no_repeat_ngram_size: continue
                generated_ngrams = {tuple(current_sequence[j:j + no_repeat_ngram_size]) for j in
                                    range(len(current_sequence) - no_repeat_ngram_size + 1)}
                if not generated_ngrams: continue
                prefix = tuple(current_sequence[-(no_repeat_ngram_size - 1):])
                for token_id in range(next_logits.shape[1]):
                    if prefix + (token_id,) in generated_ngrams:
                        next_logits[i, token_id] = float('-inf')

        # 2. FILTRIRANJE (Top-K ili Top-P) - primjenjuje se na "očišćenim" logitima
        if top_k_filter > 0:
            k_values, _ = torch.topk(next_logits, top_k_filter, dim=-1)
            min_k_value = k_values[:, -1].unsqueeze(-1)
            next_logits[next_logits < min_k_value] = float('-inf')

        elif 0 < top_p_filter < 1.0:
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p_filter
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            for i in range(next_logits.shape[0]):
                indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                next_logits[i][indices_to_remove] = float('-inf')

        log_probs = F.log_softmax(next_logits, dim=-1)
        # ... ostatak logike je uglavnom isti ...

        candidate_scores = live_scores.unsqueeze(1) + log_probs
        flat_candidate_scores = candidate_scores.view(-1)

        # Uzmi malo više kandidata radi prostora za one koji završavaju
        top_scores, top_indices = torch.topk(flat_candidate_scores, k=beam_size * 2)
        beam_indices = top_indices // log_probs.size(1)
        token_indices = top_indices % log_probs.size(1)

        new_live_hypotheses_list, new_live_scores_list, new_beam_indices = [], [], []

        for score, beam_idx, token_id in zip(top_scores, beam_indices, token_indices):
            num_generated = len(live_hypotheses[beam_idx]) - prompt_len
            # Hipoteza završava samo ako je dosegla min duljinu
            if token_id.item() == eos_token_id and num_generated >= min_new_tokens:
                seq = torch.cat([live_hypotheses[beam_idx], token_id.view(1)])
                length = len(seq)
                normalized_score = score.item() / ((5 + length) / 6) ** length_penalty_alpha
                completed_hypotheses.append({'seq': seq, 'score': normalized_score})
            # Inače, ako nije EOS, nastavi je graditi
            elif token_id.item() != eos_token_id:
                if len(new_live_hypotheses_list) < beam_size:
                    new_seq = torch.cat([live_hypotheses[beam_idx], token_id.view(1)])
                    new_live_hypotheses_list.append(new_seq)
                    new_live_scores_list.append(score)
                    new_beam_indices.append(beam_idx.item())

        if not new_live_hypotheses_list: break

        live_hypotheses = torch.stack(new_live_hypotheses_list)
        live_scores = torch.stack(new_live_scores_list)

        # VRAĆENO LOGIRANJE
        print("\n[TOP BEAMS FOR NEXT STEP]")
        for i in range(live_hypotheses.shape[0]):
            decoded_text = clean_decoded_text(tokenizer.decode(live_hypotheses[i].tolist()))
            print(f"  > \"{decoded_text}\" | Raw Score: {live_scores[i].item():.4f}")

        reorder_indices = torch.tensor(new_beam_indices, device=device)
        kv_caches = _reorder_cache(kv_caches, reorder_indices)  # Pretpostavlja se da _reorder_cache funkcija postoji

    # Finalizacija ...
    if not completed_hypotheses:
        completed_hypotheses.extend([
            {'seq': h, 'score': s.item() / ((5 + len(h)) / 6) ** length_penalty_alpha}
            for h, s in zip(live_hypotheses, live_scores)
        ])

    if not completed_hypotheses: return tokenizer.decode(prompt_ids)

    best_beam = sorted(completed_hypotheses, key=lambda x: x['score'], reverse=True)[0]
    best_sequence_ids = best_beam['seq'].tolist()

    final_text = clean_decoded_text(tokenizer.decode(best_sequence_ids))
    print(f"\n[INFO] Robust Beam Search finished.")
    print(f"[INFO] Najbolji završni beam: \"{final_text}\" (Score: {best_beam['score']:.4f})")

    return tokenizer.decode(best_sequence_ids)


def _reorder_cache(kv_caches, reorder_indices):
    """Pomoćna funkcija za preuređivanje KV cache-a."""
    if not reorder_indices.numel(): return kv_caches
    reordered_kv_caches = []
    for layer_cache in kv_caches:
        new_layer_cache = []
        for head_k, head_v in layer_cache:
            reordered_k = head_k.index_select(0, reorder_indices)
            reordered_v = head_v.index_select(0, reorder_indices)
            new_layer_cache.append((reordered_k, reordered_v))
        reordered_kv_caches.append(new_layer_cache)
    return reordered_kv_caches


def _reorder_cache(kv_caches, reorder_indices):
    """Pomoćna funkcija za preuređivanje KV cache-a."""
    reordered_kv_caches = []
    for layer_cache in kv_caches:
        new_layer_cache = []
        for head_k, head_v in layer_cache:
            reordered_k = head_k.index_select(0, reorder_indices)
            reordered_v = head_v.index_select(0, reorder_indices)
            new_layer_cache.append((reordered_k, reordered_v))
        reordered_kv_caches.append(new_layer_cache)
    return reordered_kv_caches


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
    print(clean_decoded_text(generated))


if __name__ == "__main__":
    main()