import numpy as np
import torch


def calculate_metrics(logits_list, generated_ids):
    """
    Calculates a detailed set of metrics for a given sequence.

    Args:
        logits_list (list of torch.Tensor): Logits for each generated token.
        generated_ids (torch.Tensor): The sequence of generated token IDs.

    Returns:
        dict: A dictionary containing all calculated metrics.
    """
    if not logits_list or generated_ids.numel() == 0:
        return {
            "self_informations_bits": [],
            "per_step_shannon_entropy_bits": [],
            "sequence_shannon_entropy_sum_bits": 0,
            "sequence_cross_entropy_bits": 0,
            "perplexity": 1.0  # Perplexity of an empty sequence is 1
        }

    self_informations = []
    per_step_shannon_entropies = []

    # 1. Calculate Per-Token Self-Information and Per-Step Shannon Entropy
    for i, logits in enumerate(logits_list):
        # Use log_softmax for numerical stability
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        probs = torch.exp(log_probs)

        # a) Self-Information of the chosen token
        token_id = generated_ids[i]
        chosen_token_log_prob = log_probs[token_id].item()
        self_info_bits = -chosen_token_log_prob / np.log(2)
        self_informations.append(self_info_bits)

        # b) Shannon Entropy of the entire probability distribution at this step
        # H(P) = -sum(p * log2(p)) = -sum(p * log(p) / ln(2))

        # --- MODIFICATION START ---
        # Handle the case where p=0, which causes 0 * log(0) -> NaN
        # This could happen when using 'sample' generation strategy
        # We define 0 * log(0) = 0
        product = probs * log_probs
        # Replace NaN values with 0
        product[torch.isnan(product)] = 0
        shannon_entropy_bits = -torch.sum(product / np.log(2)).item()
        # --- MODIFICATION END ---

        per_step_shannon_entropies.append(shannon_entropy_bits)

    # 2. Calculate final sequence-level metrics
    # a) Sequence Shannon Entropy (Sum of per-step uncertainties), as requested
    sequence_shannon_entropy_sum = np.sum(per_step_shannon_entropies)

    # b) Sequence Cross-Entropy (Average Self-Information), basis for Perplexity
    sequence_cross_entropy = np.mean(self_informations)

    # c) Perplexity: PPL = 2^H, where H is the cross-entropy
    perplexity = np.power(2, sequence_cross_entropy)

    return {
        "self_informations_bits": self_informations,
        "per_step_shannon_entropy_bits": per_step_shannon_entropies,
        "sequence_shannon_entropy_sum_bits": sequence_shannon_entropy_sum,
        "sequence_cross_entropy_bits": sequence_cross_entropy,
        "perplexity": perplexity
    }