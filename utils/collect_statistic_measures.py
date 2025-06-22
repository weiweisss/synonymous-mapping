import os
import json
import torch
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.calculate_metrics import calculate_metrics

# --- Configuration ---
MODEL_NAME = "Qwen/Qwen3-4B"
DECODING_STRATEGY = "greedy"  # Options: "greedy" or "sample"


def analyze_single_prompt(
        prompt,
        model=None,
        tokenizer=None,
        model_name=MODEL_NAME,
        decoding_strategy=DECODING_STRATEGY,
        output_dir="outputs",
        dataset_name="unknown"
):
    """
    针对单条prompt进行分析，并将结果保存到有序的输出目录。
    现在可以接受预加载的model和tokenizer

    Args:
        prompt: 输入的提示文本
        model: 预加载的模型（可选）
        tokenizer: 预加载的分词器（可选）
        model_name: 模型名称
        decoding_strategy: 解码策略
        output_dir: 输出目录
        dataset_name: 数据集名称（用于文件命名和元数据）
    """
    # 如果没有传入模型和tokenizer，则加载它们
    if model is None or tokenizer is None:
        print(f"Loading model: {model_name}...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded successfully.")

    # Prepare model inputs
    messages = [{"role": "user", "content": prompt}]

    # Enable thinking mode if the model name suggests it
    thinking_mode_enabled = "Qwen3" in model_name
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=thinking_mode_enabled
    )
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    input_ids_len = model_inputs.input_ids.shape[1]

    # Configure and run generation
    generation_kwargs = {
        "max_new_tokens": 40000, "output_scores": True, "return_dict_in_generate": True
    }
    if decoding_strategy == "sample":
        generation_kwargs.update({
            "do_sample": True,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "min_p": 0})
        print(f"Using 'sample' decoding.")
    else:
        generation_kwargs.update({
            "do_sample": False})
        print("Using 'greedy' decoding.")

    print("\nGenerating response...")
    with torch.no_grad():
        outputs = model.generate(**model_inputs, **generation_kwargs)
    print("Generation complete.")

    # Extract generated tokens and logits
    all_generated_ids = outputs.sequences[0][input_ids_len:]
    all_logits = [score[0] for score in outputs.scores]

    # --- Split generation into 'thinking' and 'response' parts ---
    split_index = -1
    if thinking_mode_enabled:
        try:
            think_end_token_id = tokenizer.convert_tokens_to_ids("</think>")
            split_index = (all_generated_ids == think_end_token_id).nonzero(as_tuple=True)[0][0].item() + 1
        except (IndexError, AttributeError):
            print("Warning: '</think>' token not found in generation.")
            split_index = -1

    # --- Calculate metrics for each part ---
    print("Analyzing metrics for all parts...")
    analysis = {}

    # Overall Analysis
    analysis['overall'] = calculate_metrics(all_logits, all_generated_ids)

    if split_index != -1:
        # Thinking Part Analysis
        thinking_ids = all_generated_ids[:split_index]
        thinking_logits = all_logits[:split_index]
        analysis['thinking_part'] = calculate_metrics(thinking_logits, thinking_ids)

        # Response Part Analysis
        response_ids = all_generated_ids[split_index:]
        response_logits = all_logits[split_index:]
        analysis['response_part'] = calculate_metrics(response_logits, response_ids)
    else:
        # If no thinking part, the "response" is the "overall"
        analysis['thinking_part'] = calculate_metrics([], torch.tensor([]))  # Empty
        analysis['response_part'] = analysis['overall']

    # --- Prepare JSON Output ---
    def structure_analysis_part(part_name, part_ids, metrics):
        return {
            "metrics": {
                "sequence_shannon_entropy_sum_bits": metrics.get("sequence_shannon_entropy_sum_bits", 0),
                "sequence_cross_entropy_bits": metrics.get("sequence_cross_entropy_bits", 0),
                "perplexity": metrics.get("perplexity", 1.0)
            },
            "content": tokenizer.decode(part_ids, skip_special_tokens=True).strip(),
            "per_token_analysis": [
                {
                    "token_index": i + 1,
                    "token": tokenizer.decode(token_id),
                    "self_information_bits": metrics.get("self_informations_bits", [])[i],
                    "step_shannon_entropy_bits": metrics.get("per_step_shannon_entropy_bits", [])[i],
                }
                for i, token_id in enumerate(part_ids)
            ]
        }

    results_data = {
        "metadata": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "prompt": prompt,
            "decoding_strategy": decoding_strategy,
            "generation_timestamp_utc": datetime.utcnow().isoformat(),
        },
        "analysis_breakdown": {
            "overall": structure_analysis_part("overall", all_generated_ids, analysis['overall']),
            "thinking_part": structure_analysis_part(
                "thinking_part",
                all_generated_ids[:split_index] if split_index != -1 else torch.tensor([]),
                analysis['thinking_part']
            ),
            "response_part": structure_analysis_part(
                "response_part",
                all_generated_ids[split_index:] if split_index != -1 else all_generated_ids,
                analysis['response_part']
            )
        }
    }

    # --- Save JSON Output to organized directory ---
    date_str = datetime.now().strftime('%Y%m%d')
    safe_model_name = model_name.replace('/', '_')
    safe_dataset_name = dataset_name.replace('/', '_').replace('-', '_')
    save_dir = os.path.join(output_dir, safe_model_name, decoding_strategy, date_str)
    os.makedirs(save_dir, exist_ok=True)

    # 在文件名中包含数据集名称
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f"analysis_{safe_model_name}_{safe_dataset_name}_{decoding_strategy}_{timestamp}.json"
    output_path = os.path.join(save_dir, output_filename)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=4, ensure_ascii=False)

    print(f"\nAnalysis complete. Results saved to '{output_path}'")
    print(f"\nOverall Perplexity: {analysis['overall']['perplexity']:.4f}")
    if split_index != -1:
        print(f"Thinking Perplexity: {analysis['thinking_part']['perplexity']:.4f}")
        print(f"Response Perplexity: {analysis['response_part']['perplexity']:.4f}")
    return output_path


if __name__ == "__main__":
    PROMPT = "A robe takes 2 bolts of blue fiber and half that much white fiber.  How many bolts in total does it take?"
    analyze_single_prompt(PROMPT)