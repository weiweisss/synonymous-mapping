import json
import os
import datetime
import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.collect_statistic_measures import analyze_single_prompt
from utils.preprocess_dataset import get_dataset_prompts


# --- 命令行参数解析 ---
def parse_args():
    parser = argparse.ArgumentParser(description="生成和分析模型响应")
    parser.add_argument('--model_name', type=str, default="Qwen/Qwen3-4B",
                        choices=[
                            "Qwen/Qwen3-4B",
                            "Qwen/Qwen3-8B",
                            "Qwen/Qwen3-14B"
                        ],
                        help="要使用的模型名称")
    parser.add_argument('--decoding_strategy', type=str, default="sample",
                        choices=["greedy", "sample"], help="解码策略")
    parser.add_argument('--output_dir', type=str, default="outputs",
                        help="输出目录")
    parser.add_argument('--dataset_name', type=str, default="manual",
                        help="数据集名称")
    parser.add_argument('--data_path', type=str, default=None,
                        help="数据集路径（如果适用）")
    parser.add_argument('--max_samples', type=int, default=0,
                        help="最大样本数 (设置为0或负数表示处理所有数据)")
    parser.add_argument('--question_key', type=str, default="question",
                        help="问题字段的键名")
    return parser.parse_args()


def get_prompt_list(args):
    """根据配置获取prompt列表"""
    try:
        # 如果max_samples为0或负数，则设置为None表示处理所有数据
        max_samples = args.max_samples if args.max_samples > 0 else None

        prompt_list = get_dataset_prompts(
            dataset_name=args.dataset_name,
            data_path=args.data_path,
            max_samples=max_samples,
            question_key=args.question_key
        )
        print(f"Successfully loads {len(prompt_list)} prompts from {args.dataset_name} datasets")
        return prompt_list
    except Exception as e:
        print(f"加载数据集失败: {e}")
        print("使用默认prompt列表")
        return [
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "What is the capital of France?"
        ]

# --- Main execution logic ---
def main():
    # 解析命令行参数
    args = parse_args()

    # 1. --- 加载模型和tokenizer（仅一次） ---
    print(f"Loading model: {args.model_name}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Fatal: Could not load model or tokenizer. Error: {e}")
        return

    # 2. --- 获取prompt列表 ---
    prompt_list = get_prompt_list(args)
    if not prompt_list:
        print("No prompts to process. Exiting.")
        return

    # 3. --- 准备输出目录 ---
    date_str = datetime.datetime.now().strftime('%Y%m%d')
    safe_model_name = args.model_name.replace('/', '_')
    summary_save_dir = os.path.join(args.output_dir, safe_model_name, args.decoding_strategy, date_str)
    os.makedirs(summary_save_dir, exist_ok=True)

    print(f"\nStarting generation and analysis. The results will be saved to: {summary_save_dir}")
    result_paths = []

    # 4. --- 处理每个prompt ---
    for i, prompt in enumerate(prompt_list, 1):
        print(f"\n--- Processing prompt {i}/{len(prompt_list)} ---")
        print(f"Content: {prompt}")

        try:
            # 使用预加载的模型和tokenizer
            result_path = analyze_single_prompt(
                prompt=prompt,
                model=model,
                tokenizer=tokenizer,
                model_name=args.model_name,
                decoding_strategy=args.decoding_strategy,
                output_dir=args.output_dir,
                dataset_name=args.dataset_name
            )
            result_paths.append({"prompt": prompt, "result_path": result_path})
            print(f"✓ Completed. Result saved to: {result_path}")
        except Exception as e:
            print(f"✗ Processing failed: {e}")
            result_paths.append({"prompt": prompt, "result_path": None, "error": str(e)})

    # 5. --- 保存所有结果摘要 ---
    safe_dataset_name = args.dataset_name.replace('/', '_').replace('-', '_')
    summary_filename = f"{safe_dataset_name}_generation_summary.json"
    result_json_path = os.path.join(summary_save_dir, summary_filename)

    # 在摘要中添加数据集信息
    summary_data = {
        "dataset_info": {
            "dataset_name": args.dataset_name,
            "total_prompts": len(prompt_list),
            "max_samples": args.max_samples,
            "generation_date": date_str,
            "model_name": args.model_name,
            "decoding_strategy": args.decoding_strategy
        },
        "results": result_paths
    }

    with open(result_json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=4, ensure_ascii=False)
    print(f"\n--- All prompts processed. Summary saved to: {result_json_path} ---")


if __name__ == "__main__":
    main()