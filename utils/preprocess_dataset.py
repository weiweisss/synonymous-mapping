import json
import os
import csv
import pandas as pd
from typing import List, Dict, Any

PREFIX = "/media/Dataset/cot_datasets/"


def load_json(file_path: str) -> Any:
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def load_jsonl(file_path: str) -> List[Dict]:
    """加载JSONL文件（每行一个JSON对象）"""
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def load_parquet(file_path: str) -> pd.DataFrame:
    """加载Parquet文件"""
    return pd.read_parquet(file_path)


def preprocess_gsm8k(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理GSM8K数据集
    GSM8K格式: {"question": "...", "answer": "..."}
    """
    print(f"Processing GSM8K dataset, the data_path is {data_path}")
    if data_path.endswith('.jsonl'):
        data = load_jsonl(data_path)
    else:
        data = load_json(data_path)

    prompt_list = []
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break

        question = item.get('question', '')
        if question:
            prompt_list.append(question.strip())

    return prompt_list


def preprocess_reclor(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理ReClor数据集
    ReClor格式: {"context": "...", "question": "...", "answers": [...], "label": ...}
    """
    print(f"Processing ReClor dataset, the data_path is {data_path}")
    if data_path.endswith('.jsonl'):
        data = load_jsonl(data_path)
    else:
        data = load_json(data_path)

    prompt_list = []
    for i, item in enumerate(data):
        if max_samples and i >= max_samples:
            break

        context = item.get('context', '')
        question = item.get('question', '')
        if context and question:
            # 组合上下文和问题
            prompt = f"{context}\n\nQuestion: {question}"
            prompt_list.append(prompt.strip())

    return prompt_list


def preprocess_math(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理MATH数据集
    MATH格式: 每个JSON文件包含 {"problem": "...", "solution": "..."}
    """
    print(f"Processing MATH dataset, the data_path is {data_path}")

    prompt_list = []
    count = 0

    # MATH数据集是按目录组织的，每个文件是一个问题
    if os.path.isdir(data_path):
        # 遍历所有子目录
        for subject in os.listdir(data_path):
            subject_path = os.path.join(data_path, subject)
            if os.path.isdir(subject_path):
                for filename in os.listdir(subject_path):
                    if filename.endswith('.json'):
                        if max_samples and count >= max_samples:
                            break

                        file_path = os.path.join(subject_path, filename)
                        try:
                            item = load_json(file_path)
                            problem = item.get('problem', '')
                            if problem:
                                prompt_list.append(problem.strip())
                                count += 1
                        except Exception as e:
                            print(f"Error loading {file_path}: {e}")
                            continue
                if max_samples and count >= max_samples:
                    break
    else:
        # 如果是单个文件
        if data_path.endswith('.jsonl'):
            data = load_jsonl(data_path)
        else:
            data = load_json(data_path)

        for i, item in enumerate(data):
            if max_samples and i >= max_samples:
                break

            problem = item.get('problem', '')
            if problem:
                prompt_list.append(problem.strip())

    return prompt_list


def preprocess_h_arc(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理H-ARC数据集
    H-ARC格式: CSV文件，包含网格转换任务
    实际列名: exp_name, task_type, hashed_id, joint_id_task, task_name, task_number,
             attempt_number, num_actions, solved, test_output_grid, first_written_solution,
             last_written_solution, complete
    """
    print(f"Processing H-ARC dataset, the data_path is {data_path}")

    prompt_list = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break

                # H-ARC实际的列名
                task_name = row.get('task_name', '')
                test_output_grid = row.get('test_output_grid', '')
                first_solution = row.get('first_written_solution', '')

                if task_name and test_output_grid:
                    prompt = f"Task: {task_name}\nOutput Grid: {test_output_grid}"
                    if first_solution:
                        prompt += f"\nFirst Solution: {first_solution}"
                    prompt_list.append(prompt.strip())

        print(f"成功处理H-ARC数据集，获得 {len(prompt_list)} 个样本")
    except Exception as e:
        print(f"处理H-ARC数据集时出错: {e}")
        print(f"请检查文件路径: {data_path}")

    return prompt_list


def preprocess_swag(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理SWAG数据集
    SWAG格式: CSV文件，包含常识推理选择题
    实际列名: video-id, fold-ind, startphrase, sent1, sent2, gold-source,
             ending0, ending1, ending2, ending3, label
    """
    print(f"Processing SWAG dataset, the data_path is {data_path}")

    prompt_list = []
    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if max_samples and i >= max_samples:
                    break

                # SWAG包含sent1和sent2构成的情境描述
                sent1 = row.get('sent1', '')
                sent2 = row.get('sent2', '')
                startphrase = row.get('startphrase', '')

                # 优先使用startphrase，如果没有则组合sent1和sent2
                if startphrase:
                    context = startphrase.strip()
                elif sent1:
                    context = sent1.strip()
                    if sent2:
                        context += " " + sent2.strip()
                else:
                    continue  # 跳过没有有效内容的行

                if context:
                    prompt_list.append(context)

        print(f"成功处理SWAG数据集，获得 {len(prompt_list)} 个样本")
    except Exception as e:
        print(f"处理SWAG数据集时出错: {e}")
        print(f"请检查文件路径: {data_path}")

    return prompt_list


def preprocess_drop(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理DROP数据集
    DROP格式: Parquet文件，包含阅读理解问题
    字段: section_id, query_id, passage, question, answers_spans
    """
    print(f"Processing DROP dataset, the data_path is {data_path}")

    # 加载parquet文件
    df = load_parquet(data_path)

    # 显示数据集基本信息
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    prompt_list = []

    # 处理样本 - DROP需要结合passage和question
    for i, row in df.iterrows():
        if max_samples and i >= max_samples:
            break

        passage = row.get('passage', '')
        question = row.get('question', '')

        if pd.notna(passage) and pd.notna(question) and passage and question:
            # 组合passage和question形成完整的阅读理解prompt
            prompt = f"Passage: {passage.strip()}\n\nQuestion: {question.strip()}"
            prompt_list.append(prompt)

    return prompt_list


def preprocess_numina_math_cot(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理NuminaMath-CoT数据集
    NuminaMath-CoT格式: Parquet文件，包含数学推理问题
    字段: source, problem, solution, messages
    """
    print(f"Processing NuminaMath-CoT dataset, the data_path is {data_path}")

    # 加载parquet文件
    df = load_parquet(data_path)

    # 显示数据集基本信息
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    prompt_list = []

    # 处理样本 - 使用problem字段作为问题
    for i, row in df.iterrows():
        if max_samples and i >= max_samples:
            break

        problem = row.get('problem', '')

        if pd.notna(problem) and problem:
            prompt_list.append(problem.strip())

    return prompt_list


def preprocess_cot_gsm8k(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理CoT-GSM8K数据集
    CoT-GSM8K格式: Parquet文件，包含思维链推理的数学问题
    字段: question, answer, prompt, response
    """
    print(f"Processing CoT-GSM8K dataset, the data_path is {data_path}")

    # 加载parquet文件
    df = load_parquet(data_path)

    # 显示数据集基本信息
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    prompt_list = []

    # 处理样本 - 使用question字段作为问题
    for i, row in df.iterrows():
        if max_samples and i >= max_samples:
            break

        question = row.get('question', '')

        if pd.notna(question) and question:
            prompt_list.append(question.strip())

    return prompt_list


def preprocess_cot_flan(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理CoT-FLAN数据集
    CoT-FLAN格式: Parquet文件，包含指令跟随和推理任务
    字段: instruction, input, output
    """
    print(f"Processing CoT-FLAN dataset, the data_path is {data_path}")

    # 加载parquet文件
    df = load_parquet(data_path)

    # 显示数据集基本信息
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    prompt_list = []

    # 处理样本 - 组合instruction和input作为完整的问题
    for i, row in df.iterrows():
        if max_samples and i >= max_samples:
            break

        instruction = row.get('instruction', '')
        input_text = row.get('input', '')

        if pd.notna(instruction) and instruction:
            # 如果有input，则组合instruction和input
            if pd.notna(input_text) and input_text:
                prompt = f"{instruction.strip()}\n\nInput: {input_text.strip()}"
            else:
                prompt = instruction.strip()
            prompt_list.append(prompt)

    return prompt_list


def preprocess_commonsense_qa(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理CommonsenseQA数据集
    CommonsenseQA格式: Parquet文件，包含常识推理选择题
    字段: id, question, question_concept, choices, answerKey
    """
    print(f"Processing CommonsenseQA dataset, the data_path is {data_path}")

    # 加载parquet文件
    df = load_parquet(data_path)

    # 显示数据集基本信息
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    prompt_list = []

    # 处理样本 - 使用question字段作为问题
    for i, row in df.iterrows():
        if max_samples and i >= max_samples:
            break

        question = row.get('question', '')

        if pd.notna(question) and question:
            prompt_list.append(question.strip())

    return prompt_list


def preprocess_aquarat(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理AQuA-RAT数据集
    AQuA-RAT格式: Parquet文件，包含数学推理选择题
    字段: question, options, rationale, correct
    """
    print(f"Processing AQuA-RAT dataset, the data_path is {data_path}")

    # 加载parquet文件
    df = load_parquet(data_path)

    # 显示数据集基本信息
    print(f"数据集形状: {df.shape}")
    print(f"列名: {list(df.columns)}")

    prompt_list = []

    # 处理样本 - 使用question字段作为问题
    for i, row in df.iterrows():
        if max_samples and i >= max_samples:
            break

        question = row.get('question', '')

        if pd.notna(question) and question:
            prompt_list.append(question.strip())

    return prompt_list


def preprocess_cot_collection(data_path: str, max_samples: int = None) -> List[str]:
    """
    预处理CoT-Collection数据集
    CoT-Collection格式: 大型JSON文件，包含思维链推理样本
    字段: source, target, rationale, config, task, prompt
    注意: 这是一个大文件(2.25GB)，建议使用较小的max_samples值
    """
    print(f"Processing CoT-Collection dataset, the data_path is {data_path}")
    print("警告: 这是一个大文件，正在流式处理...")

    prompt_list = []
    processed_count = 0

    try:
        with open(data_path, 'r', encoding='utf-8') as f:
            # 跳过开头的 "{"
            f.read(1)

            current_entry = ""
            brace_count = 0
            in_string = False
            escape_next = False

            while True:
                char = f.read(1)
                if not char:  # 文件结束
                    break

                current_entry += char

                # 处理字符串和转义
                if escape_next:
                    escape_next = False
                    continue

                if char == '\\':
                    escape_next = True
                    continue

                if char == '"' and not escape_next:
                    in_string = not in_string
                    continue

                if in_string:
                    continue

                # 计算大括号
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1

                    # 如果找到完整的条目
                    if brace_count == 0:
                        try:
                            # 找到键值对的分隔点
                            if '":' in current_entry:
                                # 分割键和值
                                colon_pos = current_entry.find('":')
                                if colon_pos > 0:
                                    # 提取值部分（从冒号后开始）
                                    value_part = current_entry[colon_pos + 2:].strip()
                                    if value_part.endswith(','):
                                        value_part = value_part[:-1]  # 移除末尾逗号

                                    # 解析值部分
                                    entry_value = json.loads(value_part)

                                    # 提取source字段
                                    source = entry_value.get('source', '')
                                    if source and source.strip():
                                        prompt_list.append(source.strip())
                                        processed_count += 1

                                        if processed_count % 1000 == 0:
                                            print(f"已处理 {processed_count} 个样本...")

                                        if max_samples and processed_count >= max_samples:
                                            print(f"达到最大样本数 {max_samples}，停止处理")
                                            return prompt_list

                        except json.JSONDecodeError as e:
                            # 跳过无法解析的条目
                            if processed_count < 10:  # 只在前10个错误时显示详细信息
                                print(f"JSON解析错误: {e}")
                                print(f"问题条目前100字符: {current_entry[:100]}...")
                        except Exception as e:
                            print(f"处理条目时出错: {e}")

                        # 重置状态
                        current_entry = ""
                        brace_count = 0

                        # 跳过可能的逗号
                        next_char = f.read(1)
                        if next_char and next_char not in [',', '}']:
                            f.seek(f.tell() - 1)  # 回退一个字符

    except Exception as e:
        print(f"处理文件时出错: {e}")

    print(f"总共处理了 {processed_count} 个样本")
    return prompt_list


def get_dataset_prompts(dataset_name: str, data_path: str = None, max_samples: int = None, **kwargs) -> List[str]:
    """
    根据数据集名称获取prompt列表

    Args:
        dataset_name: 数据集名称 ("gsm8k", "reclor", "math", "h-arc", "swag", "swagaf-master", "drop", "NuminaMath-CoT", "cot_gsm8k", "cot_flan", "commonsense_qa", "aquarat", "CoT-Collection", "manual")
        data_path: 数据文件路径（如果是预定义数据集则忽略，使用默认路径）
        max_samples: 最大样本数量


    Returns:
        List[str]: prompt列表
    """

    if dataset_name == "manual":
        # 手动定义的prompt列表
        return [
            "A robe takes 2 bolts of blue fiber and half that much white fiber. How many bolts in total does it take?",
            "If you have 3 apples and eat one, how many are left?",
            "What is the capital of France?",
            "Solve for x: 2x + 5 = 15",
            "A train travels 120 miles in 2 hours. What is its average speed?",
        ]

    elif dataset_name == "gsm8k":
        data_path = PREFIX + "gsm8k/grade_school_math/data/train.jsonl"
        return preprocess_gsm8k(data_path, max_samples)

    elif dataset_name == "reclor":
        data_path = PREFIX + "reclor_data/train.json"
        return preprocess_reclor(data_path, max_samples)

    elif dataset_name == "math":
        data_path = PREFIX + "MATH/train"
        return preprocess_math(data_path, max_samples)

    elif dataset_name == "h-arc":
        data_path = PREFIX + "h-arc/summary_data.csv"
        return preprocess_h_arc(data_path, max_samples)

    elif dataset_name == "swag" or dataset_name == "swagaf-master":
        data_path = PREFIX + "swagaf-master/data/train.csv"
        return preprocess_swag(data_path, max_samples)

    elif dataset_name == "drop":
        data_path = PREFIX + "drop/datasets--ucinlp--drop/snapshots/95cda593fae71b60b5b19f82de3fcf3298c1239c/data/train-00000-of-00001.parquet"
        return preprocess_drop(data_path, max_samples)

    elif dataset_name == "NuminaMath-CoT":
        data_path = PREFIX + "NuminaMath-CoT/datasets--AI-MO--NuminaMath-CoT/snapshots/9d8d210c9f6a36c8f3cd84045668c9b7800ef517/data/train-00000-of-00005.parquet"
        return preprocess_numina_math_cot(data_path, max_samples)

    elif dataset_name == "cot_gsm8k":
        data_path = PREFIX + "cot_gsm8k/datasets--Dahoas--cot_gsm8k/snapshots/390e9dc4b08b5d98d2351dc792ea61a875860a34/data/train-00000-of-00001-8dd500b0958ec712.parquet"
        return preprocess_cot_gsm8k(data_path, max_samples)

    elif dataset_name == "cot_flan":
        data_path = PREFIX + "cot_flan/datasets--causal-lm--cot_flan/snapshots/f4906987ae138c3db3362d46106a3800a9340bd0/data/train-00000-of-00001-a303bca389389dc7.parquet"
        return preprocess_cot_flan(data_path, max_samples)

    elif dataset_name == "commonsense_qa":
        data_path = PREFIX + "commonsense_qa/datasets--tau--commonsense_qa/snapshots/94630fe30dad47192a8546eb75f094926d47e155/data/train-00000-of-00001.parquet"
        return preprocess_commonsense_qa(data_path, max_samples)

    elif dataset_name == "aquarat":
        data_path = PREFIX + "aquarat/datasets--deepmind--aqua_rat/snapshots/33301c6a050c96af81f63cad5562cb5363e88971/raw/train-00000-of-00001.parquet"
        return preprocess_aquarat(data_path, max_samples)

    elif dataset_name == "CoT-Collection":
        data_path = PREFIX + "CoT-Collection/datasets--kaist-ai--CoT-Collection/snapshots/c9d352cdc119df4a4f7526d100e4acb4a72a7a5c/data/CoT_collection_en.json"
        return preprocess_cot_collection(data_path, max_samples)

    else:
        raise ValueError(f"不支持的数据集类型: {dataset_name}")


if __name__ == '__main__':
    # 示例用法
    print("=== 数据集预处理示例 ===")

    # 1. 使用手动定义的prompts
    manual_prompts = get_dataset_prompts("manual")
    print(f"手动定义prompts数量: {len(manual_prompts)}")
    for i, prompt in enumerate(manual_prompts[:3], 1):
        print(f"  {i}. {prompt}")

    # 2. 如果有GSM8K数据集文件，可以这样使用：
    # gsm8k_prompts = get_dataset_prompts("gsm8k", "path/to/gsm8k.jsonl", max_samples=10)
    # print(f"GSM8K prompts数量: {len(gsm8k_prompts)}")

    # 3. 如果有ReClor数据集文件，可以这样使用：
    # reclor_prompts = get_dataset_prompts("reclor", max_samples=10)
    # print(f"ReClor prompts数量: {len(reclor_prompts)}")

    # 4. 如果有MATH数据集文件，可以这样使用：
    # math_prompts = get_dataset_prompts("math", max_samples=10)
    # print(f"MATH prompts数量: {len(math_prompts)}")

    # 5. 如果有H-ARC数据集文件，可以这样使用：
    # h_arc_prompts = get_dataset_prompts("h-arc", max_samples=10)
    # print(f"H-ARC prompts数量: {len(h_arc_prompts)}")

    # 6. 如果有SWAG数据集文件，可以这样使用：
    # swag_prompts = get_dataset_prompts("swag", max_samples=10)
    # print(f"SWAG prompts数量: {len(swag_prompts)}")

    # 7. 如果有DROP数据集文件，可以这样使用：
    # drop_prompts = get_dataset_prompts("drop", max_samples=10)
    # print(f"DROP prompts数量: {len(drop_prompts)}")

    # 8. 如果有NuminaMath-CoT数据集文件，可以这样使用：
    # numina_prompts = get_dataset_prompts("NuminaMath-CoT", max_samples=10)
    # print(f"NuminaMath-CoT prompts数量: {len(numina_prompts)}")

    # 9. 如果有CoT-GSM8K数据集文件，可以这样使用：
    # cot_gsm8k_prompts = get_dataset_prompts("cot_gsm8k", max_samples=10)
    # print(f"CoT-GSM8K prompts数量: {len(cot_gsm8k_prompts)}")

    # 10. 如果有CoT-FLAN数据集文件，可以这样使用：
    # cot_flan_prompts = get_dataset_prompts("cot_flan", max_samples=10)
    # print(f"CoT-FLAN prompts数量: {len(cot_flan_prompts)}")

    # 11. 如果有CommonsenseQA数据集文件，可以这样使用：
    # commonsense_qa_prompts = get_dataset_prompts("commonsense_qa", max_samples=10)
    # print(f"CommonsenseQA prompts数量: {len(commonsense_qa_prompts)}")

    # 12. 如果有AQuA-RAT数据集文件，可以这样使用：
    # aquarat_prompts = get_dataset_prompts("aquarat", max_samples=10)
    # print(f"AQuA-RAT prompts数量: {len(aquarat_prompts)}")

    # 13. 如果有CoT-Collection数据集文件，可以这样使用：
    # cot_collection_prompts = get_dataset_prompts("CoT-Collection", max_samples=100)  # 建议使用较小的样本数
    # print(f"CoT-Collection prompts数量: {len(cot_collection_prompts)}")

    print("\n=== 使用示例 ===")
    print("在generate.py中使用:")
    print("from utils.preprocess_dataset import get_dataset_prompts")
    print("")
    print("prompt_list = get_dataset_prompts('manual')")
    print("prompt_list = get_dataset_prompts('gsm8k', 'data/gsm8k.jsonl', max_samples=10)")