import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import textwrap
import os
import re
import argparse
import sys

# Set non-interactive backend for servers without display
matplotlib.use('Agg')

# Set up argument parser
parser = argparse.ArgumentParser(description='Visualize token metrics from JSON analysis file.')
parser.add_argument('json_path', type=str, help='Path to the JSON analysis file (required)')
args = parser.parse_args()

# 检查文件是否存在
if not os.path.exists(args.json_path):
    print(f"错误: 文件 '{args.json_path}' 不存在")
    sys.exit(1)

# 提取JSON文件名（不含扩展名）
json_filename = os.path.basename(args.json_path)
json_name = os.path.splitext(json_filename)[0]

# 创建输出目录
output_dir = os.path.join(os.path.dirname(args.json_path), f"{json_name}_plots")
os.makedirs(output_dir, exist_ok=True)

try:
    with open(args.json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
except Exception as e:
    print(f"加载JSON文件时出错: {e}")
    sys.exit(1)

# 设置图像路径
overall_img = os.path.join(output_dir, 'overall.png')
think_img = os.path.join(output_dir, 'think.png')
response_img = os.path.join(output_dir, 'response.png')

# 从JSON数据中提取模型信息用于标题
model_name = data['metadata']['model_name']
decoding_strategy = data['metadata']['decoding_strategy']
generation_timestamp = data['metadata']['generation_timestamp_utc'][:10]  # 只取日期部分


# Function to escape special characters in tokens
def escape_token(token):
    if token == ' ':
        return '[space]'
    return (
        token.replace('$', r'[$]')
        .replace('\\', r'\\')
        .replace('\n', r'\n')
        .replace('\"', r'\"')
    )


# Function to create and save a plot for given tokens and metrics
def plot_metrics(tokens, self_info, shannon_entropy, title, output_file, tokens_per_row=20):
    num_tokens = len(tokens)
    if num_tokens == 0:
        print(f"警告: '{title}'部分没有数据可绘制")
        return

    num_rows = (num_tokens + tokens_per_row - 1) // tokens_per_row
    fig_height = num_rows * 2.5
    fig_width = 15

    plt.rcParams['text.usetex'] = False  # Disable LaTeX rendering
    fig, axes = plt.subplots(num_rows, 1, figsize=(fig_width, fig_height), constrained_layout=True)
    if num_rows == 1:
        axes = [axes]
    elif num_rows == 0:  # 没有行的情况
        return

    # 添加主标题
    fig.suptitle(f"{title}\nModel: {model_name} | Strategy: {decoding_strategy} | Date: {generation_timestamp}",
                 fontsize=14, y=0.98)

    for row in range(num_rows):
        start_idx = row * tokens_per_row
        end_idx = min(start_idx + tokens_per_row, num_tokens)
        row_tokens = tokens[start_idx:end_idx]
        row_self_info = self_info[start_idx:end_idx]
        row_shannon = shannon_entropy[start_idx:end_idx]

        if not row_tokens:  # 检查是否有token
            continue

        ax = axes[row] if num_rows > 1 else axes
        x = np.arange(len(row_tokens))

        # Bar plot for self-information and Shannon entropy
        bar_width = 0.35
        ax.bar(x - bar_width / 2, row_self_info, bar_width, label='Self-Information (bits)', color='skyblue')
        ax.bar(x + bar_width / 2, row_shannon, bar_width, label='Shannon Entropy (bits)', color='salmon')

        # Set token labels, wrapping long tokens
        wrapped_tokens = [textwrap.fill(token, width=5) for token in row_tokens]
        ax.set_xticks(x)
        ax.set_xticklabels(wrapped_tokens, rotation=45, ha='right', fontsize=8)

        # Labels and title
        ax.set_ylabel('Bits')
        ax.set_title(f'Tokens {start_idx + 1} to {end_idx}')
        if row == 0:
            ax.legend()

        # 解决ylim警告问题
        max_value = max(
            max(row_self_info) if row_self_info else 0,
            max(row_shannon) if row_shannon else 0
        )
        min_value = min(
            min(row_self_info) if row_self_info else 0,
            min(row_shannon) if row_shannon else 0
        )

        # 确保ylim范围有效
        if max_value <= min_value:
            max_value = min_value + 1  # 防止相等的情况

        padding = (max_value - min_value) * 0.1  # 10%的填充
        ax.set_ylim(max(0, min_value - padding), max_value + padding)

    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"已保存: {output_file}")


# Extract data for overall
overall_data = data['analysis_breakdown']['overall']['per_token_analysis']
overall_tokens = [escape_token(entry['token']) for entry in overall_data]
overall_self_info = [entry['self_information_bits'] for entry in overall_data]
overall_shannon = [entry['step_shannon_entropy_bits'] for entry in overall_data]

# Extract data for response_part (assuming similar structure)
response_data = data['analysis_breakdown'].get('response_part', {}).get('per_token_analysis', [])
response_tokens = [escape_token(entry['token']) for entry in response_data]
response_self_info = [entry['self_information_bits'] for entry in response_data]
response_shannon = [entry['step_shannon_entropy_bits'] for entry in response_data]

# Extract data for think section (between <think> and </think>)
think_tokens = []
think_self_info = []
think_shannon = []
in_think = False
for entry in overall_data:
    if entry['token'] == '<think>':
        in_think = True
        continue
    if entry['token'] == '</think>':
        in_think = False
        continue
    if in_think:
        think_tokens.append(escape_token(entry['token']))
        think_self_info.append(entry['self_information_bits'])
        think_shannon.append(entry['step_shannon_entropy_bits'])

# Generate plots
plot_metrics(overall_tokens, overall_self_info, overall_shannon,
             'Overall Token Metrics', overall_img)
plot_metrics(think_tokens, think_self_info, think_shannon,
             'Think Section Metrics', think_img)
plot_metrics(response_tokens, response_self_info, response_shannon,
             'Response Part Metrics', response_img)

print(f"\n所有可视化结果已保存到: {output_dir}")