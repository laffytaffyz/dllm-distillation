import os
import json
import glob
import re
import sys
import wandb
import torch
import torch.nn.functional as F


def flatten_metrics(results_dict):
    """
    Flatten the results dictionary to remove task prefixes and create clean
    metric names.

    Example:
    Input: {"mbpp": {"pass_at_1,create_test": 0.15,
                     "pass_at_1_stderr,create_test": 0.01}}
    Output: {"pass@1": 0.15, "pass@1_stderr": 0.01}
    """
    flattened = {}

    for task_name, task_results in results_dict.items():
        for metric_name, metric_value in task_results.items():
            # Skip non-numeric metadata like "alias"
            if not isinstance(metric_value, (int, float)):
                continue

            # Clean up metric names
            clean_name = metric_name

            # Remove task-specific suffixes like ",create_test"
            if "," in clean_name:
                clean_name = clean_name.split(",")[0]

            # Convert pass_at_1 to pass@1 for better readability
            clean_name = clean_name.replace("pass_at_", "pass@")

            # Store the metric
            flattened[clean_name] = metric_value

    return flattened


def upload_results_after_eval(wandb_project_name=None):
    """Hook function to upload results after evaluation completes."""
    # Only run on main process (rank 0) to avoid duplicate uploads
    try:
        import torch.distributed as dist
        if dist.is_initialized() and dist.get_rank() != 0:
            return
    except ImportError:
        # torch.distributed not available, assume single process
        pass

    # Additional check using environment variables
    rank = int(os.environ.get('RANK', 0))
    if rank != 0:
        return

    # Get output path from command line args
    output_path = None
    for i, arg in enumerate(sys.argv):
        if arg == "--output_path" and i + 1 < len(sys.argv):
            output_path = sys.argv[i + 1]
            break

    if output_path and os.path.exists(output_path):
        # Find result and sample files
        results_pattern = os.path.join(output_path, "*", "results_*.json")
        samples_pattern = os.path.join(output_path, "*", "samples_*.jsonl")
        results_files = glob.glob(results_pattern)
        samples_files = glob.glob(samples_pattern)

        if results_files:
            # Get the latest files based on timestamp in filename
            latest_results = get_latest_file(results_files)
            latest_samples = (
                get_latest_file(samples_files) if samples_files else None
            )

            # Extract timestamps and validate consistency
            results_timestamp = extract_timestamp_from_filename(latest_results)
            samples_timestamp = extract_timestamp_from_filename(latest_samples)

            assert results_timestamp == samples_timestamp, (
                f"Timestamp mismatch: results={results_timestamp}, "
                f"samples={samples_timestamp}"
            )

            # Load results file to extract metrics and metadata
            with open(latest_results, 'r') as f:
                results_data = json.load(f)

            # Extract config from command line for wandb
            config = extract_config_from_args()
            config["evaluation_timestamp"] = results_timestamp

            # Add metadata from results file to config (excluding "results")
            for key, value in results_data.items():
                if key != "results":
                    # Flatten nested dictionaries for better wandb display
                    if isinstance(value, dict):
                        for sub_key, sub_value in value.items():
                            config[f"{key}_{sub_key}"] = sub_value
                    else:
                        config[key] = value

            # Get wandb project name from parameter or config
            if wandb_project_name is None:
                wandb_project_name = config.get(
                    'wandb_project_name', 'dllm-eval'
                )
            config['wandb_project_name'] = wandb_project_name

            # Initialize wandb
            model_name = config.get('model', 'unknown').replace('/', '-')
            wandb.init(
                project=wandb_project_name,
                config=config,
                name=f"eval-{model_name}-{results_timestamp}"
            )

            # Log evaluation metrics with flattened structure
            metrics = flatten_metrics(results_data.get("results", {}))

            # Extract and add metadata parameters (like temperature)
            configs = results_data.get("configs", {})
            for task_name, task_config in configs.items():
                metadata = task_config.get("metadata", {})
                for param_name, param_value in metadata.items():
                    metrics[param_name] = param_value

            wandb.log(metrics)

            # Upload results as artifact
            artifact = wandb.Artifact("evaluation-results", type="results")
            artifact.add_file(latest_results)
            wandb.log_artifact(artifact)

            # Upload samples as artifact
            artifact = wandb.Artifact("evaluation-samples", type="samples")
            artifact.add_file(latest_samples)
            wandb.log_artifact(artifact)

            wandb.finish()


def get_latest_file(file_list):
    """Get the latest file based on timestamp in filename."""
    if not file_list:
        return None

    # Sort files by timestamp extracted from filename
    def extract_timestamp(filepath):
        # Extract timestamp like "2025-08-21T20-45-09.520028"
        match = re.search(
            r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)',
            filepath
        )
        return match.group(1) if match else ""

    return max(file_list, key=extract_timestamp)


def extract_timestamp_from_filename(filepath):
    """Extract timestamp from filename."""
    pattern = r'(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}\.\d+)'
    match = re.search(pattern, filepath)
    return match.group(1) if match else "unknown"


def extract_config_from_args():
    """Extract configuration from command line arguments."""
    config = {}

    args = sys.argv
    for i, arg in enumerate(args):
        if arg.startswith("--") and i + 1 < len(args):
            key = arg[2:]  # Remove --
            value = args[i + 1]
            if not value.startswith("--"):  # Ensure it's not another flag
                config[key] = value

    return config


def calculate_log_likelihood_from_logits(
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    ignore_index: int = -100
) -> tuple[float, int]:
    """
    Calculate log-likelihood from model logits and target token IDs.
    
    Args:
        logits: Model logits [batch, seq_len, vocab_size] or [seq_len, vocab_size]
        target_ids: Target token IDs [batch, seq_len] or [seq_len]
        ignore_index: Token ID to ignore in loss calculation
        
    Returns:
        Tuple of (total_log_likelihood, num_valid_tokens)
    """
    # Handle single sequence case
    if logits.dim() == 2:
        logits = logits.unsqueeze(0)
        target_ids = target_ids.unsqueeze(0)
    
    # Flatten for loss calculation
    logits_flat = logits.view(-1, logits.size(-1))
    targets_flat = target_ids.view(-1)
    
    # Calculate log probabilities
    log_probs = F.log_softmax(logits_flat, dim=-1)
    
    # Get log probability of target tokens
    target_log_probs = log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    
    # Mask out ignored tokens
    mask = (targets_flat != ignore_index)
    valid_log_probs = target_log_probs * mask.float()
    
    # Sum log probabilities (equivalent to product of probabilities in log space)
    total_log_likelihood = valid_log_probs.sum().item()
    num_valid_tokens = mask.sum().item()
    
    return total_log_likelihood, num_valid_tokens


def get_rolling_token_windows(
    token_list: list[int],
    prefix_token: int,
    max_seq_len: int,
    context_len: int = 1
) -> list[tuple[list[int], list[int]]]:
    """
    Generate rolling token windows for loglikelihood_rolling.
    
    Args:
        token_list: List of token IDs
        prefix_token: Prefix token ID (e.g., BOS/EOS)
        max_seq_len: Maximum sequence length
        context_len: Context length for rolling window
        
    Returns:
        List of (context, continuation) tuples
    """
    windows = []
    
    # Add prefix token
    tokens_with_prefix = [prefix_token] + token_list
    
    # Generate rolling windows
    for i in range(len(tokens_with_prefix)):
        # Context is everything up to position i
        context = tokens_with_prefix[:i+1]
        
        # Continuation is the next tokens up to max_seq_len
        continuation_start = i + 1
        continuation_end = min(continuation_start + max_seq_len - len(context), len(tokens_with_prefix))
        
        if continuation_start < len(tokens_with_prefix):
            continuation = tokens_with_prefix[continuation_start:continuation_end]
            if len(continuation) > 0:
                windows.append((context, continuation))
    
    return windows
