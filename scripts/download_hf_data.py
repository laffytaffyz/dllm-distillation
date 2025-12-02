import argparse
import os
from datasets import load_dataset
from huggingface_hub import snapshot_download


"""
Options:
1. Streaming mode (recommended - no download): 
   python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --mode streaming

2. Download with reduced workers (safer):
   python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data --max_workers 4

3. Use hf_transfer for faster downloads:
   HF_HUB_ENABLE_HF_TRANSFER=1 python3 scripts/download_hf_data.py --repo_id fredzzp/fine_code --local_dir ./data
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo_id", type=str, default="HuggingFaceFW/fineweb")
    parser.add_argument("--local_dir", type=str, default="./fineweb/")
    parser.add_argument("--allow_patterns", type=str, default=None)
    parser.add_argument("--max_workers", type=int, default=4, 
                       help="Number of parallel workers (default: 4, reduce if getting SSL errors)")
    parser.add_argument("--mode", type=str, default="download", choices=["download", "streaming"],
                       help="'download' to download files, 'streaming' to load on-the-fly without downloading")
    args = parser.parse_args()

    repo_id = args.repo_id
    local_dir = args.local_dir
    allow_patterns = args.allow_patterns
    max_workers = args.max_workers
    mode = args.mode

    if mode == "streaming":
        # Streaming mode - loads data on-the-fly, no local download
        print(f"Loading dataset {repo_id} in streaming mode (no local download)...")
        try:
            dataset = load_dataset(repo_id, streaming=True, split="train")
            print(f"Dataset loaded successfully! You can iterate over it without downloading.")
            print("Example usage:")
            print("  for sample in dataset:")
            print("      # Process sample")
            print("      break  # Just to show it works")
            
            # Show first sample
            for i, sample in enumerate(dataset):
                print(f"\nFirst sample keys: {list(sample.keys())}")
                if i == 0:
                    break
        except Exception as e:
            print(f"Error loading dataset: {e}")
            print("Falling back to download mode...")
            mode = "download"
    
    if mode == "download":
        # Download mode with reduced workers and better error handling
        print(f"Downloading dataset {repo_id} to {local_dir}...")
        print(f"Using {max_workers} workers (reduced from 100 to prevent connection issues)")
        
        try:
            folder = snapshot_download(
                repo_id,
                repo_type="dataset",
                local_dir=local_dir,
                allow_patterns=allow_patterns,
                max_workers=max_workers,  # Reduced from 100
                resume_download=True,  # Resume if interrupted
            )
            print(f"Download complete! Files saved to: {folder}")
        except Exception as e:
            print(f"Download failed: {e}")
            print("\nTips to fix:")
            print("1. Try reducing --max_workers further (e.g., --max_workers 2)")
            print("2. Use streaming mode: --mode streaming")
            print("3. Run in screen/tmux: screen -S download, then run the script")
            print("4. Use hf_transfer: HF_HUB_ENABLE_HF_TRANSFER=1 python3 ...")
            raise
