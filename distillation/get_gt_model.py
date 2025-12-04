import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

# 1. Load Model (Teacher)
model_id = "fredzzp/open-dcoder-0.5B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    torch_dtype=torch.bfloat16
).cuda().eval()

# 2. Setup Input (Fully Masked Sequence)
prompt = "def quicksort(arr):"
input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
seq_len = 128  # Target total length
batch_size = input_ids.shape[0]

# Fill the rest of the sequence with [MASK] tokens
mask_token_id = tokenizer.mask_token_id
pad_len = seq_len - input_ids.shape[1]
masked_suffix = torch.full((batch_size, pad_len), mask_token_id, device="cuda")
x_t = torch.cat([input_ids, masked_suffix], dim=1)

# 3. The Unrolled Diffusion Loop
num_steps = 64  # Standard diffusion steps
collected_logits = []

print(f"Starting sampling ({num_steps} steps)...")

for step in range(num_steps):
    # Calculate current time 't' (from 1.0 down to 0.0)
    t = 1.0 - (step / num_steps)
    
    with torch.no_grad():
        # A. Forward Pass
        # We explicitly ask for logits. The model sees the current x_t.
        outputs = model(x_t)
        logits = outputs.logits # Shape: [Batch, Seq_Len, Vocab]
        
        # --- SAVE YOUR LOGITS HERE ---
        # Only save the 'response' part (ignore the fixed prompt)
        response_logits = logits[:, input_ids.shape[1]:, :]
        collected_logits.append(response_logits.cpu()) 
        
        # B. Prediction (Denoising)
        # The model predicts the fully clean x_0 at every step
        pred_ids = logits.argmax(dim=-1)
        
        # C. Re-Masking Strategy (The "Scheduler")
        # In discrete diffusion, we don't just keep the prediction.
        # We keep the "confident" tokens and re-mask the "uncertain" ones based on time t.
        
        # 1. Get confidence scores
        probs = F.softmax(logits, dim=-1)
        confidence = probs.max(dim=-1).values
        
        # 2. Determine how many tokens to keep (Linear schedule)
        # At t=1.0 (start), keep 0%. At t=0.0 (end), keep 100%.
        # We only mask the response part, not the prompt!
        num_to_keep = int((1 - t) * pad_len)
        
        # 3. Create the new mask for the response
        # Find indices of the most confident predictions in the response
        resp_conf = confidence[:, input_ids.shape[1]:]
        # Get top-k confident indices
        topk_indices = resp_conf.topk(num_to_keep, dim=1).indices
        
        # 4. Construct next x_t
        # Start with all MASKs
        new_suffix = torch.full((batch_size, pad_len), mask_token_id, device="cuda")
        # Fill in the high-confidence tokens we predicted
        gathered_preds = pred_ids[:, input_ids.shape[1]:]
        new_suffix.scatter_(1, topk_indices, gathered_preds.gather(1, topk_indices))
        
        # Update x_t for next step
        x_t = torch.cat([input_ids, new_suffix], dim=1)

print(f"Done. Collected {len(collected_logits)} steps of logits.")