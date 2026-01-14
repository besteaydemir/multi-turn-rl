# Batch Inference Execution Order & Debugging Guide

## 📋 Execution Order (Detailed)

### Phase 1: Initialization (happens ONCE at start)
```
1. Load VSI-Bench dataset and filter route_planning questions
2. Create experiment folder: experiment_logs/YYYYMMDD_HHMMSS_batched/
3. For EACH question:
   - Create folder: experiment_logs/.../q001/, q002/, etc.
   - Load mesh (.ply file)
   - Render 4 candidate views (0°, 90°, 180°, 270°)
   - Select best initial view
   - Save: render_00.png, cam_pose_00.npy, initial_view_selection.json
```

**Key Point**: ALL question folders are created upfront before any inference!

### Phase 2: Batched Turns (turns 0 to MAX_BATCHED_TURNS-1)

For MAX_BATCHED_TURNS=4, BATCH_SIZE=2, with 10 questions:

```
Turn 0:
  ├─ Mini-batch 1: [Q1, Q2]
  │   ├─ Prepare messages with Q1's images + Q2's images
  │   ├─ Single batched inference → 2 outputs
  │   ├─ Q1: Save step_01/qwen_raw_output.txt, render_01.png
  │   └─ Q2: Save step_01/qwen_raw_output.txt, render_01.png
  ├─ Mini-batch 2: [Q3, Q4]
  │   └─ (same process)
  ├─ Mini-batch 3: [Q5, Q6]
  ├─ Mini-batch 4: [Q7, Q8]
  └─ Mini-batch 5: [Q9, Q10]

Turn 1:
  ├─ Mini-batch 1: [Q1, Q2]
  │   ├─ Q1: Save step_02/qwen_raw_output.txt, render_02.png
  │   └─ Q2: Save step_02/qwen_raw_output.txt, render_02.png
  ├─ Mini-batch 2: [Q3, Q4]
  ... (continues)

Turn 2:
  └─ (same mini-batch pattern, saves step_03/)

Turn 3:
  └─ (same mini-batch pattern, saves step_04/)
```

### Phase 3: Sequential Turns (remaining turns)

```
Q1: Turns 4, 5, 6, 7, 8
  ├─ Turn 4: Single inference → Save step_05/qwen_raw_output.txt, render_05.png
  ├─ Turn 5: Single inference → Save step_06/qwen_raw_output.txt, render_06.png
  ... (until turn 8 or early termination)

Q2: Turns 4, 5, 6, 7, 8
  └─ (same process)

Q3: Turns 4, 5, 6, 7, 8
  └─ (same process)

... (all questions complete sequentially)
```

### Phase 4: Finalization

```
For each question:
  ├─ Calculate total time = batch_time_accumulated + sequential_time
  ├─ Check correctness (model_answer vs ground_truth)
  ├─ Save to results.json and results.csv
  └─ Print summary
```

## 🐛 Bug Fixes Applied

### 1. Empty Qwen Outputs (CRITICAL BUG)
**Problem**: All `qwen_raw_output.txt` files were empty

**Debugging Added**:
- Print output_text type and length
- Show if output is empty before saving
- Print first 200 chars of non-empty outputs
- Log input text lengths
- Debug tokenization and generation process

**Look for these in logs**:
```
[DEBUG] Output text length: 0
[DEBUG] Output is empty: True
⚠️  WARNING: Empty output from model!
```

### 2. Timing Tracking (Per Question)
**Problem**: Batched time was not distributed properly to each question

**Fixed**:
- Each question tracks `batch_time_accumulated` (its share of batch inference)
- Each question tracks `sequential_time` (its individual sequential inference)
- Total time = batch_time_accumulated + sequential_time

**CSV now shows**:
```csv
question,scene_id,gt_answer,model_answer,time_seconds,num_steps,timestamp
"...",42049,B,B,45.3,8,251215-230102
```

Where `time_seconds = 45.3` might be:
- Batched time: 4 turns × 5s/question = 20s
- Sequential time: 4 turns × 6.3s = 25.3s
- Total: 45.3s

### 3. Extensive Debugging Output

**New debug messages show**:

```
[DEBUG] ========== BATCH GENERATION START ==========
[DEBUG] Number of questions in batch: 2
[DEBUG]   Q001: 3 images in history
[DEBUG]   Q002: 3 images in history
[DEBUG] Applying chat template to 2 messages...
[DEBUG] Chat templates applied. Text lengths: [2543, 2501]
[DEBUG] Processing vision info for batch...
[DEBUG] Vision info processed. Images: 6, Videos: 0
[DEBUG] Tokenizing inputs...
[DEBUG] Inputs prepared. Shapes: ['input_ids: torch.Size([2, 1543])', ...]
[DEBUG] Starting model.generate()...
[DEBUG] Generation complete. Output shape: torch.Size([2, 256])
[DEBUG] Trimmed output lengths: [128, 134]
[DEBUG] Decoded 2 outputs. Lengths: [512, 503]
[DEBUG] ========== BATCH GENERATION END ==========

[DEBUG] ===== Q001 Turn 0 Processing =====
[DEBUG] Output text type: <class 'str'>
[DEBUG] Output text length: 512
[DEBUG] Output is empty: False
[Q001 Turn 0] Output preview (first 200 chars):
Looking at the current view, I can see a kitchen area with cabinets and ...
```

## 🔍 How to Debug Empty Outputs

If outputs are still empty, check these in order:

1. **Model Loading**:
   ```
   [INFO] Loading Qwen3 model on device: cuda
   [INFO] Qwen3 model loaded.
   ```

2. **Tokenization**:
   ```
   [DEBUG] Inputs prepared. Shapes: [...]
   ```
   - If shapes are wrong or missing, tokenization failed

3. **Generation**:
   ```
   [DEBUG] Generation complete. Output shape: torch.Size([2, 256])
   ```
   - If output shape shows only padding length (e.g., [2, 1]), model didn't generate

4. **Decoding**:
   ```
   [DEBUG] Decoded 2 outputs. Lengths: [512, 503]
   ```
   - If lengths are 0, decoding issue

5. **Check Error Logs**:
   - Look in `logs/vsi_bench_batched_*.err` for any CUDA errors or warnings

## 📊 Expected Log Flow

**Good run**:
```
[INFO] 🔧 Initializing all question states...
[Q001] 📂 Loading mesh: ...
[Q001] ✅ Initialized (best view: 90°)
[Q002] 📂 Loading mesh: ...
[Q002] ✅ Initialized (best view: 0°)
...

🔄 BATCHED TURN 0/3
[INFO] Processing 10 active questions in mini-batches of 2
[INFO] 🔄 Mini-batch [1-2]: Questions [1, 2]
[DEBUG] ========== BATCH GENERATION START ==========
[DEBUG] Number of questions in batch: 2
[INFO] ⏱️  Batch generation completed in 12.5s (6.25s per question)
[Q001 Turn 0] Output preview (first 200 chars): ...
[Q002 Turn 0] Output preview (first 200 chars): ...
```

**Bad run (empty outputs)**:
```
[DEBUG] ========== BATCH GENERATION START ==========
[DEBUG] Decoded 2 outputs. Lengths: [0, 0]  ← PROBLEM HERE
⚠️  WARNING: Empty output from model!
⚠️  WARNING: Empty output from model!
⚠️  No valid movement, using fallback
```

## 🚀 Testing

Run a small test first:
```python
# In load_vsi_bench_questions(), add:
questions = questions[:2]  # Test with just 2 questions
```

Then:
```bash
sbatch run_evaluation_batched.sh
tail -f logs/vsi_bench_batched_*.out
```

Watch for the debug messages to see where it fails!
