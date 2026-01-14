# Batch Inference Implementation

## Overview

Created `render_point_cloud_qwen_angle_batched.py` - a batched version of the original pipeline that parallelizes question processing for significant speedup.

## Key Features

### 🚀 Batch Processing Strategy

Instead of running questions sequentially (Q1 fully, then Q2 fully, etc.), the batched version:

1. **Batched Phase** (Turns 0 to `MAX_BATCHED_TURNS-1`):
   - Processes Turn 0 for ALL questions together in one batch
   - Then Turn 1 for ALL questions together
   - Then Turn 2 for ALL questions together
   - Then Turn 3 for ALL questions together (if MAX_BATCHED_TURNS=4)

2. **Sequential Phase** (Remaining turns):
   - After the batched turns, completes each question individually
   - Q1: Turns 4,5,6,7,8
   - Q2: Turns 4,5,6,7,8
   - Q3: Turns 4,5,6,7,8

### ⚙️ Configuration

```python
MAX_BATCHED_TURNS = 4  # Default: batch first 4 turns
```

You can override this via command line:
```bash
--max-batched-turns 6  # Batch first 6 turns instead
```

## Usage

### Run batch evaluation:
```bash
python render_point_cloud_qwen_angle_batched.py --batch
```

### Run with custom batched turns:
```bash
python render_point_cloud_qwen_angle_batched.py --batch --max-batched-turns 6
```

### Run single question (uses original sequential pipeline):
```bash
python render_point_cloud_qwen_angle_batched.py --ply /path/to/mesh.ply --question "Your question"
```

### Continue from previous run:
```bash
python render_point_cloud_qwen_angle_batched.py --batch --continue recent
```

## Performance Benefits

### Expected Speedup

For N questions with M batched turns:
- **Without batching**: N × (M + remaining_turns) × inference_time_per_turn
- **With batching**: M × batch_inference_time + N × remaining_turns × inference_time

**Example** with 10 questions, MAX_BATCHED_TURNS=4:
- Old: 10 × 8 = 80 sequential inferences
- New: 4 batched (processing 10 at once) + 10 × 4 = 44 inference calls
- **Estimated speedup**: ~1.8x faster

Actual speedup depends on:
- Batch size (number of active questions)
- GPU memory capacity
- Image resolution and count per turn

### Memory Considerations

Batch processing increases GPU memory usage proportionally to batch size. Monitor VRAM usage:

```bash
nvidia-smi -l 1  # Monitor GPU memory in real-time
```

If you run out of memory, reduce:
- `MAX_BATCHED_TURNS` (fewer batched turns)
- Number of concurrent questions (split evaluation into chunks)
- `IMAGE_WH` resolution

## Implementation Details

### Core Components

1. **`QuestionState` class**: Maintains per-question state (mesh, camera poses, history)

2. **`initialize_question_state()`**: Sets up mesh, initial views, and camera for each question

3. **`run_batched_turn()`**: 
   - Prepares batch of messages (one per active question)
   - Uses Qwen3-VL's native batch processing with `process_vision_info()`
   - Single `.generate()` call processes all questions simultaneously
   - Updates all question states with results

4. **`complete_question_sequentially()`**: Finishes remaining turns for individual questions

5. **`main_vsi_bench_loop()`**: Orchestrates batched and sequential phases

### Turn Order Guarantee

The implementation ensures correct ordering:
- All questions experience turns in order: 0, 1, 2, 3, ...
- No question skips ahead while another is behind
- Camera states are properly maintained per question

### Early Termination

If a question completes early (model sets `done=true`):
- It's removed from active batch
- Remaining questions continue batched processing
- Completed question proceeds to result collection

## Differences from Original

| Feature | Original | Batched |
|---------|----------|---------|
| Processing | Sequential (one question at a time) | Batched (multiple questions per turn) |
| Experiment folder | `experiment_logs/TIMESTAMP/` | `experiment_logs/TIMESTAMP_batched/` |
| Inference calls | N × num_steps | MAX_BATCHED_TURNS + N × remaining_steps |
| Memory usage | Lower (1 question at a time) | Higher (multiple questions) |
| Speed | Baseline | ~1.5-2x faster |

## Monitoring

The script outputs detailed progress:

```
🔄 BATCHED TURN 0/3
[INFO] Processing 10 active questions in batch
[INFO] 🚀 Running batch generation for 10 questions...
[INFO] ⏱️  Batch generation completed in 45.2s (4.52s per question)

🔄 BATCHED TURN 1/3
[INFO] Processing 9 active questions in batch
[Q003] ✅ Final answer: C

📝 SEQUENTIAL PHASE: Completing remaining turns
[Q001] Continuing sequentially from turn 4...
[Q002] Continuing sequentially from turn 4...
```

## Troubleshooting

### Out of Memory Error
- Reduce `MAX_BATCHED_TURNS` to 2 or 3
- Lower `IMAGE_WH` resolution
- Process fewer questions at once

### Slower than expected
- Check GPU utilization: `nvidia-smi`
- Verify batch size is > 1 (check active questions)
- Ensure no CPU bottleneck in rendering

### Questions completing at different turns
- This is normal! Questions can finish early if model is confident
- Sequential phase handles remaining turns individually

## Testing

Test with small subset:
```python
# In the code, modify load_vsi_bench_questions():
questions = questions[:3]  # Test with first 3 questions only
```

Run and verify:
1. Turn 0-3 process in batch (3 questions at once)
2. Turns 4-8 process sequentially
3. All outputs saved correctly per question
4. Results JSON contains all 3 questions
