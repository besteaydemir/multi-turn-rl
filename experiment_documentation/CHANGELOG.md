# Experiment Changelog

All notable events and changes to the VSI-Bench evaluation experiments.

---

## [2026-02-02] - Full VSI-Bench Evaluation with All Question Types

### Summary
**Submitted comprehensive evaluation of entire VSI-Bench dataset (5,130 questions) across all 10 question types with both Sequential and Video pipelines.**

### Jobs Submitted
- **Job IDs**: 5466685-5466688 (4 jobs, 32 array tasks)
- **Time**: 2026-02-02 00:42 CET
- **Status**: Queued on `mcml-dgx-a100-40x8`

| Job ID | Name | Model | Pipeline | Questions | Splits | Time Limit |
|--------|------|-------|----------|-----------|--------|------------|
| 5466685 | seq_4B_all | Qwen3-VL-4B | Sequential | 4,512 | 8 | 12h |
| 5466686 | seq_8B_all | Qwen3-VL-8B | Sequential | 4,512 | 8 | 16h |
| 5466687 | video_4B_all | Qwen3-VL-4B | Video | 5,130 | 8 | 4h |
| 5466688 | video_8B_all | Qwen3-VL-8B | Video | 5,130 | 8 | 6h |

### Question Types Evaluated

**Sequential (9 types, 4,512 questions)**:
- 5 MCQ types: route_planning, object_rel_distance, object_rel_direction_{easy,medium,hard}
- 4 Numerical types: object_counting, object_abs_distance, object_size_estimation, room_size_estimation

**Video (10 types, 5,130 questions)**:
- Same 9 types as Sequential
- 1 Temporal type: obj_appearance_order (618 questions from ScanNet/ScanNet++)

### Implementation Changes

#### Added `dataset="all"` Support
Modified [utils/data.py](../utils/data.py) to load all 3 datasets simultaneously:
```python
# Now supports: dataset="all" or "combined"
questions = load_vsi_bench_questions(question_types=ALL_SEQUENTIAL_QUESTION_TYPES, dataset='all')
# Returns 5,130 questions from arkitscenes, scannet, scannetpp combined
```

#### Updated Argument Parsing
- [evaluation/sequential.py](../evaluation/sequential.py): Added `--dataset all` choice
- [evaluation/video_baseline.py](../evaluation/video_baseline.py): Added `--dataset all` choice
- Both scripts now automatically use correct mesh directory per question's source dataset

#### Simplified Submission Script
Created [evaluation/submit_all_vsi_bench.sh](../evaluation/submit_all_vsi_bench.sh):
- Reduced from 28 jobs to 4 jobs by using `--dataset all --question-types all`
- Increased splits from 4 to 8 to handle larger combined dataset
- Adjusted time limits based on model size

### Technical Details

**vLLM Configuration**:
- Sequential: `max_images=32` (supports multi-step exploration)
- Video: `max_images=48` (supports up to 48 video frames)
- Backend: vLLM with FP16, 85% GPU memory utilization

**Dataset Distribution**:
- ARKitScenes: 1,601 questions (31.2%)
- ScanNet: 2,071 questions (40.4%)
- ScanNet++: 1,458 questions (28.4%)

**Image Input Method**:
- Sequential: Multiple separate images (one per reasoning step)
- Video: Multiple frames as separate images (treated as video sequence)
- Both use same underlying vLLM multi-image support (`limit_mm_per_prompt`)

### Validation
✅ Tested with GPU using 4B model:
- Video baseline: Successfully loaded 5,130 questions, processed 2 test questions with 8 frames
- Sequential: Successfully loaded 4,512 questions, confirmed no temporal questions
- Verified dataset field correctly populated for multi-dataset mode

### Expected Results
**Output Locations**:
- Sequential 4B: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/4B/`
- Sequential 8B: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/8B/`
- Video 4B: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/4B/`
- Video 8B: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/8B/`

**Metrics**:
- MCQ types: Accuracy
- Numerical types: Mean Relative Accuracy (MRA)
- Temporal types: Accuracy (Kendall Tau correlation for ordering)

---

## [2026-02-01] - Data Availability Investigation

### Added
- Comprehensive mesh availability check across all three datasets
- Created `/tmp/check_meshes.py` for precise mesh verification
- Created experiment documentation folder
- Created EXPERIMENT_LOG.md and CHANGELOG.md

### Discovered
- ARKitScenes: 100% complete (151/150 required scenes)
- ScanNet: 100% complete (88/88 required scenes, 312 total with meshes)
- ScanNet++: 96% complete (48/50 required scenes)
- Missing ScanNet++ scenes: 3864514494, 5942004064
- Sequential ScanNet++ has 536/561 questions (25 missing)
- Video ScanNet++ has 561/561 questions (uses alternative data)

### Investigation Details
**Time**: 2026-02-01 18:00 - 21:20 UTC

**Steps Taken**:
1. Checked question counts in result files
2. Verified mesh file availability for all datasets
3. Attempted to download missing ScanNet++ scenes
4. Tested download script with existing scenes
5. Confirmed server-side unavailability of 2 scenes

**Findings**:
- ScanNet++ server returns HTTP 404 for mesh files
- Both missing scenes return 404 on v1 and v2 endpoints
- Download script works correctly (verified with test downloads)
- Token is valid (can access split files)
- Issue is server-side, not configuration problem

### Technical Notes
- All existing meshes were downloaded on 2026-01-30
- Mesh download endpoint appears to be broken as of 2026-02-01
- Split files still accessible
- Version parameter `?version=v1` is correct (minor version within v2 major release)

---

## [2026-01-31] - Evaluation Jobs Completed

### Completed
- All Sequential method evaluations (4B and 8B models)
- All Video method evaluations (4B and 8B models)
- Total: 12 evaluation jobs (2 methods × 2 models × 3 datasets)

### Job Submission Details
**Submission method**: SLURM batch jobs using sbatch
- Sequential jobs: `evaluation/submit_sequential.sbatch`
- Video jobs: `evaluation/submit_video_baseline.sbatch`
- Batch submission: `evaluation/submit_all_combined.sh`

**Resources per job**:
- Partition: `mcml-dgx-a100-40x8`
- GPU: 1x A100 (40GB)
- CPUs: 8
- Memory: 64GB
- Walltime: 45min (Video), 10hrs (Sequential)
- Array jobs: 4 splits per dataset (parallel processing)

**Backend**: vLLM (migrated from HuggingFace for 10-30x speedup)

### Fixed
- Video baseline backend configuration issue
- Jobs resubmitted and completed successfully
- vLLM multiprocessing issue (`VLLM_WORKER_MULTIPROC_METHOD=spawn`)

### Results Location
- Sequential 4B: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/4B/`
- Sequential 8B: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/8B/`
- Video 4B: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/4B/`
- Video 8B: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/8B/`

---

## [2026-01-21 to 2026-01-25] - vLLM Migration

### Changed
- Migrated inference backend from HuggingFace Transformers to vLLM
- Reason: 10-30x faster inference through continuous batching
- Created unified backend abstraction in `utils/inference.py`

### Issues Resolved
1. **Multimodal input formatting**: Fixed prompt template to match HF exactly
2. **CUDA multiprocessing**: Set `VLLM_WORKER_MULTIPROC_METHOD=spawn`
3. **KV cache**: Documented image token costs (200-400 tokens per image)
4. **API differences**: Mapped HF's `generate()` kwargs to vLLM's `SamplingParams`

### Documentation Added
- Created `VLLM_MIGRATION_PITFALLS.md` with comprehensive migration guide
- 300+ lines documenting common issues and solutions

---

## [2026-01-20] - Backend Comparison Tests

### Added
- Comparison scripts: `evaluation/compare_backends.py`
- Test results showing vLLM is 10-30x faster than HF
- Verified output consistency between backends

---

## [2026-01-30] - ScanNet++ Data Download

### Downloaded
- 48 ScanNet++ scenes from nvs_sem_val split
- Mesh files (mesh_aligned_0.05.ply) for all scenes
- DSLR image data for scenes

### Data Location
- `/dss/mcmlscratch/06/di38riq/data/`
- `/dss/mcmlscratch/06/di38riq/splits/`

---

## [Earlier] - Initial Setup

### Completed
- ARKitScenes dataset download and setup
- ScanNet dataset download and setup
- VSI-Bench integration
- Model setup (Qwen3-VL-4B and 8B)
- Baseline implementation (Sequential and Video methods)

---

**Changelog Format**: [YYYY-MM-DD] - Title
