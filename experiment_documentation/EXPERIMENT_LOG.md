# VSI-Bench Evaluation Experiment Log

## Project Overview
Multi-turn visual spatial reasoning evaluation using VSI-Bench dataset across three scene datasets: ARKitScenes, ScanNet, and ScanNet++.

---

## Experiment Timeline

### 2026-01-31: Initial Evaluations Completed
- **Status**: All baseline evaluation jobs submitted and completed
- **Methods**: Sequential and Video
- **Models**: Qwen3-VL-4B and Qwen3-VL-8B
- **Datasets**: ARKitScenes, ScanNet, ScanNet++

### 2026-02-01: Data Availability Investigation
- **Issue**: Question count discrepancies detected in results
- **Investigation**: Comprehensive mesh availability check across all datasets
- **Resolution**: Identified 2 missing ScanNet++ scenes (unavailable on server)

---

## Completed Experiments

### Sequential Method

#### Qwen3-VL-4B
| Dataset | Frames | Status | Result Path |
|---------|--------|--------|-------------|
| ARKitScenes | 4 | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/4B/*arkitscenes*` |
| ScanNet | 4 | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/4B/*scannet*` |
| ScanNet++ | 4 | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/4B/*scannetpp*` |

#### Qwen3-VL-8B
| Dataset | Frames | Status | Result Path |
|---------|--------|--------|-------------|
| ARKitScenes | 4 | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/8B/20260131_202543_sequential_Qwen3-VL-8B_arkitscenes_split1of1_3steps/` |
| ScanNet | 4 | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/8B/*scannet*` |
| ScanNet++ | 4 | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/8B/20260131_202543_sequential_Qwen3-VL-8B_scannetpp_split1of1_3steps/` |

### Video Method

#### Qwen3-VL-4B
| Dataset | Status | Result Path |
|---------|--------|-------------|
| ARKitScenes | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/4B/*arkitscenes*` |
| ScanNet | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/4B/*scannet*` |
| ScanNet++ | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/4B/*scannetpp*` |

#### Qwen3-VL-8B
| Dataset | Status | Result Path |
|---------|--------|-------------|
| ARKitScenes | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/8B/*arkitscenes*` |
| ScanNet | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/8B/*scannet*` |
| ScanNet++ | ✅ Complete | `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/8B/*scannetpp*` |

---

## Data Availability Status

### ARKitScenes
- **Required Scenes**: 150
- **Available Scenes**: 151 (100%)
- **Status**: ✅ **COMPLETE**
- **Location**: `/dss/mcmlscratch/06/di38riq/arkit_vsi`
- **VSI-Bench Questions**: 645 total

### ScanNet
- **Required Scenes**: 88
- **Available Scenes**: 88/312 with meshes (100% of required)
- **Status**: ✅ **COMPLETE**
- **Location**: `/dss/mcmlscratch/06/di38riq/scans`
- **VSI-Bench Questions**: 666 total
- **Note**: 312 scenes have meshes out of 400 total directories

### ScanNet++
- **Required Scenes**: 50 (from nvs_sem_val split)
- **Available Scenes**: 48 (96%)
- **Status**: ⚠️ **48/50 AVAILABLE**
- **Location**: `/dss/mcmlscratch/06/di38riq/data`
- **VSI-Bench Questions**: 561 total (536 available with current scenes)
- **Missing Scenes**:
  - `3864514494` - Directory exists but scans/ folder empty
  - `5942004064` - Does not exist
- **Missing Questions**: 25 (~4.5% of ScanNet++ questions)

#### Missing Scene Investigation
- Both scenes listed in validation split: `/dss/mcmlscratch/06/di38riq/splits/nvs_sem_val.txt`
- Download attempts return HTTP 404 from ScanNet++ server
- Tested on both v1 and v2 endpoints - not available on either
- Server-side issue confirmed: mesh download endpoint returning 404 for new downloads
- All currently available scenes were downloaded on 2026-01-30

---

## Question Type Coverage

### Expected Question Counts per Dataset
- **ARKitScenes**: 645 MCQ questions
- **ScanNet**: 666 MCQ questions  
- **ScanNet++**: 561 MCQ questions (536 with available scenes)

### Observed Results
| Method | Dataset | 4B Questions | 8B Questions | Status |
|--------|---------|--------------|--------------|--------|
| Sequential | ARKitScenes | TBD | TBD | ✅ Complete |
| Sequential | ScanNet | TBD | TBD | ✅ Complete |
| Sequential | ScanNet++ | 536 | 536 | ⚠️ 25 missing (unavailable scenes) |
| Video | ARKitScenes | TBD | TBD | ✅ Complete |
| Video | ScanNet | TBD | TBD | ✅ Complete |
| Video | ScanNet++ | 561 | 561 | ✅ Complete (uses alternative data) |

---

## Important File Paths

### Code & Configuration
- **Main codebase**: `/dss/dsshome1/06/di38riq/rl_multi_turn/`
- **VSI-Bench integration**: `/dss/dsshome1/06/di38riq/rl_multi_turn/src/`
- **Download scripts**: `/dss/dsshome1/06/di38riq/rl_multi_turn/data_download/`
- **ScanNet++ config**: `/dss/dsshome1/06/di38riq/rl_multi_turn/data_download/download_scannetpp.yml`

### Inference Scripts
- **Sequential evaluation**: `evaluation/sequential.py`
  - Multi-turn reasoning with 3D mesh rendering
  - Supports both HF and vLLM backends
  - Command: `python sequential.py --backend vllm --dataset DATASET --steps N`
  
- **Video evaluation**: `evaluation/video_baseline.py`
  - Video frame concatenation approach
  - Supports both HF and vLLM backends
  - Command: `python video_baseline.py --backend vllm --dataset DATASET --num-frames N`

- **Backend abstraction**: `utils/inference.py`
  - Unified API for HuggingFace and vLLM
  - Classes: `HFBackend`, `VLLMBackend`
  - Factory: `create_inference_backend(backend="vllm")`

### Job Submission Scripts (SLURM)

#### Video Baseline
- **Script**: `evaluation/submit_video_baseline.sbatch`
- **Partition**: `mcml-dgx-a100-40x8`
- **Resources**: 1 GPU, 8 CPUs, 64GB RAM, 45min walltime
- **Array jobs**: 1-4 (parallel processing of question splits)
- **Usage**:
  ```bash
  sbatch evaluation/submit_video_baseline.sbatch
  MODEL_ID="Qwen/Qwen3-VL-4B-Instruct" sbatch evaluation/submit_video_baseline.sbatch
  NUM_FRAMES=16 DATASET=scannet sbatch evaluation/submit_video_baseline.sbatch
  ```

#### Sequential Evaluation
- **Script**: `evaluation/submit_sequential.sbatch` (similar structure)
- **Resources**: 1 GPU, 8 CPUs, 64GB RAM, longer walltime (~10hrs)
- **Backend**: Uses vLLM by default for faster inference

#### Batch Submission
- **Script**: `evaluation/submit_all_combined.sh`
- **Submits**: All 12 experiments (2 models × 2 methods × 3 datasets)
- **Usage**: `bash evaluation/submit_all_combined.sh`

#### Key SLURM Parameters
```bash
#SBATCH --job-name=video_baseline
#SBATCH --partition=mcml-dgx-a100-40x8
#SBATCH --qos=mcml
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --array=1-4              # 4 parallel splits

# vLLM environment variables
export VLLM_USE_MODELSCOPE=False
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn  # CRITICAL for CUDA
```

### Dataset Locations
- **ARKitScenes**: `/dss/mcmlscratch/06/di38riq/arkit_vsi/`
- **ScanNet**: `/dss/mcmlscratch/06/di38riq/scans/`
- **ScanNet++**: `/dss/mcmlscratch/06/di38riq/data/`
- **ScanNet++ splits**: `/dss/mcmlscratch/06/di38riq/splits/`

### Experiment Results
- **All experiments**: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/`
- **Sequential 4B**: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/4B/`
- **Sequential 8B**: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Sequential/8B/`
- **Video 4B**: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/4B/`
- **Video 8B**: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir/experiment_logs/Video/8B/`

---

## Technical Details

### Models
- **Qwen3-VL-4B**: 4 billion parameter visual-language model (`Qwen/Qwen3-VL-4B-Instruct`)
- **Qwen3-VL-8B**: 8 billion parameter visual-language model (`Qwen/Qwen3-VL-8B-Instruct`)

### Inference Backend: vLLM Migration

#### Why vLLM?
Initially used HuggingFace Transformers, but migrated to vLLM for:
- **10-30x faster inference** through continuous batching
- **Better GPU utilization** with PagedAttention
- **Lower memory footprint** for large models
- **Production-grade serving** capabilities

#### HuggingFace vs vLLM Issues Encountered
1. **Multimodal input formatting**: Had to ensure identical prompt structure between HF and vLLM
2. **KV cache management**: Image tokens cost 200-400 tokens per image (15 images = 3000-6000 tokens)
3. **CUDA multiprocessing**: Required `export VLLM_WORKER_MULTIPROC_METHOD=spawn`
4. **Token generation**: vLLM uses `SamplingParams` instead of HF's `generate()` kwargs

**Documentation**: [VLLM_MIGRATION_PITFALLS.md](../VLLM_MIGRATION_PITFALLS.md)

#### Backend Configuration
- **HuggingFace**: Sequential processing, one request at a time
- **vLLM**: Continuous batching with PagedAttention
- **Backend abstraction**: `utils/inference.py` provides unified API

### Evaluation Methods
1. **Sequential**: Multi-turn reasoning with mesh rendering (4 frames per scene)
   - Uses custom rendering pipeline for 3D mesh navigation
   - Model decides camera movements based on spatial reasoning
   - Iterative refinement through multiple observation steps
   
2. **Video**: Concatenated video frames (4-16 frames)
   - Samples equally-spaced frames from scene recordings
   - All frames passed to model simultaneously
   - Tests temporal understanding across full scene

### VSI-Bench Dataset Statistics
- **Total unique scenes**: 238 (150 ARKitScenes + 88 ScanNet + 50 ScanNet++ - 2 missing)
- **Total MCQ questions**: 1872 (645 + 666 + 561)
- **Available questions**: 1847 (98.7%)

---

## Known Issues & Resolutions

### Issue 1: Video Backend Configuration (2026-01-31)
- **Problem**: Video baseline jobs failed with backend errors
- **Resolution**: Backend configuration fixed, jobs resubmitted successfully

### Issue 2: ScanNet++ Question Count Discrepancy (2026-02-01)
- **Problem**: Sequential method showing 536/561 questions for ScanNet++
- **Investigation**: Comprehensive mesh availability check performed
- **Root Cause**: 2 scenes (3864514494, 5942004064) missing from server
- **Status**: Confirmed as unavailable server-side, cannot be downloaded
- **Impact**: 4.5% of ScanNet++ questions unavailable for Sequential method
- **Workaround**: Video method has all 561 questions (uses alternative data source)

### Issue 3: Mesh Download Endpoint (2026-02-01)
- **Problem**: Cannot download new mesh files from ScanNet++ server
- **Status**: Server returning HTTP 404 for mesh files
- **Tested**: Both v1 and v2 endpoints return 404
- **Token validity**: Confirmed valid (can access split files)
- **Resolution**: Use existing 48 scenes downloaded on 2026-01-30

### Issue 4: vLLM CUDA Initialization Failure (2026-02-04)
- **Problem**: Trajectory test jobs (5470803-5470806) failed with `RuntimeError: CUDA driver initialization failed`
- **Root Cause**: vLLM v1 engine's internal multiprocessing wasn't using spawn method, causing CUDA initialization issues in subprocesses
- **Error Location**: `/dss/dsshome1/06/di38riq/miniconda3/envs/env/lib/python3.11/site-packages/vllm/v1/engine/core.py`
- **Solution Applied**:
  1. Added `export VLLM_WORKER_MULTIPROC_METHOD=spawn` environment variable to all job scripts
  2. Updated trajectory test scripts with complete environment setup (HF_HOME, TRANSFORMERS_CACHE, TORCH_HOME)
  3. Added proper conda initialization via full path: `source /dss/dsshome1/06/di38riq/miniconda3/etc/profile.d/conda.sh`
  4. Added GPU debug output for visibility: `nvidia-smi -L`
- **Files Modified**:
  - `evaluation/test_trajectory_4B_8frames.sh`
  - `evaluation/test_trajectory_4B_16frames.sh`
  - `evaluation/test_trajectory_8B_8frames.sh`
  - `evaluation/test_trajectory_8B_16frames.sh`
- **Status**: Fix applied, jobs resubmitted with updated scripts
- **Reference**: Working implementation found in `submit_full_vsi_bench.sh`

---

## Next Steps

### Immediate
- [ ] Extract and analyze MCQ results from all completed experiments
- [ ] Compare Sequential vs Video method performance
- [ ] Compare 4B vs 8B model performance
- [ ] Generate performance tables and visualizations

### Future
- [ ] Contact ScanNet++ dataset maintainers about missing scenes
- [ ] Re-run ScanNet++ Sequential evaluation if missing scenes become available
- [ ] Consider alternative 3D reconstruction methods for missing scenes

---

## Contact & Resources

### ScanNet++ Dataset
- **Website**: https://kaldir.vc.in.tum.de/scannetpp/
- **Download URL**: https://scannetpp.mlsg.cit.tum.de/scannetpp/download/
- **GitHub Issues**: ScanNet++ repository (for reporting missing scenes)

### Authentication
- **Token file**: `/dss/dsshome1/06/di38riq/rl_multi_turn/data_download/download_scannetpp.yml`
- **Token status**: Valid as of 2026-02-01

---

**Last Updated**: 2026-02-01 21:20 UTC  
**Maintained by**: Experiment tracking system
