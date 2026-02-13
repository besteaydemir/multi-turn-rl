# VSI-Bench Performance Analysis Summary

## Generated Files

### 1. Performance Comparison Plots
- **overall_accuracy_comparison.png**: Overall accuracy across frame counts for 4B/8B Sequential/Video
- **overall_mra_comparison.png**: Overall MRA scores across frame counts for 4B/8B Sequential/Video
- **category_accuracy_comparison.png**: Per-category accuracy breakdown (9 question types)
- **category_mra_comparison.png**: Per-category MRA breakdown (9 question types)

### 2. Aggregated CSV Files (16 total)
- Sequential: 4B/8B × 4/8/16/32 frames (8 files)
- Video: 4B/8B × 4/8/16/32 frames (8 files)
- Each CSV contains deduplicated results with unique (scene_id, question) pairs

### 3. Summary Statistics

#### Sequential Evaluation Results (Current Status)

**4B Model:**
| Frames | Questions | Completion | Accuracy | MRA   |
|--------|-----------|------------|----------|-------|
| 4f     | 4457      | 98.8%      | 7.83%    | 41.10%|
| 8f     | 4457      | 98.8%      | 7.52%    | 39.92%|
| 16f    | 4457      | 98.8%      | 6.91%    | 38.69%|
| 32f    | 3929      | 87.1%      | 6.16%    | 37.72%|

**8B Model:**
| Frames | Questions | Completion | Accuracy | MRA   |
|--------|-----------|------------|----------|-------|
| 4f     | 4457      | 98.8%      | 3.19%    | 36.81%|
| 8f     | 4457      | 98.8%      | 4.08%    | 35.10%|
| 16f    | 4457      | 98.8%      | 5.32%    | 36.63%|
| 32f    | 2162      | 47.9%      | 3.93%    | 32.14%|

#### Video Evaluation Results
- All configurations complete (5130/5130 questions)
- MRA scores not computed in current CSV exports (needs recomputation)

## Key Findings

### 1. Duplicate Question Handling
- **Finding**: Multiple runs of the same question produce **identical outputs**
- **Evidence**: Tested 286 duplicate instances in 4B 16f - all had same responses
- **Implication**: Model inference is deterministic; duplicates can be safely removed
- **Method**: Used (scene_id, question) tuples as unique identifiers

### 2. Completion Status
- **Persistent Issue**: 55 questions consistently missing across 6 configurations (4B/8B × 4/8/16f)
- **Major Gaps**: 
  - 4B 32f: 583 questions remaining (87.1% complete)
  - 8B 32f: 2,350 questions remaining (47.9% complete)
- **Attempted Fix**: Resume jobs submitted but failed due to **disk quota exceeded**

### 3. Performance Trends (Preliminary - Sequential only)
- **4B Model**: 
  - Best accuracy at 4 frames (7.83%)
  - Decreasing trend with more frames
  - MRA remains relatively stable (37-41%)
  
- **8B Model**:
  - Peak accuracy at 16 frames (5.32%)
  - Lower than 4B model (unexpected)
  - MRA range: 32-37%

### 4. Storage Issue
- **Critical**: Resume jobs (Feb 8, 2026) failed with "Disk quota exceeded"
- **Location**: `/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir`
- **Impact**: Cannot complete remaining 3,263 questions until space freed

## Next Steps

1. **Immediate**: Free up disk space on storage system
2. **Resume**: Re-submit remaining evaluation jobs (4B 32f: 2 splits, 8B 32f: 12 splits)
3. **Video Metrics**: Recompute MRA scores for Video evaluation CSVs
4. **Analysis**: Generate final plots after Sequential evaluation completes

## Technical Notes

- **Question Types**: 10 types found (9 sequential + obj_appearance_order)
- **Total Questions**: 4,512 per configuration (Sequential), 5,130 (Video)
- **Deduplication**: CSV aggregation removed 0-286 duplicates per configuration
- **Determinism**: Confirmed via duplicate output analysis

---
Generated: February 8, 2026
Location: `/dss/dsshome1/06/di38riq/rl_multi_turn/analysis/`
