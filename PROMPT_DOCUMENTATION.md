# VSI-Bench Prompt Documentation

This document explains the prompts used for each question type in both **Sequential** and **Video Baseline** evaluation pipelines, showing the verbatim prompt template and what changes for each question type.

---

## Overview

### Sequential Evaluation (`sequential.py`)
- **Mode**: Embodied AI agent exploring a 3D room through interactive movement
- **Key Features**:
  - Agent makes decisions at each step (explore more or answer)
  - Receives rendered images from mesh as it moves
  - Can rotate, translate, and strafe in the environment
  - Maximum of N exploration steps before forced final answer
  - Full camera pose and trajectory tracking

### Video Baseline (`video_baseline.py`)
- **Mode**: Passive video viewer looking at pre-recorded video frames
- **Key Features**:
  - Fixed set of equally-spaced frames from actual video
  - No agent-controlled exploration
  - Single pass through frames
  - Simpler prompt structure
  - Designed for comparison with embodied agent

---

## Sequential Pipeline Prompt Template

### Main Prompt Structure (from `build_instruction_text()`)

```
You are an embodied agent exploring a 3D room to answer a spatial reasoning question.

---
{task_description}

**Question:** {question}
{options_text}{direction_note}
{bbox_text}
---
## Your Exploration Journey:

Below are ALL the images you have captured as you moved through the room.
**IMPORTANT: Focus on the MOST RECENT image (bottom of list) first, then connect it to your journey.**
- Image 0 = your initial starting position
- Image 1+ = views after each of your movements

---
**OUTPUT FORMAT:**

Think step-by-step BEFORE providing JSON:

1. **Current observation:** What do I see in the most recent image?
2. **Objects found so far:** Which objects from the question have I identified?
3. **Missing information:** What objects/perspectives do I still need to see?
4. **Current opinion on answer:** Based on what I've seen, what's my best guess?
5. **Next action:** Should I explore more or answer now?

**DECISION RULE:**
- If you have STRONG confidence in your answer → Set "done": true, provide "answer", NO movement needed
- If you need more exploration → Provide movement commands, set "done": false

Then output JSON:
```json
{
  "rotation_angle_degrees": <-90 to 90, or 0 if done=true>,
  "forward_meters": <-1.0 to 1.0, or 0 if done=true>,
  "left_meters": <-0.5 to 0.5, or 0 if done=true>,
  "z_delta_meters": <-0.3 to 0.3, or 0 if done=true>,
  "answer": <"A"/"B"/"C"/"D" for MCQ or number for numerical, or null>,
  "done": <true/false>
}
```

---
{important_note}

{movement_instruction}

{movement_history_text}
```

---

## Question Types and Prompt Variables

### 1. **Route Planning**

**Variable Values:**
```
task_description: |
  **TASK: Route Planning**
  You need to understand a navigation route between objects in the room.
  - First, locate the STARTING object mentioned
  - Identify the DESTINATION object 
  - Trace the route and determine which turns are needed

answer_hint: Choose the correct sequence of turns

direction_note: (empty)
```

**Sequential Characteristics**:
- Agent can physically explore the route by moving through the space
- Direction reference frame is the agent's current facing direction
- Agent sees the route from multiple angles as they move
- Movement history shows how agent reached current position

#### Video Baseline Prompt (Verbatim):
```
You are viewing {num_frames} frames from a video walkthrough of an indoor scene.

Analyze the navigation route shown in the video to answer.

**Question:** {question}

**Answer Options:**
{choices_text}

**Instructions:**
1. Observe the objects and spatial layout across all frames
2. Determine the answer
3. Respond with a JSON object containing brief reasoning and your answer

**Response format:**
{"reasoning": "<brief explanation>", "answer": "<A, B, C, or D>"}

Be concise. Avoid repeating yourself.
```

---

### 2. **Object Relative Distance**

**Variable Values:**
```
task_description: |
  **TASK: Relative Distance Comparison**
  You need to determine which object is CLOSEST to a reference object.
  - Locate the reference object mentioned in the question
  - Find ALL candidate objects listed in the options
  - Compare their distances to the reference object

answer_hint: Choose the object that is closest (A, B, C, or D)

direction_note: (empty)
```

**Sequential Characteristics**:
- Agent can move closer to objects to better assess relative distances
- Uses room bounding box to provide scale reference
- Multiple perspectives help confirm object positions
- Can physically verify "closest" by moving toward candidates

#### Video Baseline Prompt (Verbatim):
```
You are viewing {num_frames} frames from a video walkthrough of an indoor scene.

Compare the distances between objects visible in the video.

**Question:** {question}

**Answer Options:**
{choices_text}

**Instructions:**
1. Observe the objects and spatial layout across all frames
2. Determine the answer
3. Respond with a JSON object containing brief reasoning and your answer

**Response format:**
{"reasoning": "<brief explanation>", "answer": "<A, B, C, or D>"}

Be concise. Avoid repeating yourself.
```

---

### 3. **Object Relative Direction** (3 Difficulty Levels)

**Variable Values (Same for All Difficulties):**
```
task_description: |
  **TASK: Relative Direction**
  You need to determine the direction of one object relative to another FROM A SPECIFIC VIEWPOINT.
  - First, locate WHERE you would be standing
  - Then, determine which direction you would be FACING
  - From that perspective, determine where the TARGET object is

direction_note: |
  **Direction Reference:**
  - Imagine standing at the viewpoint and facing in the specified direction
  - Front-left: 45° to your left, Front-right: 45° to your right
  - Back-left: 45° behind you to the left, Back-right: 45° behind you to the right
```

**Variable Values by Difficulty:**

| Difficulty | answer_hint |
|-----------|-------------|
| **Easy** | Choose: left or right |
| **Medium** | Choose: left, right, or back |
| **Hard** | Choose: front-left, front-right, back-left, or back-right |

**Sequential Characteristics**:
- Agent can move to the viewpoint and face the specified direction
- Can rotate 360° to verify object positions relative to agent
- Multiple images show same object from specified viewpoint
- Harder difficulty levels require more precise spatial reasoning

#### Video Baseline Prompt (Verbatim):
```
You are viewing {num_frames} frames from a video walkthrough of an indoor scene.

Determine the relative direction of objects from the specified viewpoint.

**Question:** {question}

**Answer Options:**
{choices_text}

**Instructions:**
1. Observe the objects and spatial layout across all frames
2. Determine the answer
3. Respond with a JSON object containing brief reasoning and your answer

**Response format:**
{"reasoning": "<brief explanation>", "answer": "<A, B, C, or D>"}

Be concise. Avoid repeating yourself.
```

---

### 4. **Object Counting**

**Variable Values:**
```
task_description: |
  **TASK: Object Counting**
  You need to count how many instances of a specific object category appear in the room.
  - Systematically scan the entire room to find ALL instances of the target object
  - Be careful to distinguish between similar objects (e.g., chair vs stool)
  - Count each distinct instance only once (avoid double-counting from different angles)
  - Some objects may be partially visible or in different areas of the room

answer_hint: Provide a single integer number (e.g., 3)

direction_note: (empty)
```

**Sequential Characteristics**:
- Agent is explicitly told to systematically scan the room
- Can explore unmapped areas to find all instances
- Movement history helps track which areas were visited
- Guidance on avoiding double-counting from different viewing angles
- Up to 15 exploration steps to thoroughly search

#### Video Baseline Prompt (Verbatim):
```
You are viewing {num_frames} frames from a video walkthrough of an indoor scene.

Count how many instances of the target object appear across all frames. Be careful not to double-count the same object.

**Question:** {question}

**Provide your answer as an integer (e.g., 3).**

**Instructions:**
1. Observe the objects and spatial layout across all frames
2. Determine the answer
3. Respond with a JSON object containing brief reasoning and your answer

**Response format:**
{"reasoning": "<brief explanation>", "answer": <integer>}

Be concise. Avoid repeating yourself.
```

---

### 5. **Absolute Distance Measurement**

**Variable Values:**
```
task_description: |
  **TASK: Absolute Distance Measurement**
  You need to estimate the distance between two specific objects in meters.
  - Locate BOTH objects mentioned in the question
  - Consider the CLOSEST points of each object (not centers)
  - Use visual cues like floor tiles, furniture sizes, or room dimensions for scale
  - Typical room objects for reference: chair height ~0.9m, table height ~0.7m, door height ~2m

answer_hint: Provide distance in meters with one decimal place (e.g., 1.5)

direction_note: (empty)
```

**Sequential Characteristics**:
- Room bounding box dimensions provided for absolute scale
- Agent can move to measure distance by walking between objects
- Multiple viewing angles help establish spatial relationships
- Guidance on using furniture as scale references
- Precise answer format required (decimal place)

#### Video Baseline Prompt (Verbatim):
```
You are viewing {num_frames} frames from a video walkthrough of an indoor scene.

Estimate the distance between the two objects in meters using visual cues for scale.

**Question:** {question}

**Provide your answer as a number with one decimal place (e.g., 1.5).**

**Instructions:**
1. Observe the objects and spatial layout across all frames
2. Determine the answer
3. Respond with a JSON object containing brief reasoning and your answer

**Response format:**
{"reasoning": "<brief explanation>", "answer": <number>}

Be concise. Avoid repeating yourself.
```

---

### 6. **Object Size Estimation**

**Variable Values:**
```
task_description: |
  **TASK: Object Size Estimation**
  You need to estimate the size of a specific dimension of an object in centimeters.
  - Locate the target object clearly in your views
  - Identify which dimension to measure (length/width/height as specified)
  - Use relative scale from surroundings (doorways ~80cm wide, ceiling height ~240cm)
  - Common object sizes: dining table 70-90cm tall, chair seat 45cm high, sofa 80cm tall

answer_hint: Provide size in centimeters as an integer (e.g., 75)

direction_note: (empty)
```

**Sequential Characteristics**:
- Multiple viewing angles to see object dimensions
- Can move closer/farther for better dimensional assessment
- Detailed reference objects provided
- Answer format: integer centimeters

#### Video Baseline Prompt (Verbatim):
```
You are viewing {num_frames} frames from a video walkthrough of an indoor scene.

Estimate the size of the specified object dimension in centimeters using surroundings for scale reference.

**Question:** {question}

**Provide your answer in centimeters as an integer (e.g., 75).**

**Instructions:**
1. Observe the objects and spatial layout across all frames
2. Determine the answer
3. Respond with a JSON object containing brief reasoning and your answer

**Response format:**
{"reasoning": "<brief explanation>", "answer": <integer>}

Be concise. Avoid repeating yourself.
```

---

### 7. **Room Size Estimation**

**Variable Values:**
```
task_description: |
  **TASK: Room Size Estimation**
  You need to estimate the total floor area of the room in square meters.
  - Explore to understand the full extent of the room boundaries
  - Estimate length and width dimensions separately
  - Calculate area = length × width
  - For combined spaces, sum the areas of connected regions
  - Reference: Small bedroom ~10-15m², Living room ~20-30m², Large open space ~40-60m²

answer_hint: Provide area in square meters with one decimal place (e.g., 25.5)

direction_note: (empty)
```

**Sequential Characteristics**:
- Explicit instruction to explore room boundaries
- Walk through the space to measure extent
- Up to 15 steps allows thorough room mapping
- Guidance on calculating area and handling combined spaces
- Reference room sizes provided

#### Video Baseline Prompt (Verbatim):
```
You are viewing {num_frames} frames from a video walkthrough of an indoor scene.

Estimate the total floor area of the room in square meters based on the video walkthrough.

**Question:** {question}

**Provide your answer as a number with one decimal place (e.g., 1.5).**

**Instructions:**
1. Observe the objects and spatial layout across all frames
2. Determine the answer
3. Respond with a JSON object containing brief reasoning and your answer

**Response format:**
{"reasoning": "<brief explanation>", "answer": <number>}

Be concise. Avoid repeating yourself.
```

---

## General Structural Differences

### Sequential Prompts Include:

1. **Movement Controls**: Detailed instructions on rotation, translation, strafing, vertical movement
2. **Exploration Strategy**: Guidance on how to systematically explore
3. **Step Counter**: Current step number and remaining steps (e.g., "Step 5/15")
4. **Image Sequence**: Labels for each image showing position and step number
5. **Movement History**: Log of previous movements and resulting positions
6. **JSON Output Format**:
   ```json
   {
     "rotation_angle_degrees": <-90 to 90>,
     "forward_meters": <-1.0 to 1.0>,
     "left_meters": <-0.5 to 0.5>,
     "z_delta_meters": <-0.3 to 0.3>,
     "answer": <"A"/"B"/"C"/"D" or number>,
     "done": <true/false>
   }
   ```

### Video Baseline Prompts Include:

1. **Frame Count**: Explicit number of frames being shown
2. **Simplified Instructions**: Direct, concise guidance
3. **No Movement Options**: No rotation/translation/strafe options
4. **JSON Output Format**:
   ```json
   {
     "reasoning": "<brief explanation>",
     "answer": "<A/B/C/D or number>"
   }
   ```

---

## Answer Format Comparison

### Sequential Pipeline:
- **MCQ**: Letters "A", "B", "C", "D" (string)
- **Counting**: Integer (e.g., 3)
- **Absolute Distance**: Decimal meters (e.g., 1.5)
- **Size Estimation**: Integer centimeters (e.g., 75)
- **Room Area**: Decimal square meters (e.g., 25.5)

### Video Baseline Pipeline:
- **MCQ**: Letters "A", "B", "C", "D" (string)
- **Counting**: Integer (e.g., 3)
- **Absolute Distance**: Number with decimal (e.g., 1.5)
- **Size Estimation**: Integer centimeters (e.g., 75)
- **Room Area**: Decimal square meters (e.g., 25.5)
- **Appearance Order**: Integer (order/rank)

---

## Key Behavioral Differences

| Aspect | Sequential | Video Baseline |
|--------|-----------|-----------------|
| **Agent Control** | Full (rotate, move, strafe) | None (passive viewer) |
| **Exploration** | Active, up to 15 steps | Fixed frame sequence |
| **Perspective** | Multiple dynamic viewpoints | Fixed camera path points |
| **Measurement** | Can move to verify distances | Monocular estimation only |
| **Scene Mapping** | Can explore entire room | Limited to recorded path |
| **Decision Making** | Choose when to stop exploring | Answer after viewing all frames |
| **Temporal Info** | Agent controls timing | Pre-recorded frame sequence |

---

## Prompt Philosophy

### Sequential Design Philosophy:
- **Embodied Reasoning**: Agent physically explores to gather evidence
- **Active Problem Solving**: Choose how much exploration needed
- **Detailed Guidance**: Extensive instructions for complex spatial reasoning
- **Movement-Centric**: Heavy emphasis on exploration strategy

### Video Baseline Design Philosophy:
- **Passive Observation**: Viewer analyzes fixed video content
- **Simplicity**: Concise instructions without movement options
- **Task Focus**: Direct guidance on what to analyze
- **Comparison Baseline**: Simpler than sequential to establish ceiling/floor

---

## Task-Specific Implementation Notes

### Route Planning:
- **Sequential**: Agent physically traverses the route
- **Video**: Camera path may partially show the route

### Object Relative Direction (Hardest Type):
- **Sequential**: Agent can rotate 360° at any location to verify 8-direction answer
- **Video**: Limited to whatever angles are in the recording

### Object Counting:
- **Sequential**: Agent is warned about double-counting from different angles
- **Video**: Same warning, but cannot actively re-explore areas

### Room Size Estimation:
- **Sequential**: Exploration is explicitly requested to find room boundaries
- **Video**: Relies on how comprehensive the original video walkthrough was

---

## References

- [sequential.py](sequential.py) - `build_instruction_text()` and `_get_question_type_guidance()`
- [video_baseline.py](video_baseline.py) - `build_video_prompt()`
