"""Instruction text generation for Qwen model prompts."""


def build_instruction_text(R, t, question, bbox=None, options=None, is_final_step=False, movement_history=None, step_num=0):
    """
    Build complete instruction text including movement history.
    
    Args:
        R: 3x3 rotation matrix
        t: 3x1 translation vector
        question: Question text
        bbox: (mins, maxs) bounding box tuple
        options: List of answer choices
        is_final_step: Whether this is the final step
        movement_history: List of previous movements
        step_num: Current step number
    
    Returns:
        Complete instruction text as string
    """
    import numpy as np
    
    R_rounded = np.round(R, 2).tolist()
    t_rounded = np.round(t, 2).tolist()
    instr = build_instruction_natural(R_rounded, t_rounded, question, bbox=bbox, options=options, is_final_step=is_final_step, step_num=step_num)

    # Add movement history to the instruction with clear numbering
    movement_history_text = ""
    if movement_history:
        movement_history_text = "\n---\nYour previous movements and actions (IMPORTANT: Do not repeat these movements exactly - vary your exploration strategy):\n"
        for i, movement in enumerate(movement_history, 1):
            movement_history_text += f"  Step {i}: Moved forward={movement['forward']:.2f}m, left={movement['left']:.2f}m, up={movement['z_delta']:.2f}m, rotated={movement['rotation']:.1f}°\n"
        movement_history_text += "\nAvoid repeating these exact movements. Explore new areas of the scene.\n"

    return instr + movement_history_text


def build_instruction_natural(R_rounded, t_rounded, question, bbox=None, options=None, is_final_step=False, step_num=0):
    """
    Build natural language instruction for Qwen model.
    
    Args:
        R_rounded: Rounded rotation matrix (for display)
        t_rounded: Rounded translation vector (for display)
        question: Question text
        bbox: (mins, maxs) bounding box tuple
        options: List of answer choices
        is_final_step: Whether this is the final step
        step_num: Current step number
    
    Returns:
        Instruction string
    """
    bbox_text = ""
    if bbox is not None:
        try:
            mins, maxs = bbox
            # Limit to 2 significant figures
            mins_2sf = [float(f"{x:.2g}") for x in mins]
            maxs_2sf = [float(f"{x:.2g}") for x in maxs]
            bbox_text = (
                "Scene bounding box limits (meters):\n"
                f"  x: [{mins_2sf[0]}, {maxs_2sf[0]}]\n"
                f"  y: [{mins_2sf[1]}, {maxs_2sf[1]}]\n"
                f"  z: [{mins_2sf[2]}, {maxs_2sf[2]}]\n\n"
            )
        except Exception:
            bbox_text = ""

    # Format answer options if provided
    options_text = ""
    if options and isinstance(options, (list, dict)):
        options_text = "\n**Answer Options:**\n"
        if isinstance(options, list):
            for i, opt in enumerate(options):
                options_text += f"  {chr(65+i)}) {opt}\n"
        elif isinstance(options, dict):
            for key, val in options.items():
                options_text += f"  {key}) {val}\n"
        options_text += "\n"

    # Conditional instruction text based on whether this is the final step
    if is_final_step:
        movement_instruction = """
4. **THIS IS YOUR FINAL STEP:** You have now explored the scene thoroughly. You MUST provide your final answer choice (A, B, C, or D).
   - Prioritize outputting your best answer.
   - You may provide minimal camera movement (e.g., small adjustments) or zero movement.
   - The critical requirement is that you output your final answer in the JSON response.
   - Set "done": true ONLY when you are providing a final answer.
"""
        important_note = "IMPORTANT: This is your final opportunity to answer. You MUST output your best answer choice (A/B/C/D) in the JSON and set done=true."
    else:
        movement_instruction = """
4. **Do NOT stay in the same location.** Always provide at least one non-zero movement.

5. **Answering too early is a mistake.** ONLY output an answer (A, B, C, or D) when you have thoroughly explored the scene and are CERTAIN of your answer. 
   - Most of the time, you should NOT provide an answer yet - just explore and set "done": false.
   - Only when you are completely confident should you set both "answer": "X" AND "done": true together.
   - Never output an answer while still exploring (done=false). Either explore (no answer) OR finalize (answer + done=true).

6. **When you are confident:** If you have seen enough viewpoints and are absolutely certain of your answer, set "done": true AND provide your answer.
"""
        important_note = "IMPORTANT: Keep exploring and DO NOT output an answer unless you are ready to stop (done=true). Answer and done=true should come together, not separately."

    # Replace the first sentence of the question with a coherent starting point
    question = question.replace("You are a robot beginning at", "If you begin navigating at")

    # Add a note for Qwen to find the starting place first
    starting_place_note = "\n**Note:** Your first task is to identify the starting place mentioned in the question.\n"

    instr = f"""
            You are given rendered views of a 3D room scene. You control the camera by specifying how to move it. 
            Your goal is to find the answer to the question below by exploring the scene with camera movements.

            {bbox_text}

            # Current camera position: {t_rounded} (x, y, z in meters)
            # This is exploration step {step_num}

            Question: {question}
            {options_text}

            {starting_place_note}
            {important_note}

            ---
            DETAILED REASONING INSTRUCTIONS:

            1. **Analyze the current view thoroughly:**
            - What objects and areas are visible?
            - What areas are hidden or blocked from the current viewpoint?
            - What would you need to see to better answer the question?

            2. **Plan your camera movement:**
            - Decide how much to move forward/backward (toward/away from what you're looking at)
            - Decide how much to move left/right (perpendicular to where you're looking)
            - Think about what rotation (turning left/right) might help
            - Write out your reasoning in plain English before giving the JSON.

            3. **Specify the movement:**
            - `rotation_angle_degrees`: positive = turn left, negative = turn right (e.g., 15, -30, 90)
            - `forward_meters`: positive = move forward (in the direction you're facing), negative = move backward. Range: -0.5 to +0.5 meters
            - `left_meters`: positive = strafe left, negative = strafe right. Range: -0.5 to +0.5 meters
            - `z_delta_meters`: move up (+) or down (-) to change height. Range: -0.3 to +0.3 meters

            {movement_instruction}

            ---
            FORMAT:

            First, write your detailed reasoning (3-4 sentences explaining what you see and why you're moving the camera, or why you're ready to finalize your answer).

            Then end with a JSON object:
            {{
            "rotation_angle_degrees": <number, e.g., 15 or -30>,
            "forward_meters": <number between -0.5 and 0.5>,
            "left_meters": <number between -0.5 and 0.5>,
            "z_delta_meters": <number between -0.3 and 0.3>,
            "answer": "your final answer choice (A, B, C, or D) ONLY if done=true, otherwise null",
            "done": true or false
            }}

            CRITICAL: "answer" and "done" should be consistent:
            - If done=false: answer MUST be null (you are still exploring)
            - If done=true: answer MUST be A, B, C, or D (you have decided to finalize)
            """
    return instr
