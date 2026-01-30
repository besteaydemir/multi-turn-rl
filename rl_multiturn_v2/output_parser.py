#!/usr/bin/env python3
"""
Output parser for bracketed marker format.

Parses model outputs in the format:
    [STATE]
    Free-form description...

    [PLAN]
    Why another view is needed...

    [PREDICT]
    What the next view is expected to show...

    [ACTION]
    {
      "camera_pose": [...],
      "fov": ...
    }

    [FINAL_ANSWER]
    <answer text>

This format is:
- Easier for VLMs than XML
- Compatible with vLLM
- Extendable to VAGEN-style world-model rewards
"""

import re
import json
from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
import torch

from .data_structures import Action, CameraPose, FinalAnswer


# ============================================================================
# REGEX PATTERNS
# ============================================================================

# Pattern to match bracketed headers
HEADER_PATTERN = re.compile(r'\[([A-Z_]+)\]')

# Pattern to match JSON objects
JSON_PATTERN = re.compile(r'\{[\s\S]*?\}', re.DOTALL)

# Valid section headers
VALID_HEADERS = {"STATE", "PLAN", "PREDICT", "ACTION", "FINAL_ANSWER"}


# ============================================================================
# PARSE RESULT
# ============================================================================

@dataclass
class ParseResult:
    """
    Result of parsing model output.
    
    Contains:
    - Extracted sections (STATE, PLAN, PREDICT)
    - Parsed action (from ACTION block)
    - Final answer (from FINAL_ANSWER block, if present)
    - Token position information for masking
    """
    # Raw text
    raw_text: str = ""
    
    # Parsed sections (reasoning, not supervised)
    sections: Dict[str, str] = field(default_factory=dict)
    # Keys: "STATE", "PLAN", "PREDICT", "ACTION", "FINAL_ANSWER"
    
    # Parsed action (from ACTION section JSON)
    action: Optional[Action] = None
    
    # Final answer (only if FINAL_ANSWER section exists)
    final_answer: Optional[FinalAnswer] = None
    
    # Was parsing successful?
    success: bool = True
    
    # Error message if parsing failed
    error_message: str = ""
    
    # Character positions of ACTION block (for token mapping)
    action_block_start: Optional[int] = None  # Character index
    action_block_end: Optional[int] = None    # Character index
    
    # Character positions of the JSON inside ACTION block
    action_json_start: Optional[int] = None
    action_json_end: Optional[int] = None


# ============================================================================
# OUTPUT PARSER
# ============================================================================

class OutputParser:
    """
    Parser for bracketed marker format output.
    
    Extracts:
    - Reasoning sections (STATE, PLAN, PREDICT) - not supervised
    - Action JSON (from ACTION block) - used for RL
    - Final answer (from FINAL_ANSWER block) - used for reward
    """
    
    def __init__(self, strict: bool = False):
        """
        Initialize parser.
        
        Args:
            strict: If True, require all sections. If False, be lenient.
        """
        self.strict = strict
    
    def parse(self, text: str) -> ParseResult:
        """
        Parse model output text.
        
        Args:
            text: Raw model output
            
        Returns:
            ParseResult with extracted sections and parsed action
        """
        result = ParseResult(raw_text=text)
        
        try:
            # Step 1: Extract sections by header
            sections = self._extract_sections(text)
            result.sections = sections
            
            # Step 2: Parse ACTION block if present
            if "ACTION" in sections:
                action_text = sections["ACTION"]
                action, json_start, json_end = self._parse_action(action_text)
                result.action = action
                
                # Calculate absolute positions in original text
                if json_start is not None:
                    action_block_match = re.search(r'\[ACTION\]', text)
                    if action_block_match:
                        block_start = action_block_match.end()
                        result.action_block_start = block_start
                        result.action_json_start = block_start + json_start
                        result.action_json_end = block_start + json_end
                        
                        # Find end of ACTION block (next header or end of text)
                        next_header = re.search(r'\[[A-Z_]+\]', text[block_start:])
                        if next_header:
                            result.action_block_end = block_start + next_header.start()
                        else:
                            result.action_block_end = len(text)
                
                if action is None or not action.parse_success:
                    result.success = False
                    result.error_message = action.parse_error if action else "Failed to parse ACTION JSON"
            else:
                # No ACTION block found
                if self.strict:
                    result.success = False
                    result.error_message = "No [ACTION] section found"
            
            # Step 3: Parse FINAL_ANSWER block if present
            if "FINAL_ANSWER" in sections:
                answer_text = sections["FINAL_ANSWER"].strip()
                result.final_answer = FinalAnswer(
                    answer_text=self._clean_answer(answer_text),
                    raw_text=answer_text
                )
            
        except Exception as e:
            result.success = False
            result.error_message = f"Parse error: {str(e)}"
        
        return result
    
    def _extract_sections(self, text: str) -> Dict[str, str]:
        """
        Extract sections by bracketed headers.
        
        Returns:
            Dict mapping header names to section content
        """
        sections = {}
        
        # Find all header matches with their positions
        headers = []
        for match in HEADER_PATTERN.finditer(text):
            header_name = match.group(1)
            if header_name in VALID_HEADERS:
                headers.append((match.start(), match.end(), header_name))
        
        # Extract content between headers
        for i, (start, end, name) in enumerate(headers):
            # Content starts after the header
            content_start = end
            
            # Content ends at the next header or end of text
            if i + 1 < len(headers):
                content_end = headers[i + 1][0]
            else:
                content_end = len(text)
            
            content = text[content_start:content_end].strip()
            sections[name] = content
        
        return sections
    
    def _parse_action(self, action_text: str) -> Tuple[Optional[Action], Optional[int], Optional[int]]:
        """
        Parse the ACTION section to extract camera pose JSON.
        
        Returns:
            (Action, json_start_position, json_end_position)
        """
        # Find JSON object in action text
        match = JSON_PATTERN.search(action_text)
        if not match:
            return Action(
                camera_pose=CameraPose(),
                parse_success=False,
                parse_error="No JSON object found in ACTION section"
            ), None, None
        
        json_str = match.group()
        json_start = match.start()
        json_end = match.end()
        
        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            return Action(
                camera_pose=CameraPose(),
                raw_json={},
                parse_success=False,
                parse_error=f"Invalid JSON: {str(e)}"
            ), json_start, json_end
        
        # Parse camera_pose
        camera_pose = self._parse_camera_pose(data)
        
        return Action(
            camera_pose=camera_pose,
            raw_json=data,
            parse_success=True
        ), json_start, json_end
    
    def _parse_camera_pose(self, data: Dict[str, Any]) -> CameraPose:
        """
        Parse camera pose from JSON data.
        
        Supports multiple formats:
        - {"camera_pose": [[4x4 matrix]], "fov": 60}
        - {"position": [x,y,z], "rotation": [[3x3]], "fov": 60}
        - {"transform": [[4x4 matrix]]}
        """
        import numpy as np
        
        fov = data.get("fov", 60.0)
        transform_matrix = None
        position = None
        rotation = None
        
        # Try to extract camera_pose as 4x4 matrix
        if "camera_pose" in data:
            try:
                matrix = np.array(data["camera_pose"])
                if matrix.shape == (4, 4):
                    transform_matrix = matrix
                elif matrix.shape == (16,):
                    transform_matrix = matrix.reshape(4, 4)
            except (ValueError, TypeError):
                pass
        
        # Try transform key
        if transform_matrix is None and "transform" in data:
            try:
                matrix = np.array(data["transform"])
                if matrix.shape == (4, 4):
                    transform_matrix = matrix
            except (ValueError, TypeError):
                pass
        
        # Try position + rotation
        if "position" in data:
            try:
                position = np.array(data["position"])
                if position.shape != (3,):
                    position = None
            except (ValueError, TypeError):
                pass
        
        if "rotation" in data:
            try:
                rotation = np.array(data["rotation"])
                if rotation.shape != (3, 3):
                    rotation = None
            except (ValueError, TypeError):
                pass
        
        return CameraPose(
            transform_matrix=transform_matrix,
            position=position,
            rotation=rotation,
            fov=fov
        )
    
    def _clean_answer(self, answer_text: str) -> str:
        """
        Clean up the final answer text.
        
        Extracts just the answer choice (A, B, C, D) if present.
        """
        # Try to extract single letter answer
        match = re.search(r'\b([A-D])\b', answer_text.upper())
        if match:
            return match.group(1)
        
        # Return cleaned text
        return answer_text.strip()


# ============================================================================
# TOKEN MASKER
# ============================================================================

class ActionTokenMasker:
    """
    Identifies action tokens in generated sequence.
    
    Uses character positions from parsing to map to token positions.
    """
    
    def __init__(self, tokenizer):
        """
        Initialize masker.
        
        Args:
            tokenizer: The tokenizer used for generation
        """
        self.tokenizer = tokenizer
    
    def create_mask(
        self,
        generated_text: str,
        generated_ids: torch.Tensor,
        parse_result: ParseResult
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Create action token mask.
        
        Args:
            generated_text: The full generated text
            generated_ids: Token IDs of generated sequence
            parse_result: Parsing result with character positions
            
        Returns:
            (mask tensor, start_index, end_index)
        """
        seq_len = len(generated_ids)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        
        if parse_result.action_json_start is None or parse_result.action_json_end is None:
            # Could not identify action block - use fallback
            return self._fallback_mask(generated_text, generated_ids)
        
        # Map character positions to token positions
        # This is done by decoding prefix and finding where action starts
        start_idx = None
        end_idx = None
        
        try:
            # Decode tokens one by one to find character positions
            char_pos = 0
            for i, token_id in enumerate(generated_ids):
                token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=False)
                token_end_char = char_pos + len(token_text)
                
                # Check if this token overlaps with action JSON
                if start_idx is None and token_end_char > parse_result.action_json_start:
                    start_idx = i
                
                if start_idx is not None and char_pos < parse_result.action_json_end:
                    mask[i] = True
                    end_idx = i + 1
                
                char_pos = token_end_char
                
                # Stop if we've passed the action JSON
                if char_pos >= parse_result.action_json_end:
                    break
                    
        except Exception:
            # Fallback on any error
            return self._fallback_mask(generated_text, generated_ids)
        
        return mask, start_idx, end_idx
    
    def _fallback_mask(
        self,
        generated_text: str,
        generated_ids: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[int], Optional[int]]:
        """
        Fallback: Use brace depth tracking to find JSON.
        """
        seq_len = len(generated_ids)
        mask = torch.zeros(seq_len, dtype=torch.bool)
        
        in_json = False
        brace_depth = 0
        start_idx = None
        end_idx = None
        
        for i, token_id in enumerate(generated_ids):
            token_text = self.tokenizer.decode([token_id.item()], skip_special_tokens=False)
            
            if '{' in token_text and not in_json:
                in_json = True
                start_idx = i
                brace_depth = token_text.count('{')
                mask[i] = True
            elif in_json:
                mask[i] = True
                brace_depth += token_text.count('{')
                brace_depth -= token_text.count('}')
                
                if brace_depth <= 0:
                    end_idx = i + 1
                    break
        
        return mask, start_idx, end_idx


# ============================================================================
# PROMPT TEMPLATE
# ============================================================================

def create_turn_prompt(
    question: str,
    image_paths: List[str],
    turn_number: int,
    max_turns: int,
    is_final_turn: bool = False
) -> str:
    """
    Create the prompt for a turn.
    
    Args:
        question: The question being answered
        image_paths: Paths to all images collected so far
        turn_number: Current turn (1-indexed)
        max_turns: Maximum number of turns
        is_final_turn: Whether this is the final turn
        
    Returns:
        Formatted prompt string
    """
    # Image context
    if image_paths:
        image_section = f"You have collected {len(image_paths)} view(s) of the scene."
    else:
        image_section = "No views have been collected yet."
    
    # Build prompt
    if is_final_turn:
        # Final turn - must provide answer
        prompt = f"""## Task
You are answering a spatial reasoning question about a 3D scene.
This is turn {turn_number} of {max_turns} (FINAL TURN).

## Question
{question}

## Current State
{image_section}

## Instructions
This is your FINAL turn. You must provide your answer now.

Respond in the following format:

[STATE]
Describe what you have observed across all views.

[PLAN]
Explain your reasoning process for answering the question.

[FINAL_ANSWER]
Your answer (A, B, C, or D)
"""
    else:
        # Regular turn - select next view
        prompt = f"""## Task
You are answering a spatial reasoning question about a 3D scene by selecting camera views.
This is turn {turn_number} of {max_turns}.

## Question
{question}

## Current State
{image_section}

## Instructions
Select the next camera view to help answer the question.

Respond in the following format:

[STATE]
Describe what you can see so far and what information you still need.

[PLAN]
Explain why you need another view and what it should reveal.

[PREDICT]
Describe what you expect to see from the next view.

[ACTION]
{{
  "camera_pose": [[4x4 transformation matrix as nested list]],
  "fov": 60.0
}}
"""
    
    return prompt


def create_system_prompt() -> str:
    """
    Create the system prompt for the model.
    """
    return """You are a visual spatial reasoning assistant. You explore 3D scenes by selecting camera views to gather information needed to answer questions.

For each turn, you will:
1. Analyze what you've seen so far
2. Plan what information you still need
3. Select a camera view to gather that information

On the final turn, you will provide your answer based on all views collected.

Always respond using the bracketed format specified in the instructions."""
