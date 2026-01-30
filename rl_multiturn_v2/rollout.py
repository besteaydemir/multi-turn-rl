#!/usr/bin/env python3
"""
Rollout engine for multi-turn trajectory collection.

Uses vLLM for efficient inference during rollout.
Collects all data needed for training:
- Prompts
- Generated tokens and logprobs
- Turn boundaries
- Parsed actions

No gradients are computed during rollout.
"""

import time
import torch
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Callable, Tuple
from pathlib import Path
import json

from .data_structures import Turn, Trajectory, Action, CameraPose, FinalAnswer
from .output_parser import (
    OutputParser, 
    ActionTokenMasker, 
    ParseResult,
    create_turn_prompt,
    create_system_prompt,
)


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class RolloutConfig:
    """Configuration for rollout engine."""
    
    # Episode structure
    max_turns: int = 5                    # Fixed number of views T
    
    # Generation parameters
    max_new_tokens: int = 512             # Max tokens per turn
    temperature: float = 0.7              # Sampling temperature
    top_p: float = 0.9                    # Nucleus sampling
    top_k: int = 50                       # Top-k sampling
    
    # vLLM settings
    model_id: str = "Qwen/Qwen3-VL-4B-Instruct"
    tensor_parallel_size: int = 1
    gpu_memory_utilization: float = 0.9
    
    # Paths
    cache_dir: str = "/dss/dssmcmlfs01/pn34sa/pn34sa-dss-0000/aydemir"
    
    # Retry settings
    max_parse_retries: int = 3            # Retries on parse failure
    
    # Device
    device: str = "cuda"


# ============================================================================
# VLLM ROLLOUT ENGINE
# ============================================================================

class VLLMRolloutEngine:
    """
    Rollout engine using vLLM for efficient inference.
    
    Responsibilities:
    - Generate model outputs for each turn
    - Collect tokens and log probabilities
    - Parse actions from outputs
    - Manage the multi-turn loop
    
    Does NOT:
    - Compute gradients
    - Update model weights
    - Execute rendering (handled by external renderer)
    """
    
    def __init__(
        self,
        config: RolloutConfig,
        render_fn: Optional[Callable[[CameraPose], str]] = None,
    ):
        """
        Initialize rollout engine.
        
        Args:
            config: RolloutConfig with hyperparameters
            render_fn: Function that takes CameraPose and returns rendered image path
                       If None, rendering is skipped (for testing)
        """
        self.config = config
        self.render_fn = render_fn
        
        # Initialize parser
        self.parser = OutputParser(strict=False)
        
        # Initialize vLLM (lazy loading)
        self._llm = None
        self._processor = None
        self._tokenizer = None
        self._masker = None
        
        # Statistics
        self.stats = {
            "episodes_total": 0,
            "episodes_completed": 0,
            "parse_failures": 0,
            "total_tokens_generated": 0,
        }
    
    def _init_vllm(self):
        """Lazily initialize vLLM."""
        if self._llm is not None:
            return
        
        print(f"[RolloutEngine] Initializing vLLM with {self.config.model_id}...")
        
        from vllm import LLM, SamplingParams
        from transformers import AutoProcessor
        
        self._llm = LLM(
            model=self.config.model_id,
            tensor_parallel_size=self.config.tensor_parallel_size,
            gpu_memory_utilization=self.config.gpu_memory_utilization,
            trust_remote_code=True,
            download_dir=self.config.cache_dir,
            limit_mm_per_prompt={"image": self.config.max_turns + 1, "video": 0},
        )
        
        self._processor = AutoProcessor.from_pretrained(
            self.config.model_id,
            cache_dir=self.config.cache_dir
        )
        self._tokenizer = self._processor.tokenizer
        self._masker = ActionTokenMasker(self._tokenizer)
        
        print("[RolloutEngine] vLLM initialized.")
    
    @property
    def llm(self):
        self._init_vllm()
        return self._llm
    
    @property
    def tokenizer(self):
        self._init_vllm()
        return self._tokenizer
    
    @property
    def processor(self):
        self._init_vllm()
        return self._processor
    
    def collect_trajectory(
        self,
        question: str,
        choices: List[str],
        scene_id: str,
        ground_truth: Optional[str] = None,
        initial_image_path: Optional[str] = None,
        trajectory_id: Optional[str] = None,
    ) -> Trajectory:
        """
        Collect a complete trajectory for one episode.
        
        This is the main entry point for rollout.
        
        Args:
            question: The question to answer
            choices: List of answer choices
            scene_id: Identifier for the 3D scene
            ground_truth: Correct answer (for reward computation)
            initial_image_path: Optional starting image
            trajectory_id: Optional ID for this trajectory
            
        Returns:
            Complete Trajectory with all turns and data
        """
        from vllm import SamplingParams
        
        self._init_vllm()
        
        if trajectory_id is None:
            trajectory_id = f"traj_{int(time.time())}_{self.stats['episodes_total']:04d}"
        
        # Initialize trajectory
        trajectory = Trajectory(
            trajectory_id=trajectory_id,
            question=question,
            choices=choices,
            ground_truth=ground_truth,
            scene_id=scene_id,
            max_turns=self.config.max_turns,
            model_id=self.config.model_id,
            device=self.config.device,
            start_time=time.time(),
        )
        
        # Initialize image list
        image_paths = []
        if initial_image_path:
            image_paths.append(initial_image_path)
        
        # Format question with choices
        formatted_question = self._format_question(question, choices)
        
        # Sampling parameters for generation
        sampling_params = SamplingParams(
            max_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            logprobs=1,  # Request log probabilities
        )
        
        # =====================================================================
        # MULTI-TURN ROLLOUT LOOP
        # =====================================================================
        
        for turn_idx in range(self.config.max_turns):
            is_final_turn = (turn_idx == self.config.max_turns - 1)
            
            turn_start_time = time.time()
            
            # Create prompt for this turn
            prompt_text = create_turn_prompt(
                question=formatted_question,
                image_paths=image_paths,
                turn_number=turn_idx + 1,
                max_turns=self.config.max_turns,
                is_final_turn=is_final_turn,
            )
            
            # Create messages for vLLM
            messages = self._create_messages(prompt_text, image_paths)
            
            # Generate with retries
            turn = self._generate_turn(
                turn_idx=turn_idx,
                prompt_text=prompt_text,
                messages=messages,
                image_paths=image_paths.copy(),
                question=formatted_question,
                sampling_params=sampling_params,
                is_final_turn=is_final_turn,
            )
            
            turn.generation_time_seconds = time.time() - turn_start_time
            
            # Add turn to trajectory
            trajectory.turns.append(turn)
            
            # If this is the final turn, extract final answer
            if is_final_turn:
                if turn.final_answer:
                    trajectory.final_answer_text = turn.final_answer.answer_text
                    trajectory.is_correct = (
                        ground_truth is not None and 
                        turn.final_answer.answer_text.upper() == ground_truth.upper()
                    )
                break
            
            # Execute action and get new image (if render_fn provided)
            if turn.action and turn.action.parse_success and self.render_fn:
                try:
                    new_image_path = self.render_fn(turn.action.camera_pose)
                    turn.rendered_image_path = new_image_path
                    image_paths.append(new_image_path)
                except Exception as e:
                    print(f"[RolloutEngine] Render error: {e}")
        
        # Finalize trajectory
        trajectory.end_time = time.time()
        trajectory.total_duration_seconds = trajectory.end_time - trajectory.start_time
        trajectory.num_turns = len(trajectory.turns)
        
        # Compute terminal reward (0 for now, placeholder)
        trajectory.terminal_reward = 0.0  # Zero reward as per spec
        
        # Update stats
        self.stats["episodes_total"] += 1
        self.stats["episodes_completed"] += 1
        
        return trajectory
    
    def _generate_turn(
        self,
        turn_idx: int,
        prompt_text: str,
        messages: List[Dict],
        image_paths: List[str],
        question: str,
        sampling_params,
        is_final_turn: bool,
    ) -> Turn:
        """
        Generate output for a single turn with retry logic.
        """
        from vllm import SamplingParams
        
        turn = Turn(
            turn_index=turn_idx,
            prompt_text=prompt_text,
            image_paths=image_paths,
            question=question,
            timestamp=time.time(),
        )
        
        for attempt in range(self.config.max_parse_retries):
            # Generate using vLLM
            try:
                outputs = self._call_vllm(messages, sampling_params)
            except Exception as e:
                print(f"[RolloutEngine] vLLM error on attempt {attempt + 1}: {e}")
                continue
            
            if not outputs or not outputs[0].outputs:
                continue
            
            output = outputs[0].outputs[0]
            generated_text = output.text
            
            # Extract token IDs and logprobs
            token_ids = list(output.token_ids)
            logprobs = self._extract_logprobs(output)
            
            turn.generated_text = generated_text
            turn.generated_ids = torch.tensor(token_ids, dtype=torch.long)
            turn.logprobs = logprobs
            
            self.stats["total_tokens_generated"] += len(token_ids)
            
            # Parse the output
            parse_result = self.parser.parse(generated_text)
            
            # Store reasoning blocks
            turn.reasoning_blocks = {
                k.lower(): v for k, v in parse_result.sections.items()
                if k in ["STATE", "PLAN", "PREDICT"]
            }
            
            # Handle final turn
            if is_final_turn:
                if parse_result.final_answer:
                    turn.final_answer = parse_result.final_answer
                    # Success - no action needed on final turn
                    return turn
                else:
                    # Try to extract answer from text anyway
                    import re
                    match = re.search(r'\b([A-D])\b', generated_text.upper())
                    if match:
                        turn.final_answer = FinalAnswer(
                            answer_text=match.group(1),
                            raw_text=generated_text
                        )
                        return turn
            else:
                # Regular turn - need action
                if parse_result.success and parse_result.action:
                    turn.action = parse_result.action
                    
                    # Create action token mask
                    mask, start_idx, end_idx = self._masker.create_mask(
                        generated_text, turn.generated_ids, parse_result
                    )
                    turn.action_token_mask = mask
                    turn.action_token_start = start_idx
                    turn.action_token_end = end_idx
                    
                    return turn
                else:
                    self.stats["parse_failures"] += 1
        
        # All retries exhausted - return partial turn
        return turn
    
    def _call_vllm(self, messages: List[Dict], sampling_params) -> List:
        """
        Call vLLM to generate output.
        """
        # Apply chat template
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        
        # Prepare multi-modal data
        from qwen_vl_utils import process_vision_info
        image_inputs, _ = process_vision_info(messages)
        
        # Create vLLM request
        if image_inputs:
            mm_data = {"image": image_inputs}
            outputs = self.llm.generate(
                [{"prompt": prompt, "multi_modal_data": mm_data}],
                sampling_params=sampling_params,
            )
        else:
            outputs = self.llm.generate(
                [prompt],
                sampling_params=sampling_params,
            )
        
        return outputs
    
    def _extract_logprobs(self, output) -> Optional[torch.Tensor]:
        """
        Extract log probabilities from vLLM output.
        """
        if output.logprobs is None:
            return None
        
        # vLLM returns logprobs as a list of dicts
        logprobs = []
        for lp_dict in output.logprobs:
            if lp_dict:
                # Get the logprob for the selected token
                # lp_dict is a dict mapping token_id to Logprob object
                for token_id, lp in lp_dict.items():
                    logprobs.append(lp.logprob)
                    break
            else:
                logprobs.append(0.0)
        
        return torch.tensor(logprobs, dtype=torch.float32)
    
    def _create_messages(
        self,
        prompt_text: str,
        image_paths: List[str],
    ) -> List[Dict]:
        """
        Create messages in the format expected by the model.
        """
        messages = []
        
        # System message
        messages.append({
            "role": "system",
            "content": create_system_prompt()
        })
        
        # User message with images and prompt
        content = []
        
        # Add images
        for img_path in image_paths:
            content.append({
                "type": "image",
                "image": img_path
            })
        
        # Add text prompt
        content.append({
            "type": "text",
            "text": prompt_text
        })
        
        messages.append({
            "role": "user",
            "content": content
        })
        
        return messages
    
    def _format_question(self, question: str, choices: List[str]) -> str:
        """Format question with choices."""
        if not choices:
            return question
        
        choice_text = "\n".join([
            f"{chr(65 + i)}. {choice}" 
            for i, choice in enumerate(choices)
        ])
        
        return f"{question}\n\nChoices:\n{choice_text}"
    
    def collect_batch(
        self,
        questions: List[Dict[str, Any]],
        render_fn: Optional[Callable] = None,
    ) -> List[Trajectory]:
        """
        Collect trajectories for a batch of questions.
        
        Args:
            questions: List of dicts with keys:
                - question: str
                - choices: List[str]
                - scene_id: str
                - ground_truth: Optional[str]
                - initial_image_path: Optional[str]
            render_fn: Render function (overrides self.render_fn if provided)
                
        Returns:
            List of Trajectory objects
        """
        trajectories = []
        
        if render_fn:
            self.render_fn = render_fn
        
        for i, q in enumerate(questions):
            print(f"[RolloutEngine] Collecting trajectory {i+1}/{len(questions)}: {q['scene_id']}")
            
            traj = self.collect_trajectory(
                question=q["question"],
                choices=q.get("choices", []),
                scene_id=q["scene_id"],
                ground_truth=q.get("ground_truth"),
                initial_image_path=q.get("initial_image_path"),
            )
            
            trajectories.append(traj)
        
        return trajectories
    
    def get_stats(self) -> Dict[str, Any]:
        """Get rollout statistics."""
        return self.stats.copy()


# ============================================================================
# MOCK ROLLOUT ENGINE (FOR TESTING)
# ============================================================================

class MockRolloutEngine:
    """
    Mock rollout engine for testing without GPU.
    
    Generates synthetic trajectories with correct structure.
    """
    
    def __init__(self, config: RolloutConfig):
        self.config = config
        self.stats = {"episodes_total": 0}
    
    def collect_trajectory(
        self,
        question: str,
        choices: List[str],
        scene_id: str,
        ground_truth: Optional[str] = None,
        **kwargs
    ) -> Trajectory:
        """Generate a mock trajectory."""
        trajectory = Trajectory(
            trajectory_id=f"mock_{self.stats['episodes_total']}",
            question=question,
            choices=choices,
            ground_truth=ground_truth,
            scene_id=scene_id,
            max_turns=self.config.max_turns,
            model_id="mock",
        )
        
        for turn_idx in range(self.config.max_turns):
            is_final = (turn_idx == self.config.max_turns - 1)
            
            # Mock tokens
            mock_ids = torch.randint(0, 1000, (50,))
            mock_logprobs = torch.randn(50) * 0.5 - 2.0  # Reasonable logprobs
            mock_mask = torch.zeros(50, dtype=torch.bool)
            mock_mask[30:45] = True  # Action tokens in middle
            
            turn = Turn(
                turn_index=turn_idx,
                prompt_text=f"Mock prompt for turn {turn_idx}",
                question=question,
                generated_text="[STATE]\nMock state\n[PLAN]\nMock plan\n[ACTION]\n{\"camera_pose\": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], \"fov\": 60}",
                generated_ids=mock_ids,
                logprobs=mock_logprobs,
                action_token_mask=mock_mask,
                action_token_start=30,
                action_token_end=45,
            )
            
            if is_final:
                turn.final_answer = FinalAnswer(
                    answer_text=choices[0] if choices else "A",
                    raw_text="A"
                )
            else:
                turn.action = Action(
                    camera_pose=CameraPose(
                        transform_matrix=np.eye(4),
                        fov=60.0
                    )
                )
            
            trajectory.turns.append(turn)
        
        trajectory.num_turns = len(trajectory.turns)
        trajectory.final_answer_text = trajectory.turns[-1].final_answer.answer_text if trajectory.turns[-1].final_answer else ""
        trajectory.is_correct = trajectory.final_answer_text == ground_truth if ground_truth else False
        
        self.stats["episodes_total"] += 1
        
        return trajectory
