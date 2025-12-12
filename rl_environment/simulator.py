#!/usr/bin/env python3
"""
Episode simulator for multi-turn RL pipeline.
Integrates with the baseline pipeline to run complete episodes and collect trajectory data.
"""

import torch
import numpy as np
import time
import re
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime

from .environment import (
    NavigationEnvironment, 
    Observation, 
    Action, 
    Turn, 
    Episode
)
from .masking import ActionTokenMasker


class EpisodeSimulator:
    """
    Simulates complete episodes and collects trajectory data for RL.
    """
    
    def __init__(
        self,
        model,
        processor,
        device: str = "cuda",
        max_steps: int = 10,
        track_action_tokens: bool = True,
        do_sample: bool = True,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 50,
        min_action_tokens: int = 10,  # Minimum expected action tokens
        max_action_tokens: int = 100  # Maximum expected action tokens
    ):
        """
        Initialize episode simulator.
        
        Args:
            model: Qwen3VL model
            processor: Qwen3VL processor
            device: Device to run model on
            max_steps: Maximum steps per episode
            track_action_tokens: Whether to track action token positions
            do_sample: Whether to use sampling (vs greedy)
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling threshold
            top_k: Top-k sampling threshold
            min_action_tokens: Minimum expected action tokens (for validation)
            max_action_tokens: Maximum expected action tokens (for validation)
        """
        self.model = model
        self.processor = processor
        self.device = device
        self.max_steps = max_steps
        self.track_action_tokens = track_action_tokens
        self.do_sample = do_sample
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.min_action_tokens = min_action_tokens
        self.max_action_tokens = max_action_tokens
        
        # Initialize action token masker (Step 3)
        self.masker = ActionTokenMasker(
            processor=processor,
            min_action_tokens=min_action_tokens,
            max_action_tokens=max_action_tokens
        )
        
        # Statistics tracking (Step 3 requirement)
        self.stats = {
            "episodes_total": 0,
            "episodes_dropped": 0,
            "episodes_valid": 0,
            "dropout_reasons": {},
            "masking_quality": {"good": 0, "low_confidence": 0, "fallback_used": 0}
        }
    
    def get_stats(self) -> Dict[str, Any]:
        """Get simulator statistics including masking and dropout info."""
        return {
            **self.stats,
            "masking_methods": self.masker.stats,
            "dropout_rate": self.stats["episodes_dropped"] / max(self.stats["episodes_total"], 1)
        }
    
    def log_stats(self, verbose: bool = True):
        """Print statistics summary."""
        stats = self.get_stats()
        
        if verbose:
            print("\n" + "=" * 80)
            print("EPISODE SIMULATOR STATISTICS")
            print("=" * 80)
            print(f"Total episodes: {stats['episodes_total']}")
            print(f"Valid episodes: {stats['episodes_valid']}")
            print(f"Dropped episodes: {stats['episodes_dropped']} ({stats['dropout_rate']:.1%})")
            
            if stats["dropout_reasons"]:
                print("\nDropout reasons:")
                for reason, count in stats["dropout_reasons"].items():
                    print(f"  {reason}: {count}")
            
            print("\nMasking methods:")
            for method, count in stats["masking_methods"].items():
                print(f"  {method}: {count}")
            
            print("\nMasking quality:")
            for quality, count in stats["masking_quality"].items():
                print(f"  {quality}: {count}")
            print("=" * 80 + "\n")
    
    def run_episode(
        self,
        env: NavigationEnvironment,
        initial_pose: np.ndarray,
        episode_id: str,
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ) -> Episode:
        """
        Run a complete episode and collect all trajectory data.
        
        Args:
            env: NavigationEnvironment instance
            initial_pose: Initial camera pose
            episode_id: Unique episode identifier
            output_dir: Directory to save episode data
            verbose: Whether to print progress
            
        Returns:
            Complete Episode object with all turns
        """
        start_time = time.time()
        
        # Initialize episode
        episode = Episode(
            episode_id=episode_id,
            scene_id=env.scene_id,
            question=env.question,
            choices=env.choices,
            ground_truth=env.ground_truth,
            metadata={
                "start_time": start_time,
                "max_steps": self.max_steps,
                "model_device": self.device
            }
        )
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Reset environment
        obs = env.reset(initial_pose)
        done = False
        turn_index = 0
        
        if verbose:
            print(f"\n[EPISODE {episode_id}] Starting simulation...")
            print(f"Scene: {env.scene_id}")
            print(f"Question: {env.question[:80]}...")
        
        # Episode loop
        while not done and turn_index <= self.max_steps:
            if verbose:
                print(f"\n[Turn {turn_index}] {'─' * 60}")
            
            # Run single turn
            turn = self._run_turn(
                obs=obs,
                turn_index=turn_index,
                episode_id=episode_id,
                output_dir=output_dir / f"turn_{turn_index:02d}" if output_dir else None,
                verbose=verbose
            )
            
            # Add turn to episode
            episode.add_turn(turn)
            
            # Check if action is valid and done
            if turn.action and turn.action_valid:
                # Execute action in environment
                next_obs, done = env.step(turn.action)
                turn.next_observation = next_obs
                
                # Check if episode should terminate
                if turn.action.done:
                    episode.final_answer = turn.action.answer
                    done = True
                    if verbose:
                        print(f"[Turn {turn_index}] Episode terminated by model (answer: {episode.final_answer})")
            else:
                # Invalid action, terminate episode
                if verbose:
                    print(f"[Turn {turn_index}] Invalid action, terminating episode")
                done = True
            
            # Update observation for next turn
            if not done:
                obs = next_obs
            
            turn_index += 1
        
        # Compute final reward
        episode.compute_final_reward()
        
        # Evaluate episode quality and determine if should be dropped (Step 3)
        episode_quality = self._evaluate_episode_quality(episode)
        episode.is_valid = episode_quality["is_valid"]
        episode.dropout_reason = episode_quality["dropout_reason"]
        episode.masking_quality = episode_quality["masking_quality"]
        
        # Update statistics
        self.stats["episodes_total"] += 1
        if episode.is_valid:
            self.stats["episodes_valid"] += 1
        else:
            self.stats["episodes_dropped"] += 1
            reason = episode.dropout_reason or "unknown"
            self.stats["dropout_reasons"][reason] = self.stats["dropout_reasons"].get(reason, 0) + 1
        
        self.stats["masking_quality"][episode.masking_quality] = \
            self.stats["masking_quality"].get(episode.masking_quality, 0) + 1
        
        # Update metadata
        end_time = time.time()
        episode.metadata.update({
            "end_time": end_time,
            "duration_seconds": end_time - start_time,
            "num_turns": len(episode.turns),
            "final_answer": episode.final_answer,
            "is_correct": episode.is_correct,
            "final_reward": episode.final_reward,
            "is_valid": episode.is_valid,
            "dropout_reason": episode.dropout_reason,
            "masking_quality": episode.masking_quality
        })
        
        if verbose:
            print(f"\n[EPISODE {episode_id}] Complete!")
            print(f"Answer: {episode.final_answer} (Ground Truth: {episode.ground_truth})")
            print(f"Reward: {episode.final_reward:.2f}")
            print(f"Duration: {episode.metadata['duration_seconds']:.2f}s")
            print(f"Valid: {episode.is_valid} (Quality: {episode.masking_quality})")
            if not episode.is_valid:
                print(f"Dropout reason: {episode.dropout_reason}")
        
        # Save episode data
        if output_dir:
            episode.save_full(output_dir)
            if verbose:
                print(f"[EPISODE {episode_id}] Saved to {output_dir}")
        
        return episode
    
    def _run_turn(
        self,
        obs: Observation,
        turn_index: int,
        episode_id: str,
        output_dir: Optional[Path] = None,
        verbose: bool = True
    ) -> Turn:
        """
        Run a single turn: generate action from observation.
        
        Args:
            obs: Current observation
            turn_index: Turn index
            episode_id: Episode identifier
            output_dir: Directory to save turn data
            verbose: Whether to print progress
            
        Returns:
            Complete Turn object
        """
        turn_start_time = time.time()
        
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        
        # Build context text (instruction)
        context_text = self._build_context_text(obs)
        
        # Build full prompt with images
        full_prompt, messages = self._build_prompt_with_images(obs, context_text)
        
        # Save prompt if requested
        if output_dir:
            with open(output_dir / "context_text.txt", "w") as f:
                f.write(context_text)
            with open(output_dir / "full_prompt.txt", "w") as f:
                f.write(full_prompt)
            with open(output_dir / "messages.json", "w") as f:
                json.dump(messages, f, indent=2)
        
        # Generate response with real-time token tracking
        generated_ids, generated_text, action_token_mask, json_start_idx, json_end_idx, input_length, masking_diag, context_input_ids = \
            self._generate_with_tracking(messages, verbose=verbose)
        
        # Save generated output
        if output_dir:
            with open(output_dir / "generated_text.txt", "w") as f:
                f.write(generated_text)
            torch.save(generated_ids, output_dir / "generated_ids.pt")
            if action_token_mask is not None:
                torch.save(action_token_mask, output_dir / "action_token_mask.pt")
            # Save masking diagnostics
            with open(output_dir / "masking_diagnostics.json", "w") as f:
                json.dump(masking_diag, f, indent=2)
        
        # Parse action from generated text
        action, action_valid, action_error = self._parse_action(generated_text)
        
        if verbose and action:
            print(f"[Turn {turn_index}] Action: forward={action.forward_meters:.2f}m, "
                  f"left={action.left_meters:.2f}m, rotation={action.rotation_angle_degrees:.1f}°, "
                  f"done={action.done}")
            if json_start_idx is not None and json_end_idx is not None:
                print(f"[Turn {turn_index}] JSON tokens: [{json_start_idx}, {json_end_idx})")
            print(f"[Turn {turn_index}] Masking: {masking_diag['method']} (confidence: {masking_diag['confidence']:.2f})")
        
        # Build Turn object with masking diagnostics
        turn = Turn(
            turn_index=turn_index,
            observation=obs,
            full_prompt=full_prompt,
            context_text=context_text,
            generated_ids=generated_ids,
            generated_text=generated_text,
            action_token_mask=action_token_mask,
            context_input_ids=context_input_ids,
            input_token_length=input_length,
            action_token_start_index=json_start_idx,
            action_token_end_index=json_end_idx,
            num_action_tokens=masking_diag.get("num_action_tokens", 0),
            num_reasoning_tokens=masking_diag.get("num_reasoning_tokens", 0),
            masking_method=masking_diag.get("method", "unknown"),
            masking_confidence=masking_diag.get("confidence", 0.0),
            action=action,
            action_valid=action_valid,
            action_error=action_error,
            timestamp=turn_start_time
        )
        
        # Save turn metadata
        if output_dir:
            with open(output_dir / "turn_metadata.json", "w") as f:
                json.dump(turn.to_dict(), f, indent=2)
        
        return turn
    
    def _build_context_text(self, obs: Observation) -> str:
        """
        Build instruction text from observation.
        Reuses logic from baseline pipeline.
        """
        from render_point_cloud_qwen_angle import build_instruction_text
        
        R = obs.current_rotation
        t = obs.current_position
        question = obs.question
        bbox = (obs.bbox_mins, obs.bbox_maxs)
        options = obs.choices
        is_final_step = obs.is_final_step
        movement_history = obs.movement_history
        step_num = obs.step
        
        context_text = build_instruction_text(
            R, t, question,
            bbox=bbox,
            options=options,
            is_final_step=is_final_step,
            movement_history=movement_history,
            step_num=step_num
        )
        
        return context_text
    
    def _build_prompt_with_images(
        self, 
        obs: Observation, 
        context_text: str
    ) -> Tuple[str, List[Dict]]:
        """
        Build full prompt with images for model input.
        
        Returns:
            (full_prompt_text, messages_list)
        """
        # Build history context
        history_context = "## Image History (numbered for reference):\n"
        for i, cam_pose in enumerate(obs.camera_positions):
            pos = cam_pose[:3, 3]
            history_context += f"  Image {i}: position [x={pos[0]:.2f}m, y={pos[1]:.2f}m, z={pos[2]:.2f}m]\n"
        history_context += "\nAbove are all the images you have seen so far in this exploration.\n\n"
        
        full_prompt_text = history_context + context_text
        
        # Build messages (format for Qwen processor)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": full_prompt_text}
                ]
            }
        ]
        
        # Add all images
        for img_path in obs.images:
            messages[0]["content"].insert(
                len(messages[0]["content"]) - 1,
                {"type": "image", "image": img_path}
            )
        
        return full_prompt_text, messages
    
    def _evaluate_episode_quality(self, episode: Episode) -> Dict[str, Any]:
        """
        Evaluate episode quality and determine if it should be dropped (Step 3).
        
        Dropout criteria:
        1. Masking failed completely on any turn
        2. All turns used low-confidence masking
        3. Invalid action that couldn't be parsed
        
        Returns:
            Dict with is_valid, dropout_reason, masking_quality
        """
        if len(episode.turns) == 0:
            return {
                "is_valid": False,
                "dropout_reason": "no_turns_generated",
                "masking_quality": "failed"
            }
        
        # Check for critical failures
        for turn in episode.turns:
            # Complete masking failure
            if turn.masking_method == "failed":
                return {
                    "is_valid": False,
                    "dropout_reason": "masking_failed",
                    "masking_quality": "failed"
                }
            
            # Invalid action that couldn't be executed
            if not turn.action_valid and turn.masking_method != "failed":
                return {
                    "is_valid": False,
                    "dropout_reason": f"invalid_action: {turn.action_error}",
                    "masking_quality": "fallback_used"
                }
        
        # Evaluate overall masking quality
        masking_methods = [t.masking_method for t in episode.turns]
        masking_confidences = [t.masking_confidence for t in episode.turns]
        
        # All turns used last-N fallback (very low confidence)
        if all(m == "last_n_fallback" for m in masking_methods):
            return {
                "is_valid": False,
                "dropout_reason": "all_turns_low_confidence_masking",
                "masking_quality": "fallback_used"
            }
        
        # Determine quality level
        avg_confidence = sum(masking_confidences) / len(masking_confidences)
        
        if avg_confidence >= 0.9:
            masking_quality = "good"
        elif avg_confidence >= 0.6:
            masking_quality = "low_confidence"
        else:
            masking_quality = "fallback_used"
        
        return {
            "is_valid": True,
            "dropout_reason": None,
            "masking_quality": masking_quality
        }
    
    def _generate_with_tracking(
        self, 
        messages: List[Dict],
        max_new_tokens: int = 1024,
        verbose: bool = True
    ) -> Tuple[torch.Tensor, str, Optional[torch.Tensor], Optional[int], Optional[int], int, Dict, torch.Tensor]:
        """
        Generate response and track action token positions IN REAL-TIME.
        
        Uses streaming generation to detect JSON boundaries as tokens are generated,
        avoiding post-hoc string matching.
        
        Returns:
            (generated_ids, generated_text, action_token_mask, json_start_index, json_end_index, 
             input_length, masking_diagnostics, context_input_ids)
        """
        # Prepare inputs
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]
        context_input_ids = inputs['input_ids'][0]  # Store context for RL training
        
        # Generate with sampling parameters
        with torch.no_grad():
            generated_ids_full = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=self.do_sample,
                temperature=self.temperature if self.do_sample else 1.0,
                top_p=self.top_p if self.do_sample else 1.0,
                top_k=self.top_k if self.do_sample else 50,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id
            )
        
        # Extract generated tokens (trim input)
        generated_ids = generated_ids_full[0, input_length:]
        
        # Decode to text
        generated_text = self.processor.batch_decode(
            [generated_ids],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )[0]
        
        # Track action tokens with enhanced masking (Step 3)
        action_token_mask = None
        json_start_index = None
        json_end_index = None
        masking_diagnostics = {}
        
        if self.track_action_tokens:
            action_token_mask, json_start_index, json_end_index, masking_diagnostics = \
                self.masker.identify_action_tokens(generated_ids, generated_text)
        
        return generated_ids, generated_text, action_token_mask, json_start_index, json_end_index, \
               input_length, masking_diagnostics, context_input_ids
    
    def _parse_action(self, generated_text: str) -> Tuple[Optional[Action], bool, str]:
        """
        Parse action JSON from generated text.
        
        Returns:
            (action, is_valid, error_message)
        """
        from render_point_cloud_qwen_angle import extract_first_json
        
        # Extract JSON
        json_obj = extract_first_json(generated_text)
        
        if json_obj is None:
            return None, False, "No JSON found in generated text"
        
        try:
            action = Action.from_dict(json_obj)
        except Exception as e:
            return None, False, f"Failed to parse action: {e}"
        
        # Validate action
        is_valid, error_msg = action.validate()
        
        return action, is_valid, error_msg


class EpisodeBatchCollector:
    """
    Collects multiple episodes and saves them to persistent storage (JSONL).
    Designed for efficient RL training data collection.
    """
    
    def __init__(
        self,
        simulator: EpisodeSimulator,
        output_dir: Path,
        save_format: str = "both"  # "jsonl", "full", or "both"
    ):
        """
        Initialize batch collector.
        
        Args:
            simulator: EpisodeSimulator instance
            output_dir: Base directory for all episodes
            save_format: "jsonl" (compact), "full" (all tensors), or "both"
        """
        self.simulator = simulator
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_format = save_format
        
        # JSONL file for compact episode storage
        self.jsonl_path = self.output_dir / "episodes.jsonl"
        
        # Tensor storage directory
        self.tensors_dir = self.output_dir / "tensors"
        self.tensors_dir.mkdir(exist_ok=True)
        
        self.episodes_collected = 0
    
    def collect_episode(
        self,
        env: NavigationEnvironment,
        initial_pose: np.ndarray,
        episode_id: Optional[str] = None,
        verbose: bool = True
    ) -> Episode:
        """
        Collect a single episode and save to persistent storage.
        
        Args:
            env: NavigationEnvironment instance
            initial_pose: Initial camera pose
            episode_id: Optional episode ID (auto-generated if None)
            verbose: Whether to print progress
            
        Returns:
            Episode object
        """
        if episode_id is None:
            episode_id = f"episode_{self.episodes_collected:06d}_{int(time.time())}"
        
        # Run episode
        episode_output_dir = self.output_dir / episode_id if self.save_format in ["full", "both"] else None
        episode = self.simulator.run_episode(
            env=env,
            initial_pose=initial_pose,
            episode_id=episode_id,
            output_dir=episode_output_dir,
            verbose=verbose
        )
        
        # Save to JSONL
        if self.save_format in ["jsonl", "both"]:
            episode.save_to_jsonl(self.jsonl_path)
            
            # Save tensors separately with episode_id reference
            self._save_episode_tensors(episode)
        
        self.episodes_collected += 1
        
        if verbose:
            print(f"[BATCH] Collected {self.episodes_collected} episodes")
        
        return episode
    
    def _save_episode_tensors(self, episode: Episode):
        """
        Save episode tensors in a compact binary format.
        
        Args:
            episode: Episode object
        """
        episode_tensor_dir = self.tensors_dir / episode.episode_id
        episode_tensor_dir.mkdir(exist_ok=True)
        
        # Save tensors for each turn
        for turn in episode.turns:
            turn_file = episode_tensor_dir / f"turn_{turn.turn_index:02d}.pt"
            
            # Pack all tensors for this turn
            tensor_data = {
                "generated_ids": turn.generated_ids,
                "action_token_mask": turn.action_token_mask
            }
            
            torch.save(tensor_data, turn_file)
    
    def load_episode_tensors(self, episode_id: str, turn_index: int) -> Dict[str, torch.Tensor]:
        """
        Load tensors for a specific turn.
        
        Args:
            episode_id: Episode identifier
            turn_index: Turn index
            
        Returns:
            Dict with 'generated_ids' and 'action_token_mask'
        """
        turn_file = self.tensors_dir / episode_id / f"turn_{turn_index:02d}.pt"
        return torch.load(turn_file)
    
    def iter_episodes_jsonl(self):
        """
        Iterate through all episodes in the JSONL file.
        
        Yields:
            Dict: Episode data (without tensors)
        """
        if not self.jsonl_path.exists():
            return
        
        with open(self.jsonl_path, "r") as f:
            for line in f:
                yield json.loads(line.strip())
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about collected episodes.
        
        Returns:
            Dict with episode statistics
        """
        episodes = list(self.iter_episodes_jsonl())
        
        if not episodes:
            return {"num_episodes": 0}
        
        total_turns = sum(len(ep["turns"]) for ep in episodes)
        total_correct = sum(1 for ep in episodes if ep["is_correct"])
        
        return {
            "num_episodes": len(episodes),
            "total_turns": total_turns,
            "avg_turns_per_episode": total_turns / len(episodes),
            "accuracy": total_correct / len(episodes),
            "total_correct": total_correct,
            "jsonl_path": str(self.jsonl_path),
            "tensors_dir": str(self.tensors_dir)
        }
