"""
Markov Model for Ballet Movement Transition Smoothing

This module implements a Markov model to smooth frame-level predictions
by learning transition probabilities between ballet movements.

How it works with SVM:
1. SVM makes frame-level predictions (each frame independently)
2. Markov model learns transition probabilities from training sequences
3. During inference, Markov model smooths predictions using temporal context
4. Improves accuracy by leveraging ballet movement patterns
"""

import numpy as np
import pandas as pd
from collections import defaultdict, Counter
from typing import List, Dict, Tuple, Optional
import pickle


class MovementMarkovModel:
    """
    Markov model for smoothing ballet movement predictions.
    
    Learns transition probabilities P(movement_t | movement_{t-1}) from training data
    and uses them to smooth frame-level predictions.
    """
    
    def __init__(self, smoothing_factor: float = 0.1):
        """
        Initialize the Markov model.
        
        Args:
            smoothing_factor: Laplace smoothing factor (additive smoothing)
                             Higher = more uniform transitions, Lower = more data-driven
        """
        self.smoothing_factor = smoothing_factor
        self.transition_counts = defaultdict(Counter)  # from_state -> {to_state: count}
        self.initial_counts = Counter()  # First movement in sequences
        self.state_set = set()
        self.transition_matrix = None
        self.initial_probs = None
        self.is_fitted = False
    
    def fit(self, sequences: List[List[str]]):
        """
        Learn transition probabilities from sequences of movements.
        
        Args:
            sequences: List of movement sequences, where each sequence is a list of movement labels
                      Example: [['first position', 'demi plie', 'tendu'], ...]
        """
        print(f"\nTraining Markov model on {len(sequences)} sequences...")
        
        # Count transitions
        for sequence in sequences:
            if len(sequence) == 0:
                continue
            
            # Track initial state
            self.initial_counts[sequence[0]] += 1
            self.state_set.add(sequence[0])
            
            # Count transitions
            for i in range(len(sequence) - 1):
                from_state = sequence[i]
                to_state = sequence[i + 1]
                self.transition_counts[from_state][to_state] += 1
                self.state_set.add(from_state)
                self.state_set.add(to_state)
        
        # Build transition matrix
        states = sorted(list(self.state_set))
        n_states = len(states)
        self.state_to_idx = {state: i for i, state in enumerate(states)}
        self.idx_to_state = {i: state for i, state in enumerate(states)}
        
        # Initialize transition matrix with smoothing
        self.transition_matrix = np.ones((n_states, n_states)) * self.smoothing_factor
        
        # Fill in observed transitions
        for from_state, to_state_counts in self.transition_counts.items():
            if from_state in self.state_to_idx:
                from_idx = self.state_to_idx[from_state]
                total = sum(to_state_counts.values())
                for to_state, count in to_state_counts.items():
                    if to_state in self.state_to_idx:
                        to_idx = self.state_to_idx[to_state]
                        self.transition_matrix[from_idx, to_idx] += count
        
        # Normalize rows (each row sums to 1)
        row_sums = self.transition_matrix.sum(axis=1, keepdims=True)
        self.transition_matrix = self.transition_matrix / row_sums
        
        # Compute initial probabilities
        total_initial = sum(self.initial_counts.values())
        self.initial_probs = np.zeros(n_states)
        for state, count in self.initial_counts.items():
            if state in self.state_to_idx:
                idx = self.state_to_idx[state]
                self.initial_probs[idx] = (count + self.smoothing_factor) / (total_initial + n_states * self.smoothing_factor)
        
        # Normalize initial probabilities
        self.initial_probs = self.initial_probs / self.initial_probs.sum()
        
        self.is_fitted = True
        
        print(f"  Learned transitions for {n_states} movement types")
        print(f"  Total transitions observed: {sum(sum(c.values()) for c in self.transition_counts.values())}")
    
    def fit_from_frame_predictions(self, frame_numbers: List[int], predictions: List[str]):
        """
        Learn transition probabilities from frame-level predictions.
        
        Args:
            frame_numbers: List of frame numbers (must be sorted)
            predictions: List of predicted labels for each frame
        """
        # Sort by frame number
        sorted_data = sorted(zip(frame_numbers, predictions), key=lambda x: x[0])
        frame_nums, preds = zip(*sorted_data) if sorted_data else ([], [])
        
        # Group into sequences (handle gaps in frame numbers)
        sequences = []
        current_sequence = []
        prev_frame = None
        
        for frame_num, pred in zip(frame_nums, preds):
            if prev_frame is not None and frame_num - prev_frame > 5:  # Gap > 5 frames = new sequence
                if len(current_sequence) > 0:
                    sequences.append(current_sequence)
                current_sequence = [pred]
            else:
                current_sequence.append(pred)
            prev_frame = frame_num
        
        if len(current_sequence) > 0:
            sequences.append(current_sequence)
        
        self.fit(sequences)
    
    def smooth_predictions(self, predictions: List[str], method: str = 'viterbi') -> List[str]:
        """
        Smooth predictions using Markov model.
        
        Args:
            predictions: List of frame-level predictions
            method: 'viterbi' (optimal path) or 'greedy' (local smoothing)
        
        Returns:
            Smoothed predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before smoothing predictions")
        
        if len(predictions) == 0:
            return predictions
        
        # Convert predictions to indices
        pred_indices = []
        for pred in predictions:
            if pred in self.state_to_idx:
                pred_indices.append(self.state_to_idx[pred])
            else:
                # Unknown state - use most common state or first state
                pred_indices.append(0)
        
        if method == 'viterbi':
            return self._viterbi_smoothing(pred_indices)
        elif method == 'greedy':
            return self._greedy_smoothing(pred_indices)
        else:
            raise ValueError(f"Unknown smoothing method: {method}")
    
    def _greedy_smoothing(self, pred_indices: List[int]) -> List[str]:
        """
        Greedy smoothing: adjust each prediction based on previous prediction.
        Faster but suboptimal.
        """
        smoothed = [pred_indices[0]]
        
        for i in range(1, len(pred_indices)):
            prev_idx = smoothed[i - 1]
            current_idx = pred_indices[i]
            
            # Get transition probability
            transition_prob = self.transition_matrix[prev_idx, current_idx]
            
            # If transition is unlikely, keep previous state
            # (simple heuristic - could be improved)
            if transition_prob < 0.1:  # Threshold
                smoothed.append(prev_idx)
            else:
                smoothed.append(current_idx)
        
        return [self.idx_to_state[idx] for idx in smoothed]
    
    def _viterbi_smoothing(self, pred_indices: List[int]) -> List[str]:
        """
        Viterbi algorithm: find most likely sequence given observations.
        Optimal but more computationally expensive.
        """
        n = len(pred_indices)
        n_states = len(self.state_to_idx)
        
        # DP table: dp[i][j] = max probability of state j at position i
        dp = np.zeros((n, n_states))
        backpointer = np.zeros((n, n_states), dtype=int)
        
        # Initialize first position
        for j in range(n_states):
            # Combine initial prob with observation (simplified - could use emission probs)
            dp[0][j] = np.log(self.initial_probs[j] + 1e-10)
            if j == pred_indices[0]:
                dp[0][j] += 2.0  # Strong boost if matches observation (trust SVM more)
            else:
                dp[0][j] -= 1.0  # Penalty for not matching observation
        
        # Fill DP table
        for i in range(1, n):
            for j in range(n_states):
                best_prob = float('-inf')
                best_prev = 0
                
                for k in range(n_states):
                    # Transition from k to j
                    trans_prob = np.log(self.transition_matrix[k, j] + 1e-10)
                    prob = dp[i - 1][k] + trans_prob
                    
                    if prob > best_prob:
                        best_prob = prob
                        best_prev = k
                
                # Add observation likelihood (simplified)
                # Trust SVM predictions more - only use transitions to break ties or fix obvious errors
                if j == pred_indices[i]:
                    best_prob += 2.0  # Strong boost for matching SVM prediction
                else:
                    best_prob -= 1.0  # Penalty for disagreeing with SVM
                
                dp[i][j] = best_prob
                backpointer[i][j] = best_prev
        
        # Backtrack to find best path
        best_path = [0] * n
        best_path[n - 1] = np.argmax(dp[n - 1])
        
        for i in range(n - 2, -1, -1):
            best_path[i] = backpointer[i + 1][best_path[i + 1]]
        
        return [self.idx_to_state[idx] for idx in best_path]
    
    def get_transition_probability(self, from_state: str, to_state: str) -> float:
        """Get transition probability between two states."""
        if not self.is_fitted:
            return 0.0
        
        if from_state not in self.state_to_idx or to_state not in self.state_to_idx:
            return 0.0
        
        from_idx = self.state_to_idx[from_state]
        to_idx = self.state_to_idx[to_state]
        return self.transition_matrix[from_idx, to_idx]
    
    def get_top_transitions(self, from_state: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """Get top K most likely transitions from a given state."""
        if not self.is_fitted or from_state not in self.state_to_idx:
            return []
        
        from_idx = self.state_to_idx[from_state]
        probs = self.transition_matrix[from_idx, :]
        
        top_indices = np.argsort(probs)[::-1][:top_k]
        return [(self.idx_to_state[idx], probs[idx]) for idx in top_indices]
    
    def save(self, filepath: str):
        """Save the model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'transition_matrix': self.transition_matrix,
                'initial_probs': self.initial_probs,
                'state_to_idx': self.state_to_idx,
                'idx_to_state': self.idx_to_state,
                'smoothing_factor': self.smoothing_factor,
                'is_fitted': self.is_fitted
            }, f)
    
    def load(self, filepath: str):
        """Load the model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.transition_matrix = data['transition_matrix']
            self.initial_probs = data['initial_probs']
            self.state_to_idx = data['state_to_idx']
            self.idx_to_state = data['idx_to_state']
            self.smoothing_factor = data['smoothing_factor']
            self.is_fitted = data['is_fitted']


def fit_markov_from_training_data(frame_info: List[Tuple[int, str]], 
                                   labels: List[str]) -> MovementMarkovModel:
    """
    Fit Markov model from training data with frame information.
    
    IMPORTANT: This function should be called with TRAINING DATA LABELS (ground truth),
    not with predictions. The Markov model learns the true movement transition patterns
    from the training sequences, then these patterns are used to smooth predictions
    on test data.
    
    Args:
        frame_info: List of (frame_number, camera) tuples from TRAINING data
        labels: List of GROUND TRUTH labels from training set (not predictions!)
    
    Returns:
        Fitted Markov model that has learned transition probabilities from training data
    """
    # Sort by frame number
    sorted_data = sorted(zip(frame_info, labels), key=lambda x: x[0][0])
    
    # Extract sequences
    sequences = []
    current_sequence = []
    prev_frame = None
    
    for (frame_num, camera), label in sorted_data:
        if prev_frame is not None and frame_num - prev_frame > 5:  # Gap = new sequence
            if len(current_sequence) > 0:
                sequences.append(current_sequence)
            current_sequence = [label]
        else:
            current_sequence.append(label)
        prev_frame = frame_num
    
    if len(current_sequence) > 0:
        sequences.append(current_sequence)
    
    model = MovementMarkovModel()
    model.fit(sequences)
    return model

