import torch
import torch.nn.functional as F
from typing import Optional

class ThoughtFilter:
    """
    Intelligently filters activations to determine if they should be translated.
    Uses a hybrid approach of structural (punctuation/spaces) and semantic (cosine similarity) triggers.
    """
    def __init__(self, cosine_threshold: float = 0.95):
        self.cosine_threshold = cosine_threshold
        self.last_translated_activation: Optional[torch.Tensor] = None
        self.structural_chars = set(" .,!?\n\t")

    def should_translate(self, token: str, activation: torch.Tensor) -> bool:
        """
        Determines if the current token/activation pair warrants a translation.
        """
        # 1. Structural Trigger: End of word or sentence
        if any(c in self.structural_chars for c in token):
            self.last_translated_activation = activation.detach().clone()
            return True

        # 2. Semantic Trigger: Concept shift detected via Cosine Similarity
        if self.last_translated_activation is None:
            self.last_translated_activation = activation.detach().clone()
            return True

        # Compute cosine similarity
        sim = F.cosine_similarity(
            activation.view(1, -1), 
            self.last_translated_activation.view(1, -1)
        ).item()

        if sim < self.cosine_threshold:
            self.last_translated_activation = activation.detach().clone()
            return True

        return False

    def reset(self):
        self.last_translated_activation = None

### ULTRA DETAILED CODE EXPLANATION ###
#
# This file (`filter.py`) defines the `ThoughtFilter` class, which optimizes performance by reducing the number 
# of verbalization requests sent to the NLA Verbalizer. It ensures that only meaningful or "new" concepts 
# trigger a "thought" explanation.
#
# --- Imports ---
# - `torch`: PyTorch, used for tensor manipulation.
# - `torch.nn.functional`: Used to calculate cosine similarity (`F.cosine_similarity`).
# - `typing.Optional`: Used for type hinting variables that can be `None`.
#
# --- Class: ThoughtFilter ---
#
# 1. `__init__(self, cosine_threshold=0.95)`
#    - **Purpose**: Initializes the filter with a specific sensitivity.
#    - **Parameters**: 
#        - `cosine_threshold` (float): The similarity limit. If the current activation is more than 95% similar 
#          to the last one we translated, we skip it (unless it's structural).
#    - **Variables**:
#        - `self.last_translated_activation`: (type: `Optional[torch.Tensor]`) Stores the vector of the last 
#          activation that was actually sent to the verbalizer.
#        - `self.structural_chars`: (type: `set[str]`) A set of characters that signify the end of a word, 
#          sentence, or line. These act as "structural triggers."
#
# 2. `should_translate(self, token, activation)`
#    - **Purpose**: Decides whether to verbalize the current neural state.
#    - **Parameters**:
#        - `token` (str): The text token being generated (e.g., " hello").
#        - `activation` (torch.Tensor): The internal activation vector associated with that token.
#    - **Logic Flow**:
#        - **Step 1 (Structural Check)**: If the `token` contains any character from `structural_chars` 
#          (like a space or a period), it returns `True`. This ensures we always get a thought at the end of a word.
#        - **Step 2 (Initialization)**: If no activation has been translated yet (`last_translated_activation is None`), 
#          it stores the current one and returns `True`.
#        - **Step 3 (Semantic Check)**: It calculates the cosine similarity between the current `activation` 
#          and the `last_translated_activation`.
#        - **Step 4 (Decision)**: If the similarity is *less than* the `cosine_threshold`, it means the model's 
#          internal state has "shifted" significantly (a new concept). It updates `last_translated_activation` 
#          and returns `True`.
#        - Otherwise, it returns `False`.
#
# 3. `reset(self)`
#    - **Purpose**: Clears the state, usually called at the start of a new generation.
#    - **Logic**: Sets `self.last_translated_activation` to `None`.
#
