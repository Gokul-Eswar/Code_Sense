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
