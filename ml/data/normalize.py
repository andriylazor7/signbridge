import numpy as np

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

def normalize_hand(hand: np.ndarray) -> np.ndarray:
    wrist = hand[WRIST_IDX].copy()
    translated = hand - wrist

    scale = np.linalg.norm(translated[MIDDLE_MCP_IDX]) + 1e-6
    return translated / scale


def normalize_sequence(landmarks: np.ndarray) -> np.ndarray:
    """Normalize a (T, num_hands, 21, 3) landmark sequence: each hand at each
    frame becomes wrist-relative and scaled by its middle-MCP distance."""
    wrist = landmarks[:, :, WRIST_IDX:WRIST_IDX + 1, :]
    translated = landmarks - wrist

    scale = np.linalg.norm(translated[:, :, MIDDLE_MCP_IDX, :], axis=-1) + 1e-6
    return translated / scale[:, :, None, None]