import numpy as np

WRIST_IDX = 0
MIDDLE_MCP_IDX = 9

def normalize_hand(hand: np.ndarray) -> np.ndarray:
    wrist = hand[WRIST_IDX].copy()
    translated = hand - wrist  

    scale = np.linalg.norm(translated[MIDDLE_MCP_IDX]) + 1e-6
    return translated / scale