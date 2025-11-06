from collections import Counter

plate_history = []
prev_stabilized = None

def get_stabilized_plate(history):

    """
    Function to stabilize the recognized license plate text over multiple frames.
    Purpose:
        - Character recognition can fluctuate slightly frame-to-frame.
        - This method finds the most consistent character per position to form a stable result.
    """
    if not history:
        return None
    max_len = max(len(p) for p in history)

    # Counter objects — one for each char idx,each Counter will store freq of detected char at that idx
    counts = [Counter() for _ in range(max_len)]

    # count occurrences of each character in the same position across all detected plates- typically 8 frames behind
    for p in history:
        for i, ch in enumerate(p):
            counts[i][ch] += 1

    # get final stable plate text        
    final = ""
    for c in counts:
        if not c:
            # skip idx where no char found
            continue

        # take most occuring char at that idx
        final += c.most_common(1)[0][0]
        
    return final
