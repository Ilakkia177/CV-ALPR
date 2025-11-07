from collections import Counter

plate_history = []
prev_stabilized = None
no_detect_count = 0          # counter for consecutive frames with no detection
NO_DETECT_LIMIT = 5          # reset history after 5 consecutive empty frames

def get_stabilized_plate(history):
    """
    Function to stabilize the recognized license plate text over multiple frames.
    Purpose:
        - Character recognition can fluctuate slightly frame-to-frame.
        - This method finds the most consistent character per position to form a stable result.
        - Automatically resets history if no valid detection for several frames (new car likely).
    """
    global plate_history, no_detect_count, prev_stabilized

    # if no new detection in current frame, increase no-detect counter
    if not history or not history[-1]:
        no_detect_count += 1

        # reset if too many empty frames (new car entering or scene change)
        if no_detect_count >= NO_DETECT_LIMIT:
            plate_history.clear()
            prev_stabilized = None
            no_detect_count = 0
        return prev_stabilized

    # if detection present, reset no-detect counter
    no_detect_count = 0

    # compute max length of detected plates in history
    max_len = max(len(p) for p in history)
    counts = [Counter() for _ in range(max_len)]

    # count occurrences of each character in same index across frames
    for p in history:
        for i, ch in enumerate(p):
            counts[i][ch] += 1

    # build the most frequent characters into final stabilized plate
    final = ""
    for c in counts:
        if not c:
            continue
        final += c.most_common(1)[0][0]

    prev_stabilized = final
    return final
