from collections import Counter

plate_history = []
prev_stabilized = None

def get_stabilized_plate(history):
    if not history:
        return None
    max_len = max(len(p) for p in history)
    counts = [Counter() for _ in range(max_len)]
    for p in history:
        for i, ch in enumerate(p):
            counts[i][ch] += 1
    final = ""
    for c in counts:
        if not c:
            continue
        final += c.most_common(1)[0][0]
    return final
