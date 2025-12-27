def pad_sequences(seqs, max_len):
    padded = []
    for seq in seqs:
        if len(seq) < max_len:
            seq = seq + [0] * (max_len - len(seq))
        else:
            seq = seq[:max_len]
        padded.append(seq)
    return padded
