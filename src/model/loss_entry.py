import torch.nn.functional as F

def ctc_loss(output_log_softmax, target, input_lengths, target_lengths):
    return F.ctc_loss(output_log_softmax, target, input_lengths, target_lengths)
