import torch
import torch.nn.functional as F


def create_replay_strategy_function(strategy, value):
    if strategy == 'every_token':
        def replay_every_token(logits, offset):
            return torch.ones(logits.shape[0], dtype=torch.bool, device=logits.device)
        return replay_every_token
    if strategy == 'every_x_tokens':
        def replay_every_x_tokens(logits, offset):
            replay_mask = torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
            if offset % value == 0:
                replay_mask.fill_(1)
            return replay_mask
        return replay_every_x_tokens
    if strategy == 'entropy':
        def replay_based_on_entropy(logits, offset):
            log_p = F.log_softmax(logits, dim=-1)
            p = F.softmax(logits, dim=-1)
            entropy = -(p * log_p).sum(dim=-1)
            return entropy > value
        return replay_based_on_entropy
    if strategy == 'probability':
        def replay_based_on_probability(logits, offset):
            p = F.softmax(logits, dim=-1).argmax(dim=-1)
            return p < value
        return replay_based_on_probability
    if strategy == 'never':
        def return_false(logits, offset):
            return torch.zeros(logits.shape[0], dtype=torch.bool, device=logits.device)
        return return_false
    raise Exception(f'Not supported strategy: {strategy}')
        

def insert_landmark_for_inference(input_ids, offset, chunk_size, lmk_id):
    """
    input_ids: (N, L)
    """
    left_pad = offset % chunk_size
    right_pad = chunk_size - ((input_ids.shape[1] + left_pad) % chunk_size)
    right_pad = right_pad % chunk_size

    input_ids_pad = F.pad(input_ids, (left_pad, right_pad), value=0)
    assert input_ids_pad.shape[1] % chunk_size == 0
    chunk_num = input_ids_pad.shape[1] // chunk_size

    N = input_ids.shape[0]
    L = input_ids.shape[1]
    input_ids_pad = input_ids_pad.view(N, -1, chunk_size)
    pad_ids = torch.zeros(N, input_ids_pad.shape[1], 1, dtype=input_ids.dtype, device=input_ids.device)
    pad_ids.fill_(lmk_id)
    # print(f'pad ids shape: {pad_ids.shape}')
    # print(f'input_ids_pad: {input_ids_pad.shape}')
    input_ids_ = torch.cat([pad_ids, input_ids_pad], dim=-1)
    # padded ids: pad pad x x | x x x x | x x x pad
    # corner cases: | x x x x | x x x pad   // if left_pad == 0, additionally insert a landmark
    left_truncate = 0 if left_pad == 0 else left_pad + 1
    # print(f'input_ids_ shape: {input_ids_.shape}, range: {left_truncate}: {-right_pad}')
    if right_pad > 0:
        input_ids_ = input_ids_.view(N, -1)[:, left_truncate:-right_pad]
    else:
        input_ids_ = input_ids_.view(N, -1)[:, left_truncate:]

    insert_positions = []
    for chunk_i in range(0, chunk_num + 1):
        if chunk_i * chunk_size >= left_pad and chunk_i * chunk_size < left_pad + L:
            insert_positions.append(chunk_i * (1 + chunk_size) - left_pad)

    print(f'insert positions: {insert_positions}')
    assert input_ids_.shape[1] == input_ids.shape[1] + len(insert_positions), f'{input_ids_.shape[1]} != {input_ids.shape[1] + len(insert_positions)}'    

    if len(insert_positions) > 0:
        insert_positions = torch.tensor(insert_positions, device=input_ids.device)
        assert torch.all(torch.gather(input_ids_, dim=1, index=insert_positions.unsqueeze(0)) == lmk_id)
    else:
        insert_positions = None

    return input_ids_, insert_positions