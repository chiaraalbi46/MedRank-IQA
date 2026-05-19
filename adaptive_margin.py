""" Adaptive margin computation and util functions related. """

import torch
import re

# def adaptive_ranking_loss(
#     s,
#     s_hat,
#     y,
#     severity_i,
#     severity_j,
#     m0=0.1,
#     alpha=0.3,
#     gamma=2.0
# ):

def adaptive_ranking_loss(
    s,
    s_hat,
    y,
    severity_i,
    severity_j,
    m0=0.0,
    alpha=2.0,
    gamma=2.0
):

    diff = s - s_hat
    target = (1 - 2 * y)  # y=1 -> target=-1 | y=0 -> target=1 

    # severity distance
    delta = torch.abs(severity_i - severity_j)

    # adaptive margin
    # margin = m0 + alpha * torch.sqrt(delta)
    # margin = m0 + alpha * delta

    # adaptive weight
    #weight = torch.exp(-gamma * delta)

    # hinge ranking loss
    # loss = torch.clamp(((2*y - 1) * diff) + margin, min=0)

    # weighted loss
    # loss = weight * loss

    #########
    # y=1 se i è meglio di j (severity_i < severity_j)
    # target = (2 * y - 1) 
    # diff = s - s_hat
    
    # Invece del margine rigido, usiamo una sigmoid scalata dal delta
    # Se delta è grande, la pendenza della logistica è più aggressiva
    # delta = torch.abs(sev_i - sev_j)
    
    # La log-loss non si ferma mai del tutto, spinge sempre per aumentare la separazione
    # loss = torch.log(1 + torch.exp(-alpha * delta * target * diff))

    # loss = torch.nn.functional.softplus(-target * diff * (1 + delta))

    ############
    expo = alpha * delta * target * diff
    loss = torch.log(1 + torch.exp(expo))

    return loss.mean()

def compute_severity(meta, distortion_range, real_arts_levels, real_streak_levels, real_noise_levels, device):

    level_i = meta['level_i']  # lista di level_i 
    level_j = meta['level_j']

    level_idx_i = meta['level_idx_i'] 
    level_idx_j = meta['level_idx_j']

    art_i = meta['art_i']
    art_j = meta['art_j']

    batch_size = len(level_i)

    severity_i = torch.zeros(batch_size, 1, device=device)
    severity_j = torch.zeros(batch_size, 1, device=device)

    for b in range(batch_size):

        if meta['type'][b] == 'syn_syn' or meta['type'][b] == 'fd_syn':
            # image i
            if level_idx_i[b] == -1:
                severity_i[b] = 0.0  # non importerebbe , facendo level_i + 1 fa 0 e quindi severity_i = 0
            else:
                n_levels_i = len(distortion_range[art_i[b]])
                severity_i[b] = (int(level_idx_i[b]) + 1) / n_levels_i

            # image j
            if level_idx_j[b] == -1:
                severity_j[b] = 0.0
            else:
                n_levels_j = len(distortion_range[art_j[b]])
                severity_j[b] = (int(level_idx_j[b]) + 1) / n_levels_j

        elif meta['type'][b] == 'fd_ld':
            if level_idx_i[b] == -1:
                severity_i[b] = 0.0  
                severity_j[b] = 0.5
            else:
                severity_i[b] = 0.5  
                severity_j[b] = 0.0

        elif meta['type'][b] == 'fd_real':
            # image i
            if level_idx_i[b] == -1:
                severity_i[b] = 0.0  # non importerebbe , facendo level_i + 1 fa 0 e quindi severity_i = 0
            else:
                n_levels_i = len(real_arts_levels)
                severity_i[b] = (int(level_idx_i[b]) + 1) / n_levels_i

            # image j
            if level_idx_j[b] == -1:
                severity_j[b] = 0.0
            else:
                n_levels_j = len(real_arts_levels)
                severity_j[b] = (int(level_idx_j[b]) + 1) / n_levels_j

        elif meta['type'][b] == 'real_real':
            # devo distinguere per streak e noise
            pattern = re.compile(r"streak_([\d\.]+)_noise_([\d\.]+)")
            match_i = pattern.search(art_i[b])
            streak_i, noise_i = int(match_i.group(1)), float(match_i.group(2))

            # Estrazione valori per l'immagine j
            match_j = pattern.search(art_j[b])
            streak_j, noise_j = int(match_j.group(1)), float(match_j.group(2))

            # Confronto efficiente
            diff_streak = streak_i != streak_j

            if diff_streak:
                # stesso noise, diverso streak
                n_levels = len(real_streak_levels)
                severity_i[b] = (int(level_idx_i[b]) + 1) / n_levels
                severity_j[b] = (int(level_idx_j[b]) + 1) / n_levels


            else:
                n_levels = len(real_noise_levels)
                severity_i[b] = (int(level_idx_i[b]) + 1) / n_levels
                severity_j[b] = (int(level_idx_j[b]) + 1) / n_levels

        else:
            ValueError(f"Unknown pair type: {meta['type'][b]}")

    return severity_i, severity_j

# m0 = 0.1
# alpha = 0.3
# gamma = 2

# indice = random.randrange(len(lista))
# elemento = lista[indice]

# def compute_severity(meta, distortion_range):
#     # dato l'indice dell'artefatto, lo rimappa sul range corrispondente e calcola la severity
#     # a seconda del tipo di artefatto sarà level_idx / n_levels-1 o 1 - (level_idx / n_levels-1)

#     pair_type = meta['type']

#     if pair_type == 'fd_syn':
#         art = meta['art']
#         n_levels = distortion_range[art]
#         level_idx = meta['level_idx'] 

#         severity_i = 0 ## originale
#         severity_j = (level_idx + 1) / n_levels

#     elif pair_type == 'syn_syn':
#         art = meta['art1']
#         level_idx_1 = meta['level_idx_1']
#         level_idx_2 = meta['level_idx_2']
        
#         severity_i = (level_idx_1 + 1) / n_levels
#         severity_j = (level_idx_2 + 1) / n_levels
#     else:
#         pass

#     return severity_i, severity_j

# def compute_severity(meta, distortion_range, device):

#     level_i = meta['level_i']  # lista di level_i 
#     level_j = meta['level_j']

#     level_idx_i = meta['level_idx_i'] 
#     level_idx_j = meta['level_idx_j']

#     art_i = meta['art_i']
#     art_j = meta['art_j']

#     batch_size = len(level_i)

#     severity_i = torch.zeros(batch_size, 1, device=device)
#     severity_j = torch.zeros(batch_size, 1, device=device)

#     for b in range(batch_size):

#         # image i
#         if level_idx_i[b] == -1:
#             severity_i[b] = 0.0  # non importerebbe , facendo level_i + 1 fa 0 e quindi severity_i = 0
#         else:
#             n_levels_i = len(distortion_range[art_i[b]])
#             severity_i[b] = (int(level_idx_i[b]) + 1) / n_levels_i

#         # image j
#         if level_idx_j[b] == -1:
#             severity_j[b] = 0.0
#         else:
#             n_levels_j = len(distortion_range[art_j[b]])
#             severity_j[b] = (int(level_idx_j[b]) + 1) / n_levels_j

#     return severity_i, severity_j