from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

segments = ['Baseline', 'Cue1', 'Gap1', 'Cue2', 'Gap2', 'Digit1', 'Gap3',
            'Digit2', 'Gap4', 'Resp']
onset_ms = [-200, 0, 200, 800, 1000, 1600, 1950, 2750, 3100, 3600]
duration_ms = [200, 200, 600, 200, 600, 350, 800, 350, 500, 1500]

condition_map = {'LL': ['LL3', 'LL4', 'LLV', 'RR3', 'RR4'],
                 'LR': ['LR3', 'LR4', 'RL3', 'RL4'],
                 'LX': ['LU3', 'LU4', 'LUV', 'LD3', 'LD4',
                        'RU3', 'RU4', 'RD3', 'RD4'],
                 'UU': ['UU3', 'UU4', 'UUV', 'DD3', 'DD4'],
                 'UD': ['UD3', 'UD4', 'DU3', 'DU4'],
                 'UX': ['UL3', 'UL4', 'ULV', 'UR3', 'UR4',
                        'DL3', 'DL4', 'DR3', 'DR4']
                }
            
