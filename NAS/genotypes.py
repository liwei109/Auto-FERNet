from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5',
    'ghost',
    'conv_1x1',
    'conv_3x3',
]

PARAMS = {'none': 0, 'avg_pool_3x3': 0, 'max_pool_3x3': 0, 'skip_connect': 0, 'sep_conv_3x3': 504, 'sep_conv_5x5': 888,
          'sep_conv_7x7': 1464, 'dil_conv_3x3': 252, 'dil_conv_5x5': 444, 'conv_7x1_1x7': 2016, 'conv_9x9': 11664,
          'ghost': 126, 'conv_1x1': 144, 'conv_3x3': 1296}

#   anger     disgust      fear      happy      sad       surprised     neutral
FES = [[1., 0.02718851, 0.02188297, 0.02002968, 0.04878107, 0.03223266, 0.05837538],
       [0.02718851, 1., 0.00951736, 0.01147336, 0.02775054, 0.01381716, 0.03206117],
       [0.02188297, 0.00951736, 1., 0.01524509, 0.02792287, 0.04619572, 0.03383849],
       [0.02002968, 0.01147336, 0.01524509, 1., 0.02575007, 0.03452561, 0.05580243],
       [0.04878107, 0.02775054, 0.02792287, 0.02575007, 1., 0.03472185, 0.13838511],
       [0.03223266, 0.01381716, 0.04619572, 0.03452561, 0.03472185, 1., 0.05298048],
       [0.05837538, 0.03206117, 0.03383849, 0.05580243, 0.13838511, 0.05298048, 1.]]

# search space
FER2013_search_space = Genotype(
    normal=[('sep_conv_3x3', 0), ('conv_3x3', 1), ('dil_conv_5x5', 0), ('conv_1x1', 2), ('ghost', 2),
            ('dil_conv_5x5', 3), ('skip_connect', 0), ('conv_1x1', 1)], normal_concat=range(2, 6),
    reduce=[('max_pool_3x3', 0), ('max_pool_3x3', 1), ('max_pool_3x3', 0), ('dil_conv_5x5', 2), ('avg_pool_3x3', 1),
            ('skip_connect', 2), ('avg_pool_3x3', 0), ('sep_conv_3x3', 2)], reduce_concat=range(2, 6))

Auto_FERNet = FER2013_search_space
