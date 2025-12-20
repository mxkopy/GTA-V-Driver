state_sizes = {
    'image': (1, 360, 640),
    'velocity': (3,),
    'damage': (1,)
}

action_sizes = {
    'controller': (4,),
}

visual_features_size = (8, 8)
visual_channels = [state_sizes['image'][0], 3, 3, 3, 3, 3]
device = 'cuda'