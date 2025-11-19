state_sizes = {
    'image': (3, 360, 640),
    'controller': (4,),
    'camera_direction': (3,),
    'velocity': (3,),
    # 'forward_direction': (3,),
    'damage': (1,)
}

action_sizes = {
    'controller': state_sizes['controller']
}

visual_features_size = (8, 8)
visual_channels = [3, 3, 3, 3, 3, 3]
device = 'cuda'