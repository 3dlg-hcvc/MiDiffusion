import numpy as np
import torch
from torch.utils.data import dataloader
from tqdm import tqdm

from midiffusion.networks.diffusion_scene_layout_ddpm import DiffusionSceneLayout_DDPM
from midiffusion.networks.diffusion_scene_layout_mixed import DiffusionSceneLayout_Mixed
from midiffusion.datasets.threed_front_encoding import Diffusion


def scale(x, minimum, maximum):
    """Scale the input to [-1, 1] range"""
    X = x.astype(np.float32)
    X = np.clip(X, minimum, maximum)
    X = ((X - minimum) / (maximum - minimum))
    X = 2 * X - 1
    return X

def generate_layouts(network:DiffusionSceneLayout_DDPM, encoded_dataset:Diffusion, 
                     config, num_syn_scenes, sampling_rule="random", 
                     experiment="synthesis", num_known_objects=0, 
                     batch_size=16, device="cpu", room_type_context=None,
                     custom_floorplan=None):
    """Generate specified number of object layouts using either dataset floorplans
    or a custom floorplan set."""
    
    # Sample floor layout or use custom floorplan
    if custom_floorplan is not None:
        # Use all provided floorplans
        room_feature = custom_floorplan
        # Use sequential indices for all floorplans
        sampled_indices = np.arange(num_syn_scenes).tolist()
    else:
        # Original sampling logic
        if sampling_rule == "random":
            sampled_indices = np.random.choice(len(encoded_dataset), num_syn_scenes).tolist()
        elif sampling_rule == "uniform":
            sampled_indices = np.arange(len(encoded_dataset)).tolist() * \
                (num_syn_scenes // len(encoded_dataset))
            sampled_indices += \
                np.random.choice(len(encoded_dataset), 
                                 num_syn_scenes - len(sampled_indices)).tolist()
        else:
            raise NotImplemented
    
    # network params
    with_room_mask = config["network"].get("room_mask_condition", True)

    # experiment setup
    if experiment == "synthesis":
        feature_mask = None
        print("Experiment: scene synthesis.")
    elif experiment == "scene_completion":
        assert num_known_objects > 0
        feature_mask = torch.zeros(
            (network.sample_num_points, network.point_dim + network.class_dim),
            dtype=torch.bool, device=device
        )
        feature_mask[:num_known_objects] = True
        print("Experiment: scene completion (given {} objects) using corruption-and-masking."\
              .format(num_known_objects))
    elif experiment == "furniture_arrangement":
        feature_mask = torch.zeros(
            network.point_dim + network.class_dim, dtype=torch.bool, device=device
        )
        feature_mask[network.translation_dim: 
                     network.translation_dim + network.size_dim] = True # size
        feature_mask[network.bbox_dim: 
                     network.bbox_dim + network.class_dim] = True       # class
        feature_mask = feature_mask.repeat(network.sample_num_points, 1)
        print("Experiment: furniture arrangement.")
    elif experiment == "object_conditioned":
        feature_mask = torch.zeros(
            network.sample_num_points, network.point_dim + network.class_dim, 
            dtype=torch.bool, device=device
        )
        feature_mask[:, network.bbox_dim: network.bbox_dim + network.class_dim] = True       # class
        print("Experiment: object conditioned synthesis using corruption-and-masking.")
    elif experiment == "scene_completion_conditioned":
        feature_mask = torch.zeros(
            network.sample_num_points, network.point_dim + network.class_dim, 
            dtype=torch.bool, device=device
        )
        feature_mask[:num_known_objects] = True     # existing objects
        feature_mask[:, network.bbox_dim: network.bbox_dim + network.class_dim] = True       # class
        print("Experiment: scene completion (given {} objects) conditioned on labels using corruption-and-masking."\
              .format(num_known_objects))
    else:
        raise NotImplemented
    print("Floor condition: {}.".format(with_room_mask))
    
    # Generate layouts
    network.to(device)
    network.eval()
    layout_list = []
    for i in tqdm(range(0, num_syn_scenes, batch_size)):
        scene_indices = sampled_indices[i: min(i + batch_size, num_syn_scenes)]
        
        if custom_floorplan is not None:
            # Get floor plan boundary points and normals
            floor_points = np.stack([
                custom_floorplan[ind]["floor_plan_boundary_points_normals"] 
                for ind in scene_indices
            ], axis=0)

            room_side = 6
            max_bounds = np.array([room_side,room_side,1,1])
            min_bounds = np.array([-room_side,-room_side,-1,-1])
            # scale points
            floor_points_scaled = scale(floor_points, min_bounds, max_bounds).astype(np.float32)  # Adjust min/max based on your training data
            # Convert to tensor
            current_room_feature = torch.from_numpy(floor_points_scaled).to(device)
            # breakpoint()
        else:
            # Original room feature loading logic
            if with_room_mask:
                if config["feature_extractor"]["name"] == "resnet18":
                    current_room_feature = torch.from_numpy(np.stack([
                        encoded_dataset[ind]["room_layout"] for ind in scene_indices
                    ], axis=0)).to(device)
                elif config["feature_extractor"]["name"] == "pointnet_simple":
                    current_room_feature = torch.from_numpy(np.stack([
                        encoded_dataset[ind]["fpbpn"] for ind in scene_indices
                    ], axis=0)).to(device)
        
        if experiment == "synthesis":
            input_boxes = None
        else:
            samples = list(encoded_dataset[ind] for ind in scene_indices)
            sample_params = dataloader.default_collate(samples)
            input_boxes = network.unpack_data(sample_params).to(device)

        bbox_params_list = network.generate_layout(
            room_feature=current_room_feature,
            batch_size=len(scene_indices),
            input_boxes=input_boxes,
            feature_mask=feature_mask,
            device=device,
            room_type_context=room_type_context
        )
        for bbox_params_dict in bbox_params_list:
            boxes = encoded_dataset.post_process(bbox_params_dict)
            bbox_params = {k: v.numpy()[0] for k, v in boxes.items()}
            layout_list.append(bbox_params)
    
    return sampled_indices, layout_list
