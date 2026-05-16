def get_relevant_states(state_dict, use_averaged_model=False):
    # Conditional logic to keep or delete keys
    if use_averaged_model:
        # Keep only keys that contain 'averaged_model'
        filtered_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if 'averaged_model' in k}
    else:
        # Delete keys that contain 'averaged_model'
        filtered_dict = {k.replace("module.", ""): v for k, v in state_dict.items() if 'averaged_model' not in k}

    if "num_updates_tracked" in filtered_dict:
        del filtered_dict["num_updates_tracked"]
    return filtered_dict


def denorm(x):
    return (x.clamp(-1, 1) + 1) / 2