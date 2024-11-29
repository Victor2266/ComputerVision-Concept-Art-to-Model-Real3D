import torch
import argparse
from collections import OrderedDict

def convert_checkpoint(tar_path, ckpt_path, original_ckpt_path):
    """
    Convert training checkpoint to inference checkpoint, preserving all model weights.
    """
    print(f"Loading your training checkpoint from {tar_path}")
    training_ckpt = torch.load(tar_path, map_location='cuda')
    
    print(f"Loading original model from {original_ckpt_path}")
    original_ckpt = torch.load(original_ckpt_path, map_location='cuda')
    
    # Get state dictionaries
    original_state_dict = original_ckpt['state_dict']
    trained_state_dict = training_ckpt['state_dict']
    
    print("\nOriginal model keys:", len(original_state_dict.keys()))
    print("Trained model keys:", len(trained_state_dict.keys()))
    
    # Create new state dict starting with ALL original weights
    new_state_dict = OrderedDict()
    
    # First, copy everything from the original checkpoint
    for key, value in original_state_dict.items():
        new_state_dict[key] = value
    
    # Then update only the trained decoder layers
    print("\nUpdating decoder layers:")
    updated_count = 0
    for key in trained_state_dict.keys():
        if 'decoder' in key:
            new_state_dict[key] = trained_state_dict[key]
            print(f"Updated: {key}")
            updated_count += 1
    
    print(f"\nTotal layers in new checkpoint: {len(new_state_dict)}")
    print(f"Number of updated decoder layers: {updated_count}")
    
    # Create the final checkpoint with ALL components
    final_checkpoint = {
        'state_dict': new_state_dict,
        'epoch': training_ckpt.get('epoch', 0),
        'global_step': training_ckpt.get('global_step', 0),
        'optimizer': original_ckpt.get('optimizer', None),  # Preserve optimizer state
        'scheduler': original_ckpt.get('scheduler', None),  # Preserve scheduler state
        'scaler': original_ckpt.get('scaler', None),       # Preserve scaler state
        # Add any other components from original checkpoint
        **{k: v for k, v in original_ckpt.items() if k not in ['state_dict', 'epoch', 'global_step', 'optimizer', 'scheduler', 'scaler']}
    }
    
    # Save the new checkpoint
    print(f"\nSaving complete checkpoint to {ckpt_path}")
    torch.save(final_checkpoint, ckpt_path)
    
    # Verify file size
    import os
    original_size = os.path.getsize(original_ckpt_path) / (1024 * 1024 * 1024)  # Convert to GB
    new_size = os.path.getsize(ckpt_path) / (1024 * 1024 * 1024)  # Convert to GB
    print(f"\nOriginal checkpoint size: {original_size:.2f} GB")
    print(f"New checkpoint size: {new_size:.2f} GB")
    print("Conversion complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert training checkpoint to inference checkpoint')
    parser.add_argument('--tar_path', type=str, required=True, help='Path to your training .tar checkpoint')
    parser.add_argument('--ckpt_path', type=str, required=True, help='Path to save the new .ckpt file')
    parser.add_argument('--original_ckpt_path', type=str, required=True, help='Path to original TripoSR checkpoint')
    
    args = parser.parse_args()
    
    convert_checkpoint(args.tar_path, args.ckpt_path, args.original_ckpt_path)