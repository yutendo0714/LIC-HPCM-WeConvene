"""
CheckerboardMask: Generate checkerboard patterns for parallel entropy coding

Key insight from VCT (ICLR 2022) and ELIC (CVPR 2022):
"Checkerboard pattern enables richer spatial dependencies while maintaining parallel decoding"

This module generates masks for splitting spatial grid into N groups.
Each group can be decoded in parallel given previous groups.
"""
import torch
import torch.nn as nn


class CheckerboardMask:
    """
    Generate checkerboard pattern masks for parallel entropy coding
    
    Supports both 2-group and 4-group patterns:
    
    2-group pattern (simple):
    □■□■□    Group 1: Can decode in parallel
    ■□■□■    Group 2: Can decode in parallel (using Group 1 as context)
    □■□■□
    
    4-group pattern (advanced):
    1 2 1 2    Group 1 → Group 2 → Group 3 → Group 4
    3 4 3 4    Each group internally parallel
    1 2 1 2    Provides richer context
    """
    
    @staticmethod
    def get_masks_2groups(B, C, H, W, device):
        """
        Generate 2-group checkerboard masks
        
        Group 1: (even_row, even_col) + (odd_row, odd_col)
        Group 2: (even_row, odd_col) + (odd_row, even_col)
        
        Args:
            B, C, H, W: Tensor dimensions
            device: torch device
        Returns:
            List[Tensor]: [mask1, mask2] each of shape (B, C, H, W)
        """
        mask1 = torch.zeros(B, C, H, W, device=device)
        mask2 = torch.zeros(B, C, H, W, device=device)
        
        # Checkerboard pattern
        mask1[:, :, 0::2, 0::2] = 1  # Even row, even col
        mask1[:, :, 1::2, 1::2] = 1  # Odd row, odd col
        
        mask2[:, :, 0::2, 1::2] = 1  # Even row, odd col
        mask2[:, :, 1::2, 0::2] = 1  # Odd row, even col
        
        return [mask1, mask2]
    
    @staticmethod
    def get_masks_4groups(B, C, H, W, device):
        """
        Generate 4-group checkerboard masks
        
        Group 1: (even_row, even_col) - Corners
        Group 2: (even_row, odd_col) - Horizontal edges
        Group 3: (odd_row, even_col) - Vertical edges
        Group 4: (odd_row, odd_col) - Centers
        
        This pattern ensures each pixel in later groups has maximum
        context from previously decoded groups.
        
        Args:
            B, C, H, W: Tensor dimensions
            device: torch device
        Returns:
            List[Tensor]: [mask1, mask2, mask3, mask4] each of shape (B, C, H, W)
        """
        masks = []
        
        for i in range(2):  # Row parity
            for j in range(2):  # Column parity
                mask = torch.zeros(B, C, H, W, device=device)
                mask[:, :, i::2, j::2] = 1
                masks.append(mask)
        
        return masks
    
    @staticmethod
    def visualize_pattern(H=8, W=8, num_groups=4):
        """
        Visualize checkerboard pattern (for debugging/documentation)
        
        Args:
            H, W: Grid dimensions
            num_groups: 2 or 4
        Returns:
            pattern: (H, W) numpy array with group indices
        """
        import numpy as np
        
        pattern = np.zeros((H, W), dtype=int)
        
        if num_groups == 2:
            # Group 1: White squares
            pattern[0::2, 0::2] = 1
            pattern[1::2, 1::2] = 1
            # Group 2: Black squares (0 is already set)
            pattern[0::2, 1::2] = 2
            pattern[1::2, 0::2] = 2
        
        elif num_groups == 4:
            pattern[0::2, 0::2] = 1  # Even row, even col
            pattern[0::2, 1::2] = 2  # Even row, odd col
            pattern[1::2, 0::2] = 3  # Odd row, even col
            pattern[1::2, 1::2] = 4  # Odd row, odd col
        
        return pattern
    
    @staticmethod
    def get_anchor_positions(group_idx, H, W):
        """
        Get spatial positions for a given group
        
        Useful for debugging and custom context gathering
        
        Args:
            group_idx: 0-3 for 4-group pattern, 0-1 for 2-group
            H, W: Spatial dimensions
        Returns:
            positions: List of (h, w) tuples
        """
        positions = []
        
        if group_idx == 0:  # (even, even)
            for h in range(0, H, 2):
                for w in range(0, W, 2):
                    positions.append((h, w))
        elif group_idx == 1:  # (even, odd)
            for h in range(0, H, 2):
                for w in range(1, W, 2):
                    positions.append((h, w))
        elif group_idx == 2:  # (odd, even)
            for h in range(1, H, 2):
                for w in range(0, W, 2):
                    positions.append((h, w))
        elif group_idx == 3:  # (odd, odd)
            for h in range(1, H, 2):
                for w in range(1, W, 2):
                    positions.append((h, w))
        
        return positions


# Example usage and testing
if __name__ == "__main__":
    print("Checkerboard Mask Generator")
    print("=" * 50)
    
    # Visualize 2-group pattern
    print("\n2-Group Checkerboard Pattern:")
    pattern_2 = CheckerboardMask.visualize_pattern(8, 8, num_groups=2)
    print(pattern_2)
    
    # Visualize 4-group pattern
    print("\n4-Group Checkerboard Pattern:")
    pattern_4 = CheckerboardMask.visualize_pattern(8, 8, num_groups=4)
    print(pattern_4)
    
    # Test mask generation
    print("\nTesting mask generation...")
    device = torch.device('cpu')
    masks_2 = CheckerboardMask.get_masks_2groups(1, 1, 8, 8, device)
    masks_4 = CheckerboardMask.get_masks_4groups(1, 1, 8, 8, device)
    
    print(f"2-group masks: {len(masks_2)} masks generated")
    print(f"4-group masks: {len(masks_4)} masks generated")
    
    # Verify coverage (all positions should be covered exactly once)
    coverage_2 = sum(masks_2).squeeze()
    coverage_4 = sum(masks_4).squeeze()
    
    assert torch.all(coverage_2 == 1), "2-group coverage error"
    assert torch.all(coverage_4 == 1), "4-group coverage error"
    
    print("✓ All tests passed!")
