import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def draw_matches_paper(img0, img1, kpts0, kpts1, save_path=None, 
                       color='lime',
                       linewidth=0.5, 
                       point_size=10.0):
    """
    Draw publication-style feature matches.
    """
    # 1. Prepare the side-by-side image.
    # Convert tensors to CPU numpy arrays.
    if hasattr(img0, 'cpu'): img0 = img0.cpu().numpy()
    if hasattr(img1, 'cpu'): img1 = img1.cpu().numpy()
    if hasattr(kpts0, 'cpu'): kpts0 = kpts0.cpu().numpy()
    if hasattr(kpts1, 'cpu'): kpts1 = kpts1.cpu().numpy()

    # Convert (C,H,W) to (H,W,C).
    if img0.ndim == 3 and img0.shape[0] == 3: img0 = img0.transpose(1, 2, 0)
    if img1.ndim == 3 and img1.shape[0] == 3: img1 = img1.transpose(1, 2, 0)

    # Normalize image values.
    if img0.max() <= 1.1: img0 = (img0 * 255).astype(np.uint8)
    if img1.max() <= 1.1: img1 = (img1 * 255).astype(np.uint8)

    h0, w0 = img0.shape[:2]
    h1, w1 = img1.shape[:2]
    
    height = max(h0, h1)
    width = w0 + w1
    output_img = np.zeros((height, width, 3), dtype=np.uint8)
    
    if len(img0.shape) == 2: img0 = np.stack([img0]*3, -1)
    if len(img1.shape) == 2: img1 = np.stack([img1]*3, -1)
    
    output_img[:h0, :w0] = img0
    output_img[:h1, w0:w0+w1] = img1
    
    # 2. Draw matches.
    fig = plt.figure(figsize=(20, 10), dpi=300) 
    plt.imshow(output_img)
    plt.axis('off')
    
    # Draw green lines.
    
    # Use alpha to keep dense lines readable.
    for i in range(len(kpts0)):
        plt.plot([kpts0[i, 0], kpts1[i, 0] + w0], 
                 [kpts0[i, 1], kpts1[i, 1]], 
                 c=color, 
                 linewidth=linewidth, 
                 alpha=0.7) 
    
    # Draw hollow feature points.
    # facecolors='none' keeps circles transparent.
    # edgecolors=color sets the circle border color.
    plt.scatter(kpts0[:, 0], kpts0[:, 1], 
                s=point_size, 
                facecolors='none', 
                edgecolors=color, 
                marker='o', 
                linewidths=1.0) # Circle border width.
                
    plt.scatter(kpts1[:, 0] + w0, kpts1[:, 1], 
                s=point_size, 
                facecolors='none', 
                edgecolors=color, 
                marker='o', 
                linewidths=1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
        print(f"Saved match visualization to {save_path}")
    
    plt.close()
