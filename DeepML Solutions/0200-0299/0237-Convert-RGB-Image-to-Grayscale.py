import numpy as np

def rgb_to_grayscale(image):
    """
    Convert an RGB image to grayscale using luminosity method.
    
    Args:
        image: RGB image as list or numpy array of shape (H, W, 3)
               with values in range [0, 255]
    
    Returns:
        Grayscale image as 2D list with integer values,
        or -1 if input is invalid
    """
    if not isinstance(image, list):
        return -1
    
    if not image:
        return -1
    
    H = len(image)
    if H == 0:
        return -1
    
    if not isinstance(image[0], list):
        return -1
    
    W = len(image[0])
    if W == 0:
        return -1
    
    if not isinstance(image[0][0], list):
        return -1
    
    if len(image[0][0]) != 3:
        return -1
    
    for i in range(H):
        if len(image[i]) != W:
            return -1
        for j in range(W):
            if not isinstance(image[i][j], list) or len(image[i][j]) != 3:
                return -1
            R, G, B = image[i][j]
            if not (isinstance(R, (int, float)) and isinstance(G, (int, float)) and isinstance(B, (int, float))):
                return -1
            if not (0 <= R <= 255 and 0 <= G <= 255 and 0 <= B <= 255):
                return -1
    
    grayscale = []
    for i in range(H):
        row = []
        for j in range(W):
            R, G, B = image[i][j]
            gray = 0.299 * R + 0.587 * G + 0.114 * B
            row.append(int(round(gray)))
        grayscale.append(row)
    return grayscale

image = [[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [255, 255, 255]]]
print(rgb_to_grayscale(image))