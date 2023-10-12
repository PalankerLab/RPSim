from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle
import warnings
import numpy as np

def determine_text_offset(text, draw, font):
    """
    This function detects the longest text line, determines its size and
    returns the offset along the X-axis for correctly centering the text. 
    The text x center is computed as the image center, minus the offset divided by two.|

    WARNING: This function is no longer used as I discovered the text anchor option "mm" which
    directly computes the offset for the X and Y position. 

    Return: the offset size in pixels
    """
    # Determine which line is the longest in a multiline text
    longest_line, max = '', 0
    for line in text.splitlines():
        if len(line) > max:
            longest_line = line
            max = len(line)
    
    # Compute the x offset for centering the text
    return draw.textlength(longest_line, font)/2

def determine_center_hex(path_label, center):
    """
    This function determines the location of the central electrode hexagon in
    one of the pattern.
    WARNING: no longer used, it will be replaced by determine_central_label

    Params:
        path_label - a string containing the path of the label in pkl format
        center - an int corresponding to the label of the central electrode

    """
    with open(path_label, 'rb') as f:
        label = pickle.load(f)
    # TODO automate: finding the number of the central hexagon (99 for the 100 micron pixels)
    #mask = ((label == center)*255).astype(np.uint8)
    mask = (label == center)

    # Thanks ChatGPT
    hexagon_image = Image.fromarray(mask)
    # Find the bounding box
    bbox = hexagon_image.getbbox()
    # Calculate the center coordinates
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)

    return center_x, center_y

def create_distance_matrix(electrode_pixel_size):
    """
    Returns a distance matrix (Manhattan distance) from the center of the label file for a given electrode size.
    If the number of columns is even, the 0 will be centered to the right
        (e.g. for 4 columns the distance is [-2, -1, 0, 1])
    Same principle for an even number of rows, the zero will be centered towards the bottom

    For taking non-square, as well as odd and even input images, all the tricky + 1 and -1 are unfortunately necessary :/ 

        Parameters:
            electrode_pixel_size (int): the electrode size in micron
        
        Returns:
            dist_matrix (Numpy.array): same shape as label, where each matrix entry is the Manhattan distance to the center
            label (Numpy.array): contains the pixel position of the electrode in the pattern's image 
    """
    # TODO make sure the file exists, or directly pass a working path
    file = f'../user_files/user_input/image_sequence/pixel_label_PS{electrode_pixel_size}.pkl'
    with open(file, 'rb') as f:
        label = pickle.load(f)
    
    num_rows, num_columns = label.shape
    # Create an array with increasing values, centered around 0
    horizontal_dist = np.arange(-(num_columns // 2), num_columns // 2 + 1)
    vertical_dist = np.arange(-(num_rows // 2), num_rows // 2 + 1)
    # Replicate it horizontally and vertically
    horizontal_dist = np.abs(np.tile(horizontal_dist, (num_rows + 1, 1)))
    vertical_dist = np.abs(np.tile(vertical_dist, (num_columns + 1, 1)))
    # Rotate because Numpy creates horizontal array 
    vertical_dist = np.rot90(vertical_dist, 1)

    # Crop by 1 row, if the row number is odd
    if num_rows % 2 == 1:
        horizontal_dist = horizontal_dist[ :-1, :]
    # Crop by 1 column, if the column number is odd
    if num_columns % 2 == 1:
        vertical_dist = vertical_dist[ :, :-1]

    # Sum the vertical and horizontal matrix to get the distance matrix 
    dist_matrix = horizontal_dist + vertical_dist

    # If the row or column are even, we have to crop by 1 unit here
    if num_rows % 2 == 0:
        dist_matrix = dist_matrix[ :-1, :]
    if num_columns % 2 == 0:
        dist_matrix = dist_matrix[ :, :-1]

    return dist_matrix, label

def determine_central_label(dist_matrix, label):
    """
        Return the label of the central electrode, which is later used to determine the center of the central electdoe.
            Parameters:
                dist_matrix (Numpy.array): same shape as label, where each matrix entry is the Manhattan distance to the center
                label (Numpy.array): contains the pixel position of the electrode in the pattern's image 

            Return:
                central_label (int): the label of the central label
    """
    # Start by selecting the non-zero pixels (i.e. the photodiode)
    mask = label != 0 
    # Find the distance of the closest electrode (i.e. the smallest value in dist matrix which non-zero in selection)
    min_dist = np.min(dist_matrix[mask])
    # Select all the image pixels at that distance
    mask_2 = dist_matrix == min_dist
    central_labels = label[mask_2]
    # Only keep non-zero labels 
    central_labels = central_labels[central_labels != 0]
    # These are all the electrode pixels located at the same distance to the center, draw the first one
    
    # TODO check whether we can improve this criterion
    return central_labels[0]

def find_center(electrode_pixel_size):
    """
    Finds the center of the central electrode for a certain electrode size pattern.

        Parameters:
            electrode_pixel_size (int): the electrode size in micron
        
        Return: 
            center_x (int): The x coordinate corresponding to the center of the central electrode in the pattern image 
            center_y (int): The y coordinate corresponding to the center of the central electrode in the pattern image 
    """

    dist_matrix, label = create_distance_matrix(electrode_pixel_size)
    central_label = determine_central_label(dist_matrix, label)

    # Only keep the central electrode
    mask = (label == central_label)

    # Create an empty image, except for the central electrode
    hexagon_image = Image.fromarray(mask)
    # Find the bounding box
    bbox = hexagon_image.getbbox()
    # Calculate the center coordinates - the middle between the top corners and middle betwen top and bottom corner
    center_x = int((bbox[0] + bbox[2]) / 2)
    center_y = int((bbox[1] + bbox[3]) / 2)

    return center_x, center_y

def determine_font_size(electrode_pixel_size, image_width, text_size, text):
    """
    Determines the adequate font size such that a Landolt C cover a certain
    fraction of an electrode pixel. Currently the fraction is fixed to 1.
    
    Adapted from this post on Stackoverflow: https://stackoverflow.com/questions/4902198/pil-how-to-scale-text-size-in-relation-to-the-size-of-the-image 
    It is brute force, and it could be imporved, but it works.  

    Params:
        electrode_pixel_size (int): The size in micron
        width (int): The image width (based on the pattern we are drawing on)
        text_size (float): The size of the letters with respect to the pixel size

    Returns (ImageDraw.font): The preloaded font with the correct size  
    """    
    # Image fraction represents the proportion the text should take with respect to the image.
    # In this case, the letter size are multiples of the electrode pixel size. 
    # WARNING the big assumption is that the png image of the layout has a scale such 1 pixel = micron, which seems true after measuring the images in ImageJ
    # TODO improve the criterion img_fraction to take into account word or sentences, not only letters
    img_fraction = electrode_pixel_size * text_size / image_width 

    # starting font size
    font_size = 1
    font = ImageFont.truetype("Sloan.otf", font_size)
    while font.getlength(text) < img_fraction * image_width:
        # iterate until the text size is just larger than the criteria
        font_size += 1
        font = ImageFont.truetype("Sloan.otf", font_size)
    
    return font

def determine_location_letter(electrode_pixel_size, text, letter_size, position):
    """
    Determine the location of the letter to project with respect to the electrodes used,
    letter size, and required position.

        Parameters:
            electrode_pixel_size (int): Size of the electrode in micron
            text (int): The letter (or eventually test) text to be projected 
            letter_size (float): The letter's size as a multiple of the elec. pixel size
            position ((float, float)): The position with respect to the central electrode as multiple of the elec. pixel size. Half electrode means going toward the diagonal

        Returns:
            font (ImageDraw.font): The font used for printing with correct font size
            center_x (int): Letter's X position
            center_y (int): Letter's Y position
    """
    if len(text) > 1:
        warnings.warn("This function is implemented for letters only. It is not ready yet for senteces.")

    path = f"../user_files/user_input/image_sequence/Grid_PS{electrode_pixel_size}.png"
    background = Image.open(path, 'r')
    width, height = background.size

    # Convert the pixel pattern into a PIL drawing
    draw = ImageDraw.Draw(background)    
    
    # Find the font size matching the electrode pixel size
    font = determine_font_size(electrode_pixel_size, width, letter_size, text)

    # TODO discard this part of the code once I am sure it is no longer useful
    # After finding the correct font size, test it 
    #path_label = f"../user_files/user_input/image_sequence/pixel_label_PS{electrode_pixel_size}.pkl"
    #center_x, center_y = determine_center_hex(path_label, electrode_label_number)
    # Correct for the offset of the shape
    #offset = determine_text_offset(text, draw, font) No longer used by changing the text anchor
    
    # Determine the center of the central electrode
    center_x, center_y = find_center(electrode_pixel_size)

    # Move the letter by multiple of the pixel size as requested by the user
    shift_x, shift_y = position
    # TODO the translation is not perfect, double check how to adjust it    
    # As the pattern is a honeycomb and not a grid, half pixel should be taken into account
    rounded_x = np.round(shift_x)
    if shift_x == rounded_x: # TODO do more test to check whether this calculation make sense
        center_x += shift_x * electrode_pixel_size
    else:
        center_x += (rounded_x - np.sign(shift_x) * np.sin(np.pi * 30/180)) * electrode_pixel_size 
    
    center_y +=  shift_y * np.cos(np.pi * 30/180) * electrode_pixel_size
    
    return font, center_x, center_y

def draw_overlay_projection(electrode_pixel_size, font, center, text):

    path = f"../user_files/user_input/image_sequence/Grid_PS{electrode_pixel_size}.png"
    background = Image.open(path, 'r')
    text_only = Image.new("RGB", background.size, "black")

    # Convert the images to drawings
    draw_overlay = ImageDraw.Draw(background)
    draw_projection = ImageDraw.Draw(text_only)

    # Draw the text over the background to check the correct alignment
    draw_overlay.text(center, text, font=font, fill="white", align='center', anchor='mm')
    #draw_overlay.text(center, text, font=font, fill="white", align='center')
    
    # Draw the red frame and text on the projected image (black background)
    draw_projection.text(center, text, font=font, fill="white", align='center', anchor="mm")
    draw_projection.rectangle([0, 0, background.size[0] - 1, background.size[1] - 1], outline="red", width=2)
    
    return background, text_only

def draw_overlay_projection_rotation(electrode_pixel_size, letter_size, position_user, text, rotation):
    """
    Returns two images,the actual projected image, and the projected image with the electrode pattern as background
    This is the final function which assembles all the pieces to project a letter. 

        Parameters:
            electrode_pixel_size (int): Size of the electrode in micron
            letter_size (float): The letter's size as a multiple of the elec. pixel size
            position ((float, float)): The position with respect to the central electrode as multiple of the elec. pixel size. Half electrode means going toward the diagonal
            text (int): The letter (or eventually test) text to be projected 
        
        Return:
            background_overlay (PIL.Image): The electrode pattern as background, and the projected image as overlay
            background_projected (PIL.Image): A black screen as background, and the projected image as overlay
    """
    # Define the size of the text image - TODO adapt for more than one letter
    image_size = (electrode_pixel_size * letter_size, electrode_pixel_size * letter_size)

    # Create a new image with a transparent background
    text_image = Image.new('RGBA', image_size, (0, 0, 0, 0))

    # Convert to drawing, set the font, and write text
    text_drawing = ImageDraw.Draw(text_image)
    font, center_x, center_y = determine_location_letter(electrode_pixel_size, text, letter_size, position_user)
    text_drawing.text((0, 0), text, font=font, fill="white", align='center')

    # Rotate with expansion of the image (i.e. no crop)
    text_image = text_image.rotate(rotation, expand = 1)

    # Load backgrond image (the electrode pattern)
    path = f"../user_files/user_input/image_sequence/Grid_PS{electrode_pixel_size}.png"
    background_overlay = Image.open(path, 'r')
    background_projected = Image.new("RGB", background_overlay.size, "black")

    # Compute the new position taking into account the offset required by the user, and the new size due to rotation expansion
    center_x -= text_image.size[0]/2
    center_y -= text_image.size[1]/2
    position_actual = (int(center_x), int(center_y))

    # Create the final images
    background_overlay.paste(text_image, position_actual, text_image)
    background_projected.paste(text_image, position_actual, text_image)

    # Add the red frame around the projected image
    drawing_projection = ImageDraw.Draw(background_projected)
    drawing_projection.rectangle([0, 0, background_projected.size[0] - 1, background_projected.size[1] - 1], outline="red", width=2)

    return background_overlay, background_projected
