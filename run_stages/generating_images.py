from PIL import Image, ImageDraw, ImageFont
import numpy as np
import pickle
import warnings
import numpy as np


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
    # Number of rows and columns
    rows, columns = label.shape
    vertical_dist, horizontal_dist = np.mgrid[np.ceil(-columns/2):np.ceil(columns/2), np.ceil(-rows/2):np.ceil(rows/2)]
    dist_matrix = np.abs(vertical_dist) + np.abs(horizontal_dist)

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

def determine_font_size(electrode_pixel_size, image_width, letter_size, text):
    """
    Determines the adequate font size such that a Landolt C cover a certain
    fraction of an electrode pixel. Currently the fraction is fixed to 1.
    
    Adapted from this post on Stackoverflow: https://stackoverflow.com/questions/4902198/pil-how-to-scale-text-size-in-relation-to-the-size-of-the-image 
    It is brute force, and it could be imporved, but it works.  

    Params:
        electrode_pixel_size (int): The size in micron
        width (int): The image width (based on the pattern we are drawing on)
        letter_size (float): The size of capital letter as a multiple of the pixel size

    Returns (ImageDraw.font): The preloaded font with the correct size  
    """    
    
    # All capital letters have the same size (or bounding box), and we want to calibrate the 
    # font size according to a capital letter. We don't want lower-case or punctuation characters 
    text = "C"
    
    # Image fraction represents the proportion the text should take with respect to the image.
    # In this case, the letter size are multiples of the electrode pixel size. 
    # WARNING the big assumption is that the png image of the layout has a scale such 1 pixel = micron, which seems true after measuring the images in ImageJ
    img_fraction = electrode_pixel_size * letter_size / image_width 

    # starting font size
    font_size = 1
    font = ImageFont.truetype("Sloan.otf", font_size)
    while font.getlength(text) < img_fraction * image_width:
        # Iterate until the text size is slightly larger than the criteria
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
            letter_size (float): The size of capital letter as a multiple of the pixel size
            position ((float, float)): The position with respect to the central electrode as multiple of the elec. pixel size. Half electrode means going toward the diagonal

        Returns:
            font (ImageDraw.font): The font used for printing with correct font size
            center_x (int): Letter's X position
            center_y (int): Letter's Y position
    """

    path = f"../user_files/user_input/image_sequence/Grid_PS{electrode_pixel_size}.png"
    background = Image.open(path, 'r')
    width, height = background.size

    # Convert the pixel pattern into a PIL drawing
    draw = ImageDraw.Draw(background)    
    
    # Find the font size matching the electrode pixel size
    font = determine_font_size(electrode_pixel_size, width, letter_size, text)
    
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

def convert_user_position_to_actual(electrode_pixel_size, center, position_user, width, height):
    """
    Convert the user position to a shift, add it to the central electrode, and subtract the offset of the figure size
    Return the position where the image should be located on the projected image
    """

    # TODO integrate this function in the other functions 
    # Move the letter by multiple of the pixel size as requested by the user
    center_x, center_y = center
    shift_x, shift_y = position_user
    # TODO the translation is not perfect, double check how to adjust it    
    # As the pattern is a honeycomb and not a grid, half pixel should be taken into account
    rounded_x = np.round(shift_x)
    if shift_x == rounded_x: # TODO do more test to check whether this calculation make sense
        center_x += shift_x * electrode_pixel_size
    else:
        center_x += (rounded_x - np.sign(shift_x) * np.sin(np.pi * 30/180)) * electrode_pixel_size 
    
    center_y +=  shift_y * np.cos(np.pi * 30/180) * electrode_pixel_size
    
    return int(center_x - width/2), int(center_y - height/2)


def draw_text(electrode_pixel_size, letter_size, position_user, text, rotation):
    """
    This is the final function which assembles all the pieces to project a letter. 
    
        Parameters:
            electrode_pixel_size (int): Size of the electrode in micron
            letter_size (float): The letter's size as a multiple of the elec. pixel size
            position ((float, float)): The position with respect to the central electrode as multiple of the elec. pixel size. Half electrode means going toward the diagonal
            text (int): The letter (or eventually test) text to be projected 
        
        Return:
            position_actual (int, int): The position to use when pasting the letter image into the projected/overlay image
            text_image (PIL.Image): An image of the letter on black background with the correct size and orientation
    """
    
    # Determine image size
    lines = text.splitlines()
    n_lines = len(lines)
    max_len = len(max(lines, key=len))
    image_size = (electrode_pixel_size * letter_size * max_len, electrode_pixel_size * letter_size * n_lines)
    # Create a rnew image with a transparent background
    text_image = Image.new('RGBA', image_size, (0, 0, 0, 0))

    # Convert to drawing, set the font, and write text
    text_drawing = ImageDraw.Draw(text_image)
    font, center_x, center_y = determine_location_letter(electrode_pixel_size, text, letter_size, position_user)
    text_drawing.text((0, 0), text, font=font, fill="white", align='center', spacing=0) # TODO check for the desired spacing, if non-zero, increase image_size height by spacing, eventually use ImageDraw.textbox() 
    
    # Rotate with expansion of the image (i.e. no crop)
    text_image = text_image.rotate(rotation, expand = 1)

    # Compute the new position taking into account the offset required by the user, and the new size due to rotation expansion
    center_x -= text_image.size[0]/2
    center_y -= text_image.size[1]/2
    position_actual = (int(center_x), int(center_y))
    determine_overflow(electrode_pixel_size, image_size, max_len, n_lines)

    return position_actual, text_image

def rotate_point(x, y, theta):
    """
    Rotates a point counterclockwise about the origin
        Parameters:
            x (int): X position
            y (int): Y position 
            theta (float): Rotation angle in radian
    """
    x_rot = x * np.cos(theta) - y * np.sin(theta)
    y_rot = x * np.sin(theta) + y * np.cos(theta)
    return x_rot, y_rot

def determine_overflow(electrode_pixel_size, text_size, max_len, n_lines):
    """
    This functions determines whether the text will overflow at position (0, 0) and rotation 0°.
    """
    
    # TODO put all the code inside of a class, and make the image size an attribute
    # Add pixel size as a safety margin
    img_size = 1500 - electrode_pixel_size, 1500 - electrode_pixel_size
    if text_size[0] > img_size[0]:
        recommended_size = np.floor(img_size[0] / (electrode_pixel_size * max_len))
        warnings.warn(f"Your text may be too wide, we recommend using a letter size of at least: {recommended_size}")
    if text_size[1] > img_size[1]:
        recommended_size = np.floor(img_size[1] / (electrode_pixel_size * n_lines))
        warnings.warn(f"Your text may be too large, we recommend using a letter size of at least: {recommended_size}")  

def draw_grating(electrode_pixel_size, img_size, width_grating, pitch_grating, rotation, position_user):
    """
    Draw a rectangular grating of width width_grating spaced (edge to edge) by pitch_grating
    WARNING the position is not enabled yet

        Parameters:
            electrode_pixel_size (int): Size of the electrode in micron
            img_size (tuple): the size of the projected image
            width_grating (int): The width of grating in micron
            pitch_grating (int): The shorted distance separating each grating (edge to edge) 
            rotation (float): The rotation of the grating in degrees - Between -90° and 90° included
            position_user (float, int): The position of the grating with respect to the central pixel, move with respect to the pixel size along the X-axis. Y-axis is not implemented yet.  
        
        Returns:
            grating_only (PIL.Image): The image of the grating with alpha transparency enabled
    """
    
    if (np.abs(rotation) > 90):
        raise ValueError("The rotation angle shoud be between -90° <= rotation <= 90°")
    if position_user[1] != 0:
        warnings.warn("The Y position cannot be shifted. Y is set to 0 instead.")

    width, height = img_size
    theta = np.deg2rad(rotation)

    # We want the grating to overlap the central pixel, hence offset by half the grating rotated width
    offset_x = int( np.cos(theta) * width_grating / 2 ) + position_user[0] * electrode_pixel_size
    center_x, center_y = find_center(electrode_pixel_size)

    # Compute the bottom left corner of the grating, from the image center to the right
    fwd_x_pos = np.arange(center_x - offset_x, 2 * width, width_grating + pitch_grating)
    # Compute the bottom left corner of the grating, from the image center to the left
    bckwd_x_pos = np.arange(center_x - offset_x -(width_grating + pitch_grating), - 2 * width, - (width_grating + pitch_grating))
    list_x_positions = np.hstack((bckwd_x_pos, fwd_x_pos))

    grating_only = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    drawing_grating = ImageDraw.Draw(grating_only)
       
    # # The grating always span the whole Y-axis -> it can be larger than the height, and should be high enough to withstand a rotation
    y_l, y_h = -width, height + width
    # Remember that PIL has the origin at the top left corner
    for x_pos in list_x_positions:
        points = [] 
        points.append(rotate_point(x_pos, y_l, theta))  # Top left
        points.append(rotate_point(x_pos + width_grating, y_l, theta))  # Top right
        points.append(rotate_point(x_pos + width_grating, y_h, theta))  # Bottom right
        points.append(rotate_point(x_pos, y_h, theta))  # Bottom left
        # Draw rotated rectangle
        drawing_grating.polygon(points, fill="white", outline=None)
    
    return grating_only

def draw_rectangle(electrode_pixel_size, position_user, rotation, width, height):
    
    rectangle = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(rectangle)
    draw.rectangle([0, 0, width, height], fill="white")

    # Rotate with expansion of the image (i.e. no crop)
    rectangle = rectangle.rotate(rotation, expand = 1)

    # Compute the new position taking into account the offset required by the user, and the new size due to rotation expansion
    width = rectangle.size[0]
    height = rectangle.size[1]

    # TODO make this a class and call find_center in the class constructor
    center = find_center(electrode_pixel_size)
    position_actual = convert_user_position_to_actual(electrode_pixel_size, center, position_user, width, height)
    return position_actual, rectangle

def draw_circle(electrode_pixel_size, position_user, diameter):

    circle = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
    draw = ImageDraw.Draw(circle)
    draw.ellipse([0, 0, diameter, diameter], fill="white")

   # TODO make this a class and call find_center in the class constructor
    center = find_center(electrode_pixel_size)
    position_actual = convert_user_position_to_actual(electrode_pixel_size, center, position_user, diameter, diameter)
    return position_actual, circle


def draw_projected_overlay(electrode_pixel_size, type, args):
    """
    Returns two images,the actual projected image, and the projected image with the electrode pattern as background
    A function calling all lower-level functions

        Parameters:
            electrode_pixel_size (int): Size of the electrode in micron
            type (string): The type of drawing to do
            args (misc): to be forwarded to the lower-level functions
        
        Return:
            overlay (PIL.Image): The electrode pattern as background, and the projected image as overlay
            projected (PIL.Image): A black screen as background, and the projected image as overlay
    """

    path_image = f"../user_files/user_input/image_sequence/Grid_PS{electrode_pixel_size}.png"
    #path_label = f"../user_files/user_input/image_sequence/pixel_label_PS{electrode_pixel_size}.pkl"
    
    overlay = Image.open(path_image, 'r')
    projected = Image.new("RGB", overlay.size, "black")

    # TODO make it more robust with kwargs and stuff
    if type == "grating":
        to_be_projected = draw_grating(electrode_pixel_size, overlay.size, args[0], args[1], args[2], args[3])
        position_actual = (0, 0)
    elif type == "text":
        if len(args[2]) == 0:
            raise ValueError("No text was provided!") 
        position_actual, to_be_projected = draw_text(electrode_pixel_size, args[0], args[1], args[2], args[3]) 
    elif type=="rectangle":
        position_actual, to_be_projected = draw_rectangle(electrode_pixel_size, args[0], args[1], args[2], args[3])
    elif type=="circle":
        position_actual, to_be_projected = draw_circle(electrode_pixel_size, args[0], args[1])
    else:
        raise NotImplementedError
    
    # Create the final images
    overlay.paste(to_be_projected, position_actual, to_be_projected)
    projected.paste(to_be_projected, position_actual, to_be_projected)

    # Add the red frame around the projected image
    drawing_projection = ImageDraw.Draw(projected)
    drawing_projection.rectangle([0, 0, projected.size[0] - 1, projected.size[1] - 1], outline="red", width=2)

    return overlay, projected