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

    This function is no longer useful as I discovered the text anchor option "mm" which
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

def determine_location_letter(electrode_pixel_size, electrode_label_number, text, letter_size, position):
    """
    Determine the location of the letter to project with respect to the electrodes used,
    letter size, and required position.

        Parameters:
            electrode_pixel_size (int): Size of the electrode in micron
            electrode_label_number (int): The elec. number corresponding to the central electrode the label files (e.g. pixel_label_PS100.pkl)
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

    # After finding the correct font size, test it 
    path_label = f"../user_files/user_input/image_sequence/pixel_label_PS{electrode_pixel_size}.pkl"
    center_x, center_y = determine_center_hex(path_label, electrode_label_number)
    # Correct for the offset of the shape
    #offset = determine_text_offset(text, draw, font) No longer used by changing the text anchor
    
    # Move the letter by multiple of the pixel size as requested by the user
    shift_x, shift_y = position    

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

def draw_overlay_projection_rotation(electrode_pixel_size, electrode_label_number, letter_size, position_user, text, rotation):
    """
    TODO
    """
    # Define the size of the text image - TODO adapt for more than one letter
    image_size = (electrode_pixel_size * letter_size, electrode_pixel_size * letter_size)

    # Create a new image with a transparent background
    text_image = Image.new('RGBA', image_size, (0, 0, 0, 0))

    # Convert to drawing, set the font, and write text
    text_drawing = ImageDraw.Draw(text_image)
    font, center_x, center_y = determine_location_letter(electrode_pixel_size, electrode_label_number, text, letter_size, position_user)
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