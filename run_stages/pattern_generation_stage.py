import pickle
import matplotlib.pyplot as plt
import numpy as np
import os
import warnings
from copy import deepcopy

from PIL import Image, ImageDraw, ImageFont


from configuration.stages import RunStages
from configuration.configuration_manager import Configuration

from run_stages.common_run_stage import CommonRunStage

class PatternGenerationStage(CommonRunStage):
    """
    This class implements the automated pattern generation from a configuration file.

    Attributes:
        pixel_size (int): The pixel size in micron (we refer to pixel as the hexagonal unit with photodiode, active and return electrode)
        implant_layout (PIL.Image): The image of the pixel layout
        pixel_labels (Numpy.array): Array having the same size as implant_layout. Each entry corresponds to either 0 or the pixel label (the active and return electrodes are also labeled 0, only the photodiode is non-zero)
        
        center_x, center_y (int, int): The pixels coordinate in implant_layout corresponding the center of the central hexagon
        width, height (int, int): The size in pixels (and microns) of the implant_layout image
        
        backgroud_overlay (PIL.Image): The pixel pattern as background, and the projected image as overlay
        projected (PIL.Image): A black screen as background, and the projected image as overlay
    """

    def __init__(self, *args):
        super().__init__(*args)

        self.list_subframes_as_ndarray = []
        self.script = None
        self.dict_PIL_images = {}
    
    @property
    def stage_name(self):
        return RunStages.pattern_generation.name

    def run_stage(self, *args, **kwargs):
        
        """
        The output for CurrentSequence stage will be stored through self.outputcontainer in CommonRunStage
        I just have to return it at the end of the function.
        The BMP images can be saved here, however, there is already a function in for this in CommmonRunStage
        it's just no suited for our needs. 
        """
        if Configuration().params["generate_pattern"]:

            ###x Object based implementation
            
            projection_sequence = Configuration().params["projection_sequences"]
            self.script = projection_sequence.get_script()

            # Iterate on the frames of the projection
            for frame in projection_sequence:
                # For temporary storing the outputs
                list_tmp_bmp = []
                list_tmp_array = []

                # Iterate on the subframes for the given frame
                for idx, subframe in enumerate(frame):
                    drawing_board = ImagePattern(pixel_size = Configuration().params["pixel_size"])
                    # Several patterns can be added to a drawing_board / subframe
                    for pattern in subframe:
                        # Draw the provided pattern
                        pattern.draw(drawing_board)

                    # Save subframe
                    list_tmp_bmp.append((f'Subframe{int(idx+1)}', drawing_board.save_as_PIL()))
                    list_tmp_array.append(drawing_board.save_as_array())

                    plt.rcParams['figure.facecolor'] = 'white'
                    drawing_board.show(frame.name, idx)
                
                # Save frame
                self.dict_PIL_images[frame.name] = list_tmp_bmp
                self.list_subframes_as_ndarray.append(list_tmp_array)
            
            # Current Sequence stage uses the ndarray frames and script, the PIL images are for the user only
            return [self.list_subframes_as_ndarray, self.script, self.dict_PIL_images]                        
        else:
            # If we load pre-existing patterns, we do not need to process anything
            return []  


#################### The class which actually handles the drawing ####################


class ImagePattern():
    """
    A class which stores and draws the PIL.Image containing the patterns to be projected and the patterns overlayed on the implant layout.

    Attributes:
        pixel_size (int): The pixel size in micron - More precisely the pitch between pixels
        suffix (str): Whether we use the large configuration (_lg) or regular size ()
        font_path (str): The path to the font used for printing text patterns

        implant_layout (PIL.Image): The image of the pixcel layout
        pixel_labels (Numpy.array): Array having the same size as implant_layout. Each entry corresponds to either 0 or the pixel label (the active and return electrodes are also labeled 0, only the photodiode is non-zero)
         
        center_x, center_y (int, int): The iamge pixels coordinate in implant_layout corresponding the center of the central hexagon
        width, height (int, int): The size in image pixels (and microns) of the implant_layout image
        
        backgroud_overlay (PIL.Image): The pixel pattern as background, and the projected image as overlay
        projected (PIL.Image): A black screen as background, and the projected image as overlay

        opacity (int): Opacity of the overlayed pattern onto the pixel layout 
        image_pixel_scale (float): The scale in pixel/micron of the implant layout's PNG image
        scaled_pixel (int): The size of the pixel in image-pixel 
    """

    def __init__(self, pixel_size):

        self.pixel_size = pixel_size
        suffix = Configuration().params["pixel_size_suffix"] # Whether we use the large file "_lg" or not
        self.font_path = os.path.abspath("../RPSim/utilities/Sloan.otf") if Configuration().params.get("font_path") is None else Configuration().params["font_path"]

        self.implant_layout = self.load_file(f"Grid_PS{self.pixel_size}{suffix}.png")
        self.pixel_labels = self.load_file(f"pixel_label_PS{self.pixel_size}{suffix}.pkl")

        self.center_x, self.center_y = self.find_center()
        self.width, self.height = self.implant_layout.size[0], self.implant_layout.size[1]

        self.background_overlay = self.implant_layout.copy() 
        self.background_overlay.putalpha(255) # Remove transparency to fully opaque
        self.projected = Image.new("RGB", self.background_overlay.size, "black")
        
        self.opacity = 180 # (0, 255) (transparent to opaque)
        self.image_pixel_scale = self.find_image_scale()
        self.scaled_pixel = int(self.pixel_size * self.image_pixel_scale)

    def __str__(self):

        self.background_overlay.show()
        self.projected.show()
        return "Printed the images!"
    
    def show(self, frame_name, subframe_idx):
        """
        Displays the overlay and projected image as a subplot.
        """
        # plt.clf()
        fig, axes = plt.subplots(1,2, figsize=(12, 20))
        axes[0].imshow(np.array(self.background_overlay))
        axes[1].imshow(np.array(self.projected))
        fig.suptitle(f"{frame_name} Subframe {subframe_idx + 1}", y=0.62)
        plt.show()

    def save_as_PIL(self):
        """
        Returns a deepcopy of the background overlay and projected PIL images
        """
        return deepcopy(self.background_overlay), deepcopy(self.projected)
    
    def save_as_array(self):
        """
        Returns a deepcopy of the projected image as numpy float array
        """
        return deepcopy(np.array(self.projected, dtype=float))
        

    ############################ Helper functions ############################
   
    def load_file(self, file):
        """
        Loads the implant layout image or it's label
        Parameters:
            file (string): The name of the file to load, the rest of the path is assumed
        """
        # TODO adjust path
        #path = os.path.join(".." , "user_files", "user_input", "image_sequence", file) 
        path = os.path.join("user_files", "user_input", "image_sequence", file)
        try:
            with open(path, 'rb') as f:
                if "png" in file:
                    file = Image.open(f, 'r')
                    # Keep it in memory 
                    file.load()    
                elif "pkl" in file:
                    file = pickle.load(f)
                else:
                    raise NotImplementedError     
        except OSError as error: 
            print(error) 
            print(f"The file '{file}' could not be found with path: {path}") 
            
        return file

    def rotate_point(self, x, y, theta):
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

    def determine_overflow(self, text_size, max_len, n_lines):
        """
        This functions determines whether the text will overflow at position (0, 0) and rotation 0°.
        """
        
        # Add pixel size as a safety margin
        img_size = self.width - self.scaled_pixel, self.height - self.scaled_pixel
        if text_size[0] > img_size[0]:
            recommended_size = np.floor(img_size[0] / (self.scaled_pixel * max_len))
            warnings.warn(f"Your text may be too wide, we recommend using a maximum letter size of: {recommended_size}")
        if text_size[1] > img_size[1]:
            recommended_size = np.floor(img_size[1] / (self.scaled_pixel * n_lines))
            warnings.warn(f"Your text may be too large, we recommend using a maximum letter size of: {recommended_size}")  

    def find_image_scale(self):
        """
        Determines the scale of the implant layout image.
        Return:
            - image_pixel_scale (float)

        Example with two different images:
            - For Grid_PS100.png: 
                - Distance in image pixels between pixel center to center: 100
                - Distance in microns: 100
                - image_pixel_scale: 100/100 = 1 pixel/micron
            - For Grid_PS100_lg.png
                - Distance in image pixels between pixel center to center: 75
                - Distnace in microns: 100
                - image_pixel_scale: 75/100 = 0.75 pixel/micron

        Warning! This function assumes that pixel number 1 and 2 are next to each other!
        TODO make the computation of the pixel pitch more robust 
        Note: I assumed that an error of 2 image pixels between the actual and image distance is not significant. 
        """ 

        x1, y1 = self.find_center(1)
        x2, y2 = self.find_center(2)
        pitch_pixel_in_image_pixels = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2)

        if np.abs(pitch_pixel_in_image_pixels - self.pixel_size) > 2:
            return pitch_pixel_in_image_pixels / self.pixel_size
        else:
            return 1 
    
    ############################ Functions for computing position and sizes ############################
    
    def convert_user_position_to_image_position(self, user_position, unit, width, height):
        """
        Convert the user position to the actual position where the object should be located in the final image,
        taking into account the offset of the bouding box for correct centering.

        In µm unit, the image's center is (0, 0), with negative values allowed. However, in Python the matrices
        are indexed with positive integers. So we have to convert to image pixels, shift and round the coordinates.
        Pay attention to Pyhton indexing, the Y-axis is inverted, up-down. 

            Parameters:
                user_position ((float, float)): The position with respect to the central  as multiple of the pixel size. 1.5 along X, means going toward the pixel in diagonal
                unit (str): The unit in which the pattern's parameters are encoded, either pixel or um. 
                width (int): The width of the object to paste in the final image
                height (int): The height of the object to paste in the final image

            Return:
                actual_position (int, int): The position where the image should be located on the projected/overlay image
        """

        if unit == 'pixel':
            # Move the letter by multiple of the pixel size as requested by the user
            center_x, center_y = self.center_x, self.center_y
            shift_x, shift_y = user_position
            # TODO the translation is not perfect, double check how to adjust it    
            # As the pattern is a honeycomb and not a grid, half pixel should be taken into account
            rounded_x = np.round(shift_x)
            if shift_x == rounded_x: # TODO do more test to check whether this calculation make sense
                center_x += shift_x * self.scaled_pixel
            else:
                center_x += (rounded_x - np.sign(shift_x) * np.sin(np.pi * 30/180)) * self.scaled_pixel
            
            # The Y-axis is inverted in Python
            center_y +=  -1 * shift_y * np.cos(np.pi * 30/180) * self.scaled_pixel
        
        elif unit == 'um':
            center_x, center_y = user_position
            # Convert to image pixel, and shift to positive space coordinates
            # With Pyhton the Y-axis is inverted, up-down
            center_x = center_x * self.image_pixel_scale + self.width / 2
            center_y = -1*center_y * self.image_pixel_scale + self.height / 2
        
        # Take into account the offset due to the object size, as PIL aligns 
        # from the top-left corner and not from the center of the object
        return int(center_x - width/2), int(center_y - height/2)
        
    def create_distance_matrix(self):
        """
        Returns a distance matrix (Manhattan distance) from the center of the label file for a given pixel size.
        If the number of columns is even, the 0 will be centered to the right
            (e.g. for 4 columns the distance is [-2, -1, 0, 1])
        Same principle for an even number of rows, the zero will be centered towards the bottom

        Returns:
            dist_matrix (Numpy.array): same shape as label, where each matrix entry is the Manhattan distance to the center
        """

        # Number of rows and columns
        rows, columns = self.pixel_labels.shape
        vertical_dist, horizontal_dist = np.mgrid[np.ceil(-columns/2):np.ceil(columns/2), np.ceil(-rows/2):np.ceil(rows/2)]
        dist_matrix = np.abs(vertical_dist) + np.abs(horizontal_dist)

        return dist_matrix
    
    def determine_central_label(self, dist_matrix):
        """
            Return the label of the central pixel, which is later used to determine the center of the central electdoe.
                Parameters:
                    dist_matrix (Numpy.array): same shape as label, where each matrix entry is the Manhattan distance to the center

                Return:
                    central_label (int): the label of the central label
        """
        # Start by selecting the non-zero pixels (i.e. the photodiode)
        filter_pixel = self.pixel_labels != 0 
        # Find the distance of the closest pixel (i.e. the smallest value in dist matrix which non-zero in selection)
        min_dist = np.min(dist_matrix[filter_pixel])
        # Select all the pixels at that distance
        filter_distance = dist_matrix == min_dist
        central_labels = self.pixel_labels[filter_distance]
        # Only keep non-zero labels 
        central_labels = central_labels[central_labels != 0]
        # These are all the pixels located at the same distance to the center, draw the first one
        
        # TODO check whether we can improve this criterion
        return central_labels[0]
    
    def find_center(self, pixel_label = None):
        """
        Finds the center of a given electrod within the pixel_label coordinate.
            
            Parameter:
                pixel_label (int): The label of the pixel from which we want to determine the center            
            Return: 
                center_x (int): The x coordinate corresponding to the center of the central pixel in the implant image 
                center_y (int): The y coordinate corresponding to the center of the central pixel in the implant image 
        """

        # if None, it means we are looking for the central pixel
        if pixel_label is None:
            dist_matrix = self.create_distance_matrix()
            pixel_label = self.determine_central_label(dist_matrix)
        
        # Only keep the pixel of interest
        filter_central_pixel = (self.pixel_labels == pixel_label)

        # Create an empty image, except for the central pixel
        hexagon_image = Image.fromarray(filter_central_pixel)
        # Find the bounding box
        bbox = hexagon_image.getbbox()
        # Calculate the center coordinates - the middle between the top corners and middle betwen top and bottom corner
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        return center_x, center_y
    
    def determine_font_size(self, letter_size):
        """
        Determines the font size corresponding to the desired letter size (in image pixels)
     
            Params:
                letter_size (float): The size of capital letter in image pixels
            Returns:
                (ImageDraw.font): The preloaded font with the correct size  
        
        Adapted from this post on Stackoverflow: https://stackoverflow.com/questions/4902198/pil-how-to-scale-text-size-in-relation-to-the-size-of-the-image 
        It is brute force search, but it works.

        Note: All capital letters in the Sloan font have the same bouding box: a square 5 times the stroke's width. Here we use a capital C for calibration.

        -> font.getlength returns the bounding box's length (in image pixel) of our Landolt C, 
        as the bbox is square, we just need to make it as large as the desired letter_size (which is already in image pixel).    
        """    
         
        # Starting font size
        font_size = 1
        try:
            font = ImageFont.truetype(self.font_path, font_size)
        except OSError as error: 
            print(error) 
            print(f"The font file could not be found with the path: {self.font_path}")

        text = "C"
        while font.getlength(text) < letter_size:
            # Iterate until the text size is slightly larger than the criteria
            font_size += 1
            font = ImageFont.truetype(self.font_path, font_size)
        
        return font
    
    ############################ Drawing functions ############################

    def draw_text(self, pattern):
        """
        The function drawing text at the correct location and font size. 
        
            Parameters:
                pattern (ImagePattern.Text): Text to print
            Return:
                actual_position (int, int): The position to use when pasting the letter image into the projected/overlay image
                text_image (PIL.Image): An image of the letter on black background with the correct size and orientation
        """
        
        # Send warning if lower case letters were used with Sloan font (the default)
        if any([x.islower() for x in pattern.text]) and ('Sloan' in self.font_path):
            pattern.text = pattern.text.upper()
            warnings.warn("Lowercase characters do not exist in Sloan font. All letters were converted to uppercase.")

        # Convert sizes to image based pixel
        letter_size = int(pattern.letter_size * (self.scaled_pixel if pattern.unit == "pixel" else self.image_pixel_scale))
        if letter_size <= self.scaled_pixel:
            warnings.warn("The letter size is equal or smaller to the pixel size and will not be resolved.")
        
        # Determine image size
        lines = pattern.text.splitlines()
        n_lines = len(lines)
        max_len = len(max(lines, key=len))
        image_size = (np.ceil(letter_size * max_len).astype(int), np.ceil(letter_size * n_lines).astype(int))
        # Create a rnew image with a transparent background
        text_image = Image.new('RGBA', image_size, (0, 0, 0, 0))

        # Convert to drawing, set the font, and write text
        text_drawing = ImageDraw.Draw(text_image)
        # Find the font size matching the letter size
        font = self.determine_font_size(letter_size)
        text_drawing.text((0, 0), pattern.text, font=font, fill="white", align='center', spacing=0) # TODO check for the desired vertical spacing, if non-zero, increase image_size height by spacing, eventually use ImageDraw.textbox()
        
        # Rotate with expansion of the image (i.e. no crop)
        text_image = text_image.rotate(pattern.rotation, expand = 1)

        # Compute the new position taking into account the user position, the offset due to image size
        # and the new size due to rotation expansion
        actual_position = self.convert_user_position_to_image_position(pattern.position, pattern.unit, text_image.size[0], text_image.size[1])
        
        self.determine_overflow(image_size, max_len, n_lines)

        return self.assemble_drawing(actual_position, text_image)
    
    def draw_grating(self, pattern):
        """
        Draw a rectangular grating of width width_grating spaced (edge to edge) by pitch_grating
        WARNING: The user position only allows for lateral shift

            Parameters:
                pattern (ImagePattern.Grating): Grating to draw 
                
            Returns:
                grating_only (PIL.Image): The image of the grating with alpha transparency enabled
        """
        
        # Convert distances to image based pixels
        width_grating = int(pattern.width_grating * (self.scaled_pixel if pattern.unit == "pixel" else self.image_pixel_scale))
        pitch_grating = int(pattern.pitch_grating * (self.scaled_pixel if pattern.unit == "pixel" else self.image_pixel_scale))
   
        theta = np.deg2rad(pattern.rotation)

        drift = pattern.position[0] * (self.scaled_pixel if pattern.unit == 'pixel' else self.image_pixel_scale)
        offset_x = int( np.cos(theta) * width_grating / 2 ) + drift

        # We want the grating to overlap the central pixel, hence offset by half the grating rotated width
        # offset_x = int( np.cos(theta) * width_grating / 2 ) + pattern.position[0] * self.scaled_pixel

        # Compute the bottom left corner of the grating, from the image center to the right
        fwd_x_pos = np.arange(self.center_x - offset_x, 2 * self.width, width_grating + pitch_grating)
        # Compute the bottom left corner of the grating, from the image center to the left
        bckwd_x_pos = np.arange(self.center_x - offset_x -(width_grating + pitch_grating), - 2 * self.width, - (width_grating + pitch_grating))
        list_x_positions = np.hstack((bckwd_x_pos, fwd_x_pos))

        grating_only = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        drawing_grating = ImageDraw.Draw(grating_only)
        
        # The grating always span the whole Y-axis -> it can be larger than the height, and should be high enough to withstand a rotation
        y_l, y_h = -self.width, self.height + self.width
        # Remember that PIL has the origin at the top left corner
        for x_pos in list_x_positions:
            points = [] 
            points.append(self.rotate_point(x_pos, y_l, theta))  # Top left
            points.append(self.rotate_point(x_pos + width_grating, y_l, theta))  # Top right
            points.append(self.rotate_point(x_pos + width_grating, y_h, theta))  # Bottom right
            points.append(self.rotate_point(x_pos, y_h, theta))  # Bottom left
            # Draw rotated rectangle
            drawing_grating.polygon(points, fill="white", outline=None)
        
        actual_position = (0, 0)

        return self.assemble_drawing(actual_position, grating_only)
    
    def draw_rectangle(self, pattern, fill_color = "white"):
        """
        Draws a rectangle with the given size, at given position and rotation
            Parameters:
                pattern (ImagePattern.Rectangle): The rectangle to draw
                fill_color (string): The filling color
            Returns:
                actual_position (int, int): The position to use when pasting the rectangle image into the projected/overlay image
                rectangle (PIL.Image): The image of the rectangle with alpha transparency enabled
        """
        
        if (pattern.width is None) or (pattern.height is None):
            # If one dimension is None, we do full field illumination
            # Add the pixel size to compensate the offset by the bounding box
            # -> The rectangle is centered on the central pixel and not the center of the image
            width = int(self.width + self.scaled_pixel)
            height = int(self.height + self.scaled_pixel)
        else:
            # Convert distances to image based pixels
            width = int(pattern.width * (self.scaled_pixel if pattern.unit == "pixel" else self.image_pixel_scale))
            height = int(pattern.height * (self.scaled_pixel if pattern.unit == "pixel" else self.image_pixel_scale))
        
        rectangle = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rectangle)
        draw.rectangle([0, 0, width, height], fill=fill_color)

        # Rotate with expansion of the image (i.e. no crop)
        rectangle = rectangle.rotate(pattern.rotation, expand = 1)

        # Update the size due after rotation due to the expansion
        width = rectangle.size[0]
        height = rectangle.size[1]

        actual_position = self.convert_user_position_to_image_position(pattern.position, pattern.unit, width, height)
        self.assemble_drawing(actual_position, rectangle)

    def draw_circle(self, pattern):
        """
        Draws a circle with the given diameter at given position
            Parameters:
                pattern (ImagePattern.Circle): Circle to draw
            Returns:
                actual_position (int, int): The position to use when pasting the rectangle image into the projected/overlay image
                rectangle (PIL.Image): The image of the rectangle with alpha transparency enabled
        """
        # Convert distances to image based pixels
        diameter = int(pattern.diameter * (self.scaled_pixel if pattern.unit == "pixel" else self.image_pixel_scale))
        
        circle = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
        draw = ImageDraw.Draw(circle)
        draw.ellipse([0, 0, diameter-1, diameter-1], fill="white")
        actual_position = self.convert_user_position_to_image_position(pattern.position, pattern.unit, diameter, diameter)
        self.assemble_drawing(actual_position, circle)
    
    def generate_overlayed_pattern(self, pil_img):
        """
        Generate a semi transparent image of the pattern for the overlayed image containing the pixel layout.
        Changing the transparency of the foreground without modifying the transparency of the background (here 0)
        is tricky with Pillow. It is easier to do it with numpy arrays.

        Params:
            pil_img (PIL.Image (RGBA)): The image containing the pattern only
        """
        
        array_img = np.array(pil_img)
        # Select all pixel/entries with transparency superior to 0 (i.e. all non-background)
        location_pattern = array_img[:,:,3] > 0
        # Set the opacity of the pattern to semitransparent 
        array_img[location_pattern, 3] = self.opacity
        return Image.fromarray(array_img)
    
    def assemble_drawing(self, actual_position, to_be_projected):
        """
        This functions adds a pattern to the projected image and overlayed background image. 
            Parameters:
                actual_position (int, int): The position to use when pasting the rectangle image into the projected/overlay image
                to_be_projected (PIL.Image): The pattern to be projected with alpha transparency enabled, 
        """
        
        # Create the final images
        self.projected.paste(to_be_projected, actual_position, to_be_projected)
        # Create a semitransparent overlay onto the pixel layout
        to_be_overlayed = self.generate_overlayed_pattern(to_be_projected)
        self.background_overlay.paste(to_be_overlayed, actual_position, to_be_overlayed)

        # Add the red frame around the projected image
        drawing_projection = ImageDraw.Draw(self.projected)
        drawing_projection.rectangle([0, 0, self.width - 1, self.height - 1], outline="red", width=2)
        # plt.rcParams['figure.facecolor'] = 'white'
        # plt.subplot(121)
        # plt.imshow(np.asarray(self.projected))
        # plt.title("Pattern Visualization")
        # plt.axis('off')

        # plt.subplot(122)
        # plt.imshow(np.asarray(self.background_overlay))
        # plt.title("Pattern Overlaid Visualization")
        # plt.axis('off')
        # plt.show()


########################### Classes for defining the patterns ###########################   


class Pattern():
    """
    Superclass defining the basics of a pattern. 
    Attributes:
        position (float, float): The pattern's center with respect to the central pixel in the layout  
        rotation (float): The clockwise rotation of the pattern in degrees
        unit (float): Either 'pixel' or 'um'. Pixel for features which are multiples of the pixel size, or um for micrometers.      
    """
    def __init__(self, postion = (0, 0), rotation = 0, unit = "pixel"):
        self.position = postion
        self.rotation = rotation
        
        if not isinstance(unit, str):
            raise ValueError(f"The unit should be of a type str, and equal to 'um' or 'pixel', not '{unit}'.")
        elif (unit != "pixel" and unit != "um"):
            raise ValueError(f"The provided unit should be 'um' or 'pixel' and not '{unit}'.")
        self.unit = unit

    def __str__(self):
        pass

    def draw(self, drawing_board):
        pass        


class Text(Pattern):
    """
    Class defining the parameters for drawing a text. 
    Attributes:
        position ((float, float)): The position with respect to the central pixel as multiple of the elec. pixel size. Half pixel means going toward the diagonal
        rotation (float): Clockwise rotation of the text in degrees
        unit (float): Either 'pixel' or 'um'. Pixel for features which are multiple of the pixel size, or um for micrometers.      
        text (string): The text to be projected 
        letter_size (float): The letter's size as a multiple of the elec. pixel size
        gap_size (float): When using Landolt C, the gap opening as a multiple of the pixel size

        Note: if gap size is specified, it overwrites the provided letter size!
        
        The Sloan font (which contains the Landolt C) use a square bounding box for each letter.
        The bbox is five times the width of the stroke (for the capital, I am not sure for the lower case). 
        The Landolt C has a gap width the size of a stroke width.
        
        To have a Landolt C with 1-pixel (sometimes refered to as pixel here) opening:
            -> as the gap width = stroke width
            -> and the bbox is 5 times the stroke width
            -> the letter size should be 5 times the gap size
            -> the computation are done in image pixel through self.scaled_pixel or self.image_pixel_scale
    """    
    def __init__(self, position = (0, 0), rotation = 0, text = "C", unit = "pixel", letter_size = 5, gap_size = None):
        super().__init__(position, rotation, unit)
        self.letter_size = letter_size
        self.text = text
        
        # Gap opening overwrites letter size
        if gap_size is not None:
            self.letter_size = 5 * gap_size
    
    def __str__(self):
        return f"User position: {self.position}\nRotation: {self.rotation}\nLetter size {self.letter_size}\nText {self.text}"

    def draw(self, drawing_board):
        drawing_board.draw_text(self)


class Grating(Pattern):
    """
    Class defining the parameters for drawing a grating. 
    Attributes:
        position (float, int): The position of the grating with respect to the central pixel, move with respect to the pixel size along the X-axis. Y-axis is not implemented yet.  
        rotation (float): The clockwise rotation of the grating in degrees - Between -90° and 90° included 
        unit (float): Either 'pixel' or 'um'. Pixel for features which are multiple of the pixel size, or um for micrometers.      
        width_grating (int): The width of grating in micron
        pitch_grating (int): The shortest distance separating each grating (edge to edge) in micron
    """ 
    def __init__(self, position = (0, 0), rotation = 45, unit = "pixel", width_grating = 1, pitch_grating = 1):
        super().__init__(position, rotation, unit)

        if (np.abs(rotation) > 90):
            raise ValueError("The rotation angle shoud be between -90° <= rotation <= 90°")
        if position[1] != 0:
            warnings.warn("The Y position cannot be shifted. Y is set to 0 instead.")

        self.width_grating = width_grating
        self.pitch_grating = pitch_grating
    
    def __str__(self):
        return f"User position: {self.position}\nRotation: {self.rotation}\nWidth grating {self.width_grating}\nPitch grating {self.pitch_grating}"
    
    def draw(self, drawing_board):
        drawing_board.draw_grating(self)


class Rectangle(Pattern):
    """ 
    Class defining the parameters for drawing a rectangle. 
    Attributes:          
        position (float, float): The position of the grating with respect to the central pixel  
        rotation (float): The clockwise rotation of the rectangle in degrees
        unit (float): Either 'pixel' or 'um'. Pixel for features which are multiple of the pixel size, or um for micrometers.      
        width (float): The rectangle's width in micron
        height (float): The rectangle's height in micron
    """    
    def __init__(self, position = (0, 5), rotation = 45, unit = "pixel", width  =  100, height = 100):
        super().__init__(position, rotation, unit)
        self.width = width
        self.height = height 
    
    def __str__(self):
        return f"User position: {self.position}\nRotation: {self.rotation}\nWidth {self.width}\nHeight {self.height}"
    
    def draw(self, drawing_board):
        drawing_board.draw_rectangle(self)


class Circle(Pattern):
    """ 
    Class defining the parameters for drawing a circle. 
    Note that the rotation is not used for circles. 
    Attributes:          
        position (float, float): The position of the circle with respect to the central pixel
        diameter (float): The circle diameter in micron
    """  
    def __init__(self, position = (0, 0), unit="pixel", diameter = 200):
        super().__init__(position, rotation = 0, unit=unit)
        self.diameter = diameter

    def __str__(self):
        return f"User position: {self.position}\nRotation: {self.rotation}\nDiameter {self.diameter}"
    
    def draw(self, drawing_board):
        drawing_board.draw_circle(self)


class FullField(Pattern):

    """ 
    Class defining an image covering the full field.
    True: the image is completely white
    False: the image is completly black

    Attributes:  
        fill_color (string): Either black for no activation, or white for full field activation        
    """  
    def __init__(self, fill_color = "black"):
        super().__init__()
        self.color = fill_color

    def __str__(self):
        return f"Full field coverage"
    
    def draw(self, drawing_board):
        drawing_board.draw_rectangle(Rectangle(height=None, width=None, rotation=0, position=(0,0)), fill_color = self.color)


################## Classes for organizing the creation of GIF/video sequences ##################


class Subframe():
    """
    A Subframe contains a list of patterns that is being displayed for a certain duration.

    Attributes:
        duration (float): Projection time in ms, it should be 0 < duration < ProjectionSequence.duration_ms
        patterns (list(ImagePattern)): List of patterns to diplay
    """
    def __init__(self, duration_ms, patterns=[Text(text='C', gap_size=1.2)]):

        if (np.abs(duration_ms) <= 0):
            raise ValueError("The frame duration should be larger than 0 ms!")
        if (len(patterns) == 0):
            raise ValueError("No patterns were provided for the subframe!")
        
        self.duration_ms = duration_ms 
        self.patterns = patterns

    def __iter__(self):
        """
        Iterates over the patterns of the subframe
        """
        return iter(self.patterns)


class Frame():
    """
    A Frame contains a list of subframes to be repeated a certain number of times.

    Attributes:
        repetitions (int): The number of times this frame is repeated in the GIF
        subframes (list(Subframe)): The list of subframes to display
        name (string): Optional, a meaningful name for the Frame
    """

    def __init__(self, repetitions, subframes, name="Default_title"):
        
        if (np.abs(repetitions) < 1):
            raise ValueError("The frame repetition number should be at least 1!")
        if (len(subframes) == 0):
            raise ValueError("No subframes were provided for the Frame!")
        
        self.name = name
        self.repetitions = repetitions
        self.subframes = subframes

    def __iter__(self):
        """
        Iterates over the subframes of the frame
        """
        return iter(self.subframes)
    
    def store_config(self):
        """
        Returns a dictionary containing the configuration required for pattern generation 
        """
        
        config = {"repetitions": self.repetitions, "name": self.name, "Subframes": {}}
        for idx, subframe in enumerate(self):
            config["Subframes"][f"Subframe_{idx+1}.bmp"] = subframe.duration_ms

        return config


class ProjectionSequence():
    """
    This class contains all the information required to create and save a GIF sequence

    Attributes:
        frames (list(Frame)): The frames to be displayed
        intensity_mW_mm2 (float): The intensity of the light projected in mW / mm^2
        frequency_Hz (float): The image frequency (or frame rate) in Hz. 
        time_step_ms (float): The rise and fall time of the current source in ms

        Note that the sum of the subframes' duration should equal the frame period (1 / frequency).
    """

    def __init__(self, frames, intensity_mW_mm2=1.0, frequency_Hz=10, time_step_ms=0.05):
        if (len(frames) == 0):
            raise ValueError("No frames were provided for the projection sequence!")
        if (np.abs(intensity_mW_mm2) <= 0):
            raise ValueError("The laser intensity should be more than 0 mW / mm^2!")
        if (np.abs(frequency_Hz) <= 0):
            raise ValueError("The frame duration should be more than 0 ms!")
        
        self.frames = frames
        self.intensity_mW_mm2 = intensity_mW_mm2
        self.frequency_Hz = frequency_Hz
        self.frame_period_ms = (1 / frequency_Hz)*1000 # in ms
        self.time_step_ms = time_step_ms 

        self.check_duration()

    def __iter__(self):
        """
        Iterates over the frames of the projection sequence
        """
        return iter(self.frames)
    
    def get_script(self):
        """
        Returns a list of list structured similarly to the script used with pre-existing patterns
        """
        
        second_half = []
        max_subframes = []
        # Iterate on the frames
        for frame in self:
            ls = [frame.name, frame.repetitions]
            max_subframes.append(len(frame.subframes))
            # Iterate on the subframes and extract the durations
            for subframe in frame:
                ls.append(subframe.duration_ms)
            second_half.append(ls)
        max_subframes = max(max_subframes)

        data = [
            ['Light Intensity', self.intensity_mW_mm2, 'mW/mm^2'],
            ['Frame period', self.frame_period_ms, 'ms'],
            ['Time step', self.time_step_ms, 'ms'],
            [None,'Frame Repetition'] + [f'Subframe{i}' for i in range(1, max_subframes+1)]
            ]

        data += second_half
        return data
    
    def store_config(self):
        """
        Returns a dictionary containing the configuration required for pattern generation 
        """
        
        config = {"intensity": self.intensity_mW_mm2, "frequency": self.frequency_Hz, "time_step": self.time_step_ms, "Frames": {}}
        for idx, frame in enumerate(self):
            config["Frames"][f"Frame_{idx+1}"] = frame.store_config()

        return config
    
    def check_duration(self):
        """
        Checks whether the sum of subframe duration (exposure time) matches the frame period 
        """
        for frame in self.frames:
            sum_duration = 0
            for subframe in frame:
                sum_duration += subframe.duration_ms
            
            if sum_duration != self.frame_period_ms:
                raise ValueError(f"The sum of subframe duration ({sum_duration}) does not equal the frame period ({self.frame_period_ms}) for frame '{frame.name}'!")
            
    def __str__(self):
        return str(self.store_config())
