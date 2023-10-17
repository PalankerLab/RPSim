import pickle
import numpy as np
import os
import warnings


from PIL import Image, ImageDraw, ImageFont

from configuration.stages import RunStages
from configuration.configuration_manager import Configuration

from run_stages.common_run_stage import CommonRunStage

class PatternGenerationStage(CommonRunStage):
    """
    This class implements the automated pattern generation from a configuration file.

    Attributes:
        electrode_size (int): The electrode size in micron - More precisely the pitch between electrodes
        implant_layout (PIL.Image): The image of the electrodes layout
        electrode_label (Numpy.array): Array having the same size as implant_layout. Each entry corresponds to either 0 (no electrode at this pixel) or the electrode label (1 to max number).
        
        center_x, center_y (int, int): The pixels coordinate in implant_layout corresponding the center of the central hexagon
        width, height (int, int): The size in pixels (and microns) of the implant_layout image
        
        backgroud_overlay (PIL.Image): The electrode pattern as background, and the projected image as overlay
        projected (PIL.Image): A black screen as background, and the projected image as overlay
    """

    def __init__(self, *args):
        super().__init__(*args)

        self.electrode_size = None # To be added to the arguments

        self.implant_layout = self.load_file(f"Grid_PS{self.electrode_size}.png")
        self.electrode_label = self.load_file(f"pixel_label_PS{self.electrode_size}.pkl")

        self.center_x, self.center_y = self.find_center()
        self.width, self.height = self.implant_layout.size[0], self.implant_layout.size[1]

        self.background_overlay = self.implant_layout.copy()
        self.projected = Image.new("RGB", self.background_overlay.size, "black")

    def __str__(self):

        return self.background_overlay.show(), self.projected.show()
    
    def run_stages(self, *args, **kwargs):
        
        self.draw(**kwargs)
        return self.background_overlay, self.projected
    
    ############################ Helper functions ############################
   
    def load_file(self, file):
        """
        Loads the implant layout image or it's label
        Parameters:
            file (string): The name of the file to load, the rest of the path is assumed
        """
        path = os.path.join(".." , "user_files", "user_input", "image_sequence", file) 
        try:
            with open(path, 'rb') as f:
                if "png" in file:
                    file = Image.open(f, 'r')    
                elif "pkl" in file:
                    file = pickle.load(f)
                else:
                    raise NotImplementedError     
        except OSError as error: 
            print(error) 
            print(f"The file {file} could not be found with path: {path}") 
            
        return file

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

    def determine_overflow(self, text_size, max_len, n_lines):
        """
        This functions determines whether the text will overflow at position (0, 0) and rotation 0°.
        """
        
        # TODO put all the code inside of a class, and make the image size an attribute
        # Add pixel size as a safety margin
        img_size = self.width - self.electrode_size, self.height - self.electrode_size
        if text_size[0] > img_size[0]:
            recommended_size = np.floor(img_size[0] / (self.electrode_size * max_len))
            warnings.warn(f"Your text may be too wide, we recommend using a letter size of at least: {recommended_size}")
        if text_size[1] > img_size[1]:
            recommended_size = np.floor(img_size[1] / (self.electrode_size * n_lines))
            warnings.warn(f"Your text may be too large, we recommend using a letter size of at least: {recommended_size}")  
    
    ############################ Functions for computing position and sizes ############################
    
    def convert_user_position_to_actual(self, user_position, width, height):
        """
        Convert the user position to the actual position where the object should be located in the final image.
            Parameters:
                user_position ((float, float)): The position with respect to the central electrode as multiple of the electrode size. 1.5 along X, means going toward the electrode in diagonal
                width (int): The width of the object to paste in the final image
                height (int): The height of the object to paste in the final image

            Return:
                actual_position (int, int): The position where the image should be located on the projected/overlay image
        """

        # Move the letter by multiple of the pixel size as requested by the user
        center_x, center_y = self.center_x, self.center_y
        shift_x, shift_y = user_position
        # TODO the translation is not perfect, double check how to adjust it    
        # As the pattern is a honeycomb and not a grid, half pixel should be taken into account
        rounded_x = np.round(shift_x)
        if shift_x == rounded_x: # TODO do more test to check whether this calculation make sense
            center_x += shift_x * self.electrode_size
        else:
            center_x += (rounded_x - np.sign(shift_x) * np.sin(np.pi * 30/180)) * self.electrode_size 
        
        center_y +=  shift_y * np.cos(np.pi * 30/180) * self.electrode_size
        
        # Take into account the offset due to the object size, as PIL aligns 
        # from the top-left corner and not from the center of the object
        return int(center_x - width/2), int(center_y - height/2)
        
    def create_distance_matrix(self):
        """
        Returns a distance matrix (Manhattan distance) from the center of the label file for a given electrode size.
        If the number of columns is even, the 0 will be centered to the right
            (e.g. for 4 columns the distance is [-2, -1, 0, 1])
        Same principle for an even number of rows, the zero will be centered towards the bottom

        Returns:
            dist_matrix (Numpy.array): same shape as label, where each matrix entry is the Manhattan distance to the center
        """

        # Number of rows and columns
        rows, columns = self.electrode_label.shape
        vertical_dist, horizontal_dist = np.mgrid[np.ceil(-columns/2):np.ceil(columns/2), np.ceil(-rows/2):np.ceil(rows/2)]
        dist_matrix = np.abs(vertical_dist) + np.abs(horizontal_dist)

        return dist_matrix
    
    def determine_central_label(self, dist_matrix):
        """
            Return the label of the central electrode, which is later used to determine the center of the central electdoe.
                Parameters:
                    dist_matrix (Numpy.array): same shape as label, where each matrix entry is the Manhattan distance to the center

                Return:
                    central_label (int): the label of the central label
        """
        # Start by selecting the non-zero pixels (i.e. the photodiode)
        filter_electrode = self.label != 0 
        # Find the distance of the closest electrode (i.e. the smallest value in dist matrix which non-zero in selection)
        min_dist = np.min(dist_matrix[filter_electrode])
        # Select all the image pixels at that distance
        filter_distance = dist_matrix == min_dist
        central_labels = self.label[filter_distance]
        # Only keep non-zero labels 
        central_labels = central_labels[central_labels != 0]
        # These are all the electrode pixels located at the same distance to the center, draw the first one
        
        # TODO check whether we can improve this criterion
        return central_labels[0]
    
    def find_center(self):
        """
        Finds the center of the central electrode for a certain electrode size pattern.

            Return: 
                center_x (int): The x coordinate corresponding to the center of the central electrode in the implant image 
                center_y (int): The y coordinate corresponding to the center of the central electrode in the implant image 
        """

        dist_matrix = self.create_distance_matrix()
        central_label = self.determine_central_label(dist_matrix)

        # Only keep the central electrode
        filter_central_electrode = (self.electrode_label == central_label)

        # Create an empty image, except for the central electrode
        hexagon_image = Image.fromarray(filter_central_electrode)
        # Find the bounding box
        bbox = hexagon_image.getbbox()
        # Calculate the center coordinates - the middle between the top corners and middle betwen top and bottom corner
        center_x = int((bbox[0] + bbox[2]) / 2)
        center_y = int((bbox[1] + bbox[3]) / 2)

        return center_x, center_y
    
    def determine_font_size(self, letter_size):
        """
        Determines the adequate font size such that a Landolt C span a certain
        fraction of an electrode pixel. Currently the fraction is fixed to 1.
        
        Adapted from this post on Stackoverflow: https://stackoverflow.com/questions/4902198/pil-how-to-scale-text-size-in-relation-to-the-size-of-the-image 
        It is brute force, and it could be imporved, but it works.  

            Params:
                letter_size (float): The size of capital letter as a multiple of the pixel size

            Returns:
                (ImageDraw.font): The preloaded font with the correct size  
        """    
        
        # All capital letters have the same size (or bounding box), and we want to calibrate the 
        # font size according to capital letters. We don't want to calibrate on lower-case or punctuation characters. 
        text = "C"
        image_width = self.implant_layout.size[0]
        
        # Image fraction represents the proportion the text should take with respect to the image.
        # In this case, the letter size are multiples of the electrode pixel size. 
        # WARNING the assumption is that the implant_layout PNG image is scaled such that 1 pixel = micron, which seems true after measuring the images in ImageJ
        img_fraction = self.electrode_size * letter_size / image_width 

        # Starting font size
        font_size = 1
        font_name = "Sloan.otf"
        path = os.path.join(".." , "utilities", font_name) 
        try:
            font = ImageFont.truetype(path, font_size)
        except OSError as error: 
            print(error) 
            print(f"The font {font_name} could not be found with the path: {path}")

        while font.getlength(text) < img_fraction * image_width:
            # Iterate until the text size is slightly larger than the criteria
            font_size += 1
            font = ImageFont.truetype(path, font_size)
        
        return font
    
    ############################ Drawing functions ############################

    def draw_text(self, letter_size, user_position, text, rotation):
        """
        The function computing the font size and position of the text. 
        
            Parameters:
                letter_size (float): The letter's size as a multiple of the elec. pixel size
                user_position ((float, float)): The position with respect to the central electrode as multiple of the elec. pixel size. Half electrode means going toward the diagonal
                text (string): The text to be projected 
                rotation (float): Clockwise rotation of the text in degrees
            
            Return:
                actual_position (int, int): The position to use when pasting the letter image into the projected/overlay image
                text_image (PIL.Image): An image of the letter on black background with the correct size and orientation
        """
        
        # Determine image size
        lines = text.splitlines()
        n_lines = len(lines)
        max_len = len(max(lines, key=len))
        image_size = (self.electrode_size * letter_size * max_len, self.electrode_size * letter_size * n_lines)
        # Create a rnew image with a transparent background
        text_image = Image.new('RGBA', image_size, (0, 0, 0, 0))

        # Convert to drawing, set the font, and write text
        text_drawing = ImageDraw.Draw(text_image)
        # Find the font size matching the electrode pixel size
        font = self.determine_font_size(letter_size)
        text_drawing.text((0, 0), text, font=font, fill="white", align='center', spacing=0) # TODO check for the desired spacing, if non-zero, increase image_size height by spacing, eventually use ImageDraw.textbox() 
        
        # Rotate with expansion of the image (i.e. no crop)
        text_image = text_image.rotate(rotation, expand = 1)

        # Compute the new position taking into account the user position, the offset due to image size
        # and the new size due to rotation expansion
        actual_position = self.convert_user_position_to_actual(user_position, text_image.size[0], text_image.size[1])
        
        self.determine_overflow(image_size, max_len, n_lines)

        return actual_position, text_image
    
    def draw_grating(self, width_grating, pitch_grating, rotation, user_position):
        """
        Draw a rectangular grating of width width_grating spaced (edge to edge) by pitch_grating
        WARNING: The user position only allows for lateral shift

            Parameters:
                width_grating (int): The width of grating in micron
                pitch_grating (int): The shorted distance separating each grating (edge to edge) 
                rotation (float): The clockwise rotation of the grating in degrees - Between -90° and 90° included
                user_position (float, int): The position of the grating with respect to the central pixel, move with respect to the pixel size along the X-axis. Y-axis is not implemented yet.  
            
            Returns:
                grating_only (PIL.Image): The image of the grating with alpha transparency enabled
        """
        
        if (np.abs(rotation) > 90):
            raise ValueError("The rotation angle shoud be between -90° <= rotation <= 90°")
        if user_position[1] != 0:
            warnings.warn("The Y position cannot be shifted. Y is set to 0 instead.")

        theta = np.deg2rad(rotation)

        # We want the grating to overlap the central pixel, hence offset by half the grating rotated width
        offset_x = int( np.cos(theta) * width_grating / 2 ) + user_position[0] * self.electrode_size

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
        
        return grating_only
    
    def draw_rectangle(self, user_position, rotation, width, height):
        """
        Draws a rectangle with the given size, at given position and rotation
            Parameters:
                user_position (float, float): The position of the grating with respect to the central pixel  
                rotation (float): The clockwise rotation of the rectangle
                width (float): The rectangle's width in micron
                height (float): The rectangle's height in micron
            Returns:
                actual_position (int, int): The position to use when pasting the rectangle image into the projected/overlay image
                rectangle (PIL.Image): The image of the rectangle with alpha transparency enabled
        """
        
        rectangle = Image.new("RGBA", (self.width, self.height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rectangle)
        draw.rectangle([0, 0, width, height], fill="white")

        # Rotate with expansion of the image (i.e. no crop)
        rectangle = rectangle.rotate(rotation, expand = 1)

        # Compute the new position taking into account the offset required by the user, and the new size due to rotation expansion
        width = rectangle.size[0]
        height = rectangle.size[1]

        actual_position = self.convert_user_position_to_actual(user_position, width, height)
        return actual_position, rectangle

    def draw_circle(self, user_position, diameter):
        """
        Draws a circle with the given diameter at given position
            Parameters:
                user_position (float, float): The position of the grating with respect to the central pixel  
                diameter (float): The circle diameter in micron
            Returns:
                actual_position (int, int): The position to use when pasting the rectangle image into the projected/overlay image
                rectangle (PIL.Image): The image of the rectangle with alpha transparency enabled
        """
        
        circle = Image.new("RGBA", (diameter, diameter), (0, 0, 0, 0))
        draw = ImageDraw.Draw(circle)
        draw.ellipse([0, 0, diameter, diameter], fill="white")

        actual_position = self.convert_user_position_to_actual(user_position, diameter, diameter)
        return actual_position, circle
    
    def draw(self, **kwargs):
        """
        Call the lower-level functions for drawing the patterns on two images: the overlayed background and projected image  
            Parameters:
                kwargs (misc): to be forwarded to the lower-level functions
            
        """
        
        type = kwargs["type"]
        if type == "text":
            if len(kwargs["text"]) == 0:
                raise ValueError("No text was provided!") 
            actual_position, to_be_projected = self.draw_text(kwargs["letter_size"], kwargs["user_position"], kwargs["text"], kwargs["rotation"]) 
        elif type == "grating":
            to_be_projected = self.draw_grating(kwargs["width_grating"], kwargs["pitch_grating"], kwargs["rotation"], kwargs["user_position"])
            actual_position = (0, 0)
        elif type=="rectangle":
            actual_position, to_be_projected = self.draw_rectangle(kwargs["user_position"], kwargs["rotation"], kwargs["width"], kwargs["height"])
        elif type=="circle":
            actual_position, to_be_projected = self.draw_circle(kwargs["user_position"], kwargs["diameter"])
        else:
            raise NotImplementedError
        
        # Create the final images
        self.background_overlay.paste(to_be_projected, actual_position, to_be_projected)
        self.projected.paste(to_be_projected, actual_position, to_be_projected)

        # Add the red frame around the projected image
        drawing_projection = ImageDraw.Draw(self.projected)
        drawing_projection.rectangle([0, 0, self.width - 1, self.height - 1], outline="red", width=2)