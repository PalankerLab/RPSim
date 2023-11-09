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
        electrode_size (int): The electrode size in micron - More precisely the pitch between electrodes
        implant_layout (PIL.Image): The image of the electrodes layout
        electrode_labels (Numpy.array): Array having the same size as implant_layout. Each entry corresponds to either 0 (no electrode at this pixel) or the electrode label (1 to max number).
        
        center_x, center_y (int, int): The pixels coordinate in implant_layout corresponding the center of the central hexagon
        width, height (int, int): The size in pixels (and microns) of the implant_layout image
        
        backgroud_overlay (PIL.Image): The electrode pattern as background, and the projected image as overlay
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

            ### Object based implementation
            
            projection_sequence = Configuration().params["patterns_to_generate"]
            self.script = projection_sequence.get_script()

            for frame in projection_sequence:
                # Row containing the script information (compatible with previous implementation)
                row = []
                list_tmp_bmp = []
                list_tmp_array = []

                # Add the name and number of repetitions to the script
                row.append(frame.name)
                row.append(frame.repetitions)

                for idx, subframe in enumerate(frame):
                    # Several patterns can be added to a drawing_board / subframe

                    drawing_board = ImagePattern(electrode_size = Configuration().params["pixel_size"])
                    for pattern in subframe:
                        # Draw the provided pattern
                        pattern.draw(drawing_board)

                    # Save subframe
                    row.append(subframe.duration)
                    list_tmp_bmp.append((f'Subframe{int(idx+1)}', drawing_board.save_as_PIL()))
                    list_tmp_array.append(drawing_board.save_as_array())
                
                # Save frame
                self.script.append(row)
                self.dict_PIL_images[frame.name] = list_tmp_bmp
                self.list_subframes_as_ndarray.append(list_tmp_array)
            
            # Current Sequence stage uses the ndarray frames and script, the PIL images are for the user only
            return [self.list_subframes_as_ndarray, self.script, self.dict_PIL_images]                        
        else:
            # If we load pre-existing patterns, we do not need to process anything
            return []  
    

class ImagePattern():
    """
    A class which stores and draws the PIL.Image containing the patterns to be projected and the patterns overlayed on the implant layout.

    Attributes:
        electrode_size (int): The electrode size in micron - More precisely the pitch between electrodes
        implant_layout (PIL.Image): The image of the electrodes layout
        electrode_labels (Numpy.array): Array having the same size as implant_layout. Each entry corresponds to either 0 (no electrode at this pixel) or the electrode label (1 to max number).
        
        center_x, center_y (int, int): The pixels coordinate in implant_layout corresponding the center of the central hexagon
        width, height (int, int): The size in pixels (and microns) of the implant_layout image
        
        backgroud_overlay (PIL.Image): The electrode pattern as background, and the projected image as overlay
        projected (PIL.Image): A black screen as background, and the projected image as overlay

        pixel_scale (float): The scale in pixel/micron of the implant layout's PNG image
        scaled_electrode (int): The size of the electrode in image-pixel 
    """

    def __init__(self, electrode_size):

        self.electrode_size = electrode_size

        suffix = Configuration().params["pixel_size_suffix"] # Whether we use the large file "_lg" or not
        self.implant_layout = self.load_file(f"Grid_PS{self.electrode_size}{suffix}.png")
        self.electrode_labels = self.load_file(f"pixel_label_PS{self.electrode_size}{suffix}.pkl")

        self.center_x, self.center_y = self.find_center()
        self.width, self.height = self.implant_layout.size[0], self.implant_layout.size[1]

        self.background_overlay = self.implant_layout.copy()
        self.projected = Image.new("RGB", self.background_overlay.size, "black")
        
        self.pixel_scale = self.find_image_scale()
        self.scaled_electrode = int(self.electrode_size * self.pixel_scale)

    def __str__(self):

        self.background_overlay.show()
        self.projected.show()
        return "Printed the images!"
    
    def show(self):
        """
        Displays the overlay and projected image as a subplot.
        """
        fig, axes = plt.subplots(1,2, figsize=(12, 20))
        axes[0].imshow(np.array(self.background_overlay))
        axes[0].set_title("Overlayed")
        axes[1].imshow(np.array(self.projected))
        axes[1].set_title("Projected")
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
        img_size = self.width - self.scaled_electrode, self.height - self.scaled_electrode
        if text_size[0] > img_size[0]:
            recommended_size = np.floor(img_size[0] / (self.scaled_electrode * max_len))
            warnings.warn(f"Your text may be too wide, we recommend using a letter size of at least: {recommended_size}")
        if text_size[1] > img_size[1]:
            recommended_size = np.floor(img_size[1] / (self.scaled_electrode * n_lines))
            warnings.warn(f"Your text may be too large, we recommend using a letter size of at least: {recommended_size}")  

    def find_image_scale(self):
        """
        Determines the scale of the implant layout image.
        Return:
            - pixel_scale (float)

        Example with two different images:
            - For Grid_PS100.png: 
                - Distance in pixels between electrode center to center: 100
                - Distance in microns: 100
                - pixel_scale: 100/100 = 1 pixel/micron
            - For Grid_PS100_lg.png
                - Distance in pixels between electrode center to center: 75
                - Distnace in microns: 100
                - pixel_scale: 75/100 = 0.75 pixel/micron

        Warning! This function assumes that electrode number 1 and 2 are next to each other!
        TODO make the computation of the distance between two electrode more robust 
        Note: I assumed that an error of 2 pixels between the actual and image distance is not significant. 
        """ 

        x1, y1 = self.find_center(1)
        x2, y2 = self.find_center(2)
        pitch_electrode_in_pixels = np.sqrt( (x1 - x2)**2 + (y1 - y2)**2)

        if np.abs(pitch_electrode_in_pixels - self.electrode_size) > 2:
            return pitch_electrode_in_pixels / self.electrode_size
        else:
            return 1 
    
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
            center_x += shift_x * self.scaled_electrode
        else:
            center_x += (rounded_x - np.sign(shift_x) * np.sin(np.pi * 30/180)) * self.scaled_electrode
        
        center_y +=  shift_y * np.cos(np.pi * 30/180) * self.scaled_electrode
        
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
        rows, columns = self.electrode_labels.shape
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
        filter_electrode = self.electrode_labels != 0 
        # Find the distance of the closest electrode (i.e. the smallest value in dist matrix which non-zero in selection)
        min_dist = np.min(dist_matrix[filter_electrode])
        # Select all the image pixels at that distance
        filter_distance = dist_matrix == min_dist
        central_labels = self.electrode_labels[filter_distance]
        # Only keep non-zero labels 
        central_labels = central_labels[central_labels != 0]
        # These are all the electrode pixels located at the same distance to the center, draw the first one
        
        # TODO check whether we can improve this criterion
        return central_labels[0]
    
    def find_center(self, electrode_label = None):
        """
        Finds the center of a given electrod within the pixel_label coordinate.
            
            Parameter:
                electrode_label (int): The label of the electrode from which we want to determine the center            
            Return: 
                center_x (int): The x coordinate corresponding to the center of the central electrode in the implant image 
                center_y (int): The y coordinate corresponding to the center of the central electrode in the implant image 
        """

        # if None, it means we are looking for the central electrode
        if electrode_label is None:
            dist_matrix = self.create_distance_matrix()
            electrode_label = self.determine_central_label(dist_matrix)
        
        # Only keep the electrode of interest
        filter_central_electrode = (self.electrode_labels == electrode_label)

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
        
        # WARNING the assumption is that the implant_layout PNG image is scaled such that 1 pixel = micron,  

        img_fraction = self.scaled_electrode * letter_size / image_width 

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

    def draw_text(self, user_position, rotation, text, letter_size):
        """
        The function computing the font size and position of the text. 
        
            Parameters:
                user_position ((float, float)): The position with respect to the central electrode as multiple of the elec. pixel size. Half electrode means going toward the diagonal
                rotation (float): Clockwise rotation of the text in degrees
                text (string): The text to be projected 
                letter_size (float): The letter's size as a multiple of the elec. pixel size
            
            Return:
                actual_position (int, int): The position to use when pasting the letter image into the projected/overlay image
                text_image (PIL.Image): An image of the letter on black background with the correct size and orientation
        """
        
        # Determine image size
        lines = text.splitlines()
        n_lines = len(lines)
        max_len = len(max(lines, key=len))
        image_size = (self.scaled_electrode * letter_size * max_len, self.scaled_electrode * letter_size * n_lines)
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

        return self.assemble_drawing(actual_position, text_image)
    
    def draw_grating(self, user_position, rotation, width_grating, pitch_grating):
        """
        Draw a rectangular grating of width width_grating spaced (edge to edge) by pitch_grating
        WARNING: The user position only allows for lateral shift

            Parameters:
                user_position (float, int): The position of the grating with respect to the central pixel, move with respect to the pixel size along the X-axis. Y-axis is not implemented yet.  
                rotation (float): The clockwise rotation of the grating in degrees - Between -90° and 90° included
                width_grating (int): The width of grating in micron
                pitch_grating (int): The shorted distance separating each grating (edge to edge) 
                
            Returns:
                grating_only (PIL.Image): The image of the grating with alpha transparency enabled
        """
        
        theta = np.deg2rad(rotation)

        # We want the grating to overlap the central pixel, hence offset by half the grating rotated width
        offset_x = int( np.cos(theta) * width_grating / 2 ) + user_position[0] * self.scaled_electrode

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
    
    def draw_rectangle(self, user_position = (0, 0), rotation = 0, width = None, height = None, fill_color = "white"):
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
        
        # Use full field illumination if the dimensions are not specified
        # Add electrode size due to the centering on electrode and not actual center!
        if width is None:
            width = self.width + self.scaled_electrode
        if height is None:
            height = self.height + self.scaled_electrode
        
        rectangle = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(rectangle)
        draw.rectangle([0, 0, width, height], fill=fill_color)

        # Rotate with expansion of the image (i.e. no crop)
        rectangle = rectangle.rotate(rotation, expand = 1)

        # Update the size due after rotation due to the expansion
        width = rectangle.size[0]
        height = rectangle.size[1]

        actual_position = self.convert_user_position_to_actual(user_position, width, height)
        self.assemble_drawing(actual_position, rectangle)

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
        draw.ellipse([0, 0, diameter-1, diameter-1], fill="white")
        actual_position = self.convert_user_position_to_actual(user_position, diameter, diameter)
        self.assemble_drawing(actual_position, circle)
    
    def assemble_drawing(self, actual_position, to_be_projected):
        """
        This functions adds pattern to the image to be projected and overlayed background. 
            Parameters:
                actual_position (int, int): The position to use when pasting the rectangle image into the projected/overlay image
                to_be_projected (PIL.Image): The pattern to be projected with alpha transparency enabled,
            
        """
        
        # Create the final images
        self.background_overlay.paste(to_be_projected, actual_position, to_be_projected)
        self.projected.paste(to_be_projected, actual_position, to_be_projected)

        # Add the red frame around the projected image
        drawing_projection = ImageDraw.Draw(self.projected)
        drawing_projection.rectangle([0, 0, self.width - 1, self.height - 1], outline="red", width=2)

    

class Pattern():
    """
    Superclass defining the basics of a pattern. 
    Attributes:
        user_position (float, float): The pattern's center with respect to the central pixel in the layout  
        rotation (float): The clockwise rotation of the pattern in degrees
    """
    def __init__(self, user_postion = (0, 0), rotation = 0):
        self.user_position = user_postion
        self.rotation = rotation

    def __str__(self):
        pass

    def draw(self, drawing_board):
        pass

class Text(Pattern):
    """
    Class defining the parameters for drawing a text. 
    Attributes:
        user_position ((float, float)): The position with respect to the central electrode as multiple of the elec. pixel size. Half electrode means going toward the diagonal
        rotation (float): Clockwise rotation of the text in degrees
        letter_size (float): The letter's size as a multiple of the elec. pixel size
        text (string): The text to be projected 
    """    
    def __init__(self, user_position = (0, 0), rotation = 0, text = "C", letter_size = 5):
        super().__init__(user_position, rotation)
        self.letter_size = letter_size
        self.text = text
    
    def __str__(self):
        return f"User position: {self.user_position}\nRotation: {self.rotation}\nLetter size {self.letter_size}\nText {self.text}"

    def draw(self, drawing_board):
        drawing_board.draw_text(self.user_position, self.rotation, self.text, self.letter_size)

class Grating(Pattern):
    """
    Class defining the parameters for drawing a grating. 
    Attributes:
        user_position (float, int): The position of the grating with respect to the central pixel, move with respect to the pixel size along the X-axis. Y-axis is not implemented yet.  
        rotation (float): The clockwise rotation of the grating in degrees - Between -90° and 90° included        
        width_grating (int): The width of grating in micron
        pitch_grating (int): The shortest distance separating each grating (edge to edge) in micron
    """ 
    def __init__(self, user_position = (0, 0), rotation = 45, width_grating = 75, pitch_grating = 75):
        super().__init__(user_position, rotation)

        if (np.abs(rotation) > 90):
            raise ValueError("The rotation angle shoud be between -90° <= rotation <= 90°")
        if user_position[1] != 0:
            warnings.warn("The Y position cannot be shifted. Y is set to 0 instead.")

        self.width_grating = width_grating
        self.pitch_grating = pitch_grating
    
    def __str__(self):
        return f"User position: {self.user_position}\nRotation: {self.rotation}\nWidth grating {self.width_grating}\nPitch grating {self.pitch_grating}"
    
    def draw(self, drawing_board):
        drawing_board.draw_grating(self.user_position, self.rotation, self.width_grating, self.pitch_grating)
    
class Rectangle(Pattern):
    """ 
    Class defining the parameters for drawing a rectangle. 
    Attributes:          
        user_position (float, float): The position of the grating with respect to the central pixel  
        rotation (float): The clockwise rotation of the rectangle
        width (float): The rectangle's width in micron
        height (float): The rectangle's height in micron
    """    
    def __init__(self, user_position = (0, 5), rotation = 45, width  = 100, height = 100):
        super().__init__(user_position, rotation)
        self.width = width
        self.height = height # TODO add _um
    
    def __str__(self):
        return f"User position: {self.user_position}\nRotation: {self.rotation}\nWidth {self.width}\nHeight {self.height}"
    
    def draw(self, drawing_board):
        drawing_board.draw_rectangle(self.user_position, self.rotation, self.width, self.height)
    
class Circle(Pattern):
    """ 
    Class defining the parameters for drawing a circle. 
    Note that the rotation is not used for circles. 
    Attributes:          
        user_position (float, float): The position of the circle with respect to the central pixel
        diameter (float): The circle diameter in micron
    """  
    def __init__(self, user_position = (0, 0), diameter = 200):
        super().__init__(user_position, rotation = 0)
        self.diameter = diameter

    def __str__(self):
        return f"User position: {self.user_position}\nRotation: {self.rotation}\nDiameter {self.diameter}"
    
    def draw(self, drawing_board):
        drawing_board.draw_circle(self.user_position, self.diameter)

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
        drawing_board.draw_rectangle(fill_color = self.color)


################## Classes for organizing the creation of GIF/video sequences ##################


class Subframe():
    """
    A Subframe contains a list of patterns that is being displayed for a certain duration.

    Attributes:
        duration (float): Projection time in ms, it should be 0 < duration < ProjectionSequence.duration
        patterns (list(ImagePattern)): List of patterns to diplay
    """
    def __init__(self, duration, patterns):

        if (np.abs(duration) <= 0):
            raise ValueError("The frame duration should be larger than 0 ms!")
        if (len(patterns) == 0):
            raise ValueError("No patterns were provided for the subframe!")
        
        self.duration = duration # TODO add _ms
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


class ProjectionSequence():
    """
    This class contains all the information required to create and save a GIF sequence

    Attributes:
        frames (list(Frame)): The frames to be displayed
        intensity (float): The intensity of the light projected in mW / mm^2
        frame_period (float): The frame period in ms. The reciprocal of the frame rate. Note that the sum of the subframes' duration should equal the frame period.
        time_step (float): In ms, it is an artifact from previous implementation, it can probably discarded
    """

    def __init__(self, frames, intensity=1.0, frame_period=100, time_step=0.05):
        if (len(frames) == 0):
            raise ValueError("No frames were provided for the projection sequence!")
        if (np.abs(intensity) <= 0):
            raise ValueError("The laser intensity should be more than 0 mW / mm^2!")
        if (np.abs(frame_period) <= 0):
            raise ValueError("The frame duration should be more than 0 ms!")
        
        self.frames = frames
        self.intensity = intensity
        self.frame_period = frame_period
        self.time_step = 0.05

        self.check_duration()

    def __iter__(self):
        """
        Iterates over the frames of the projection sequence
        """
        return iter(self.frames)
    
    def get_script(self):
        """
        Returns a script with similar structure to the one used with pre-existing patterns
        """
        script = [
                    ["Light Intensity", self.intensity, "mW/mm^2"],
                    ["Frame period", self.frame_period, "ms"],
                    ["Time step", self.time_step, "ms"]
                ]
        return script
    
    def check_duration(self):
        """
        Checks whether the sum of subframe duration (exposure time) matches the frame period 
        """
        for frame in self.frames:
            sum_duration = 0
            for subframe in frame:
                sum_duration += subframe.duration
            
            if sum_duration != self.frame_period:
                raise ValueError(f"The sum of subframe duration ({sum_duration}) does not equal the frame period ({self.frame_period}) for frame '{frame.name}'!")
