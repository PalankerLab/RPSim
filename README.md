RPSIM TOOLBOX
-----------------
[![DOI](https://zenodo.org/badge/508075556.svg)](https://zenodo.org/badge/latestdoi/508075556)

[**RPSIM TOOLBOX**](https://github.com/PalankerLab/RPSim) is a software package designed to simulate the circuit 
dynamics of a photovoltaic retinal prosthetic (RP) with a large number of electrodes. 
This circuit takes the optical pattern projected onto the array as the input, and computes the electric current 
injection as a function of time for each electrode.

The time-varying electric field in the electrolyte is synthesized by linear superposition of the elementary fields pre-computed in [**COMSOL**](https://www.comsol.com/). 
These elementary fields, along with the description of the projection sequence, are required inputs for this tool. 
The kernel circuit simulator is [**Xyce**](https://xyce.sandia.gov/).

The toolbox is implemented in Python and intended to be interfaced via Jupyter Notebook. 

This software was developed by [**Charles Zhijie Chen**](https://web.stanford.edu/~zcchen/) and [**Anna Kochnev Goldstein**](https://www.linkedin.com/in/anna-kochnev-goldstein/) as part of the [**Photovoltaic Retinal Prosthesis Project**](https://web.stanford.edu/~palanker/lab/retinalpros.html) in [**Palanker Lab**](https://web.stanford.edu/~palanker/lab/index.html) at Stanford University. 
It can be adapted for similar applications, such as a non-photovoltaic retinal prosthesis or other multi-electrode 
arrays (MEA) used in neural stimulation.


## Installation
In order to use the toolbox, you can either download the package's zip file or clone the repository. 
*Please note that by downloading the zip you will be able to run the tool, but won't be able to use it as a git 
repository.
If you choose to download the toolbox using the "download" option*, you can skip ahead to step 3. Otherwise, please 
follow the steps below.

1. Using your terminal, clone the repository to the directory of your choice:

    ```bash
    cd <chosen directory>
   ```
   
   ```bash
    git clone https://github.com/PalankerLab/RPSim
   ```
   
   ```bash
    cd RPSim
    ```

2. Install Git LFS:

   If Homebrew is not installed, first run (*Please note that Homebrew is a package manager for MacOS and Linux):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
   
   And then execute:
   ```bash
   brew install git-lfs
   ```
   
   Check that the installation was successful:
   ```bash
   git lfs install
   ```


3. Install Python:
   Download Python 3.9 or higher from the official website and follow the installation instructions: https://www.python.
   org/downloads/


4. Install Xyce:
   
   Download Xyce from the official website and follow the installation instructions: 
   **https://xyce.sandia.gov/downloads/Binaries.html** 
   
   Then, run the following to append the Xyce executable to the system PATH variable.
   Please make sure to adjust the version number in the command below to the downloaded executable.

   ```bash
   echo -e "\n export PATH=\"/usr/local/Xyce-Release-7.4.0-NORAD/bin:\$PATH\"" >> ~/.bash_profile
   ```
   
   ```bash
   source ~/.bash_profile
   ```
   
   If you run the following command, and get the executable path above in the output, this means the installation 
   was successful:
   ```bash
   which Xyce
   ```


5. Install Jupyter Notebook or Jupyter Lab: 
   
   If Homebrew is not installed, first run (*Please note that Homebrew is a package manager for MacOS and Linux*):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
  
   For Jupyter Lab, execute:
   ```bash
   brew install jupyterlab
   ```
   
   For Jupyter Notebook, execute:
   ```bash
   brew install jupyter
   ```
   
   *If you are not using MacOS/Linux or running into difficulties, you can also follow the instructions on the official 
   website: https://jupyter.org/install 



6. Go into the tool's main directory and run Jupyter Notebook:
   
   ```bash
    cd RPSim/
   ```

   If you installed Jupyter Notebook, execute:

   ```bash
    jupyter notebook
    ```
   
    If you installed Jupyter Lab, execute:

   ```bash
    jupyter-lab
   ```
   
    This step should open the Jupyter Notebook page in your default browser.


7. Now follow the instructions on the user_interface notebook page.

## User Interface
You can choose to update and edit the tool libraries to accommodate some advanced usages, but for basic applications 
users can interact with the tool only via the **user_interface.ipynb** (the user interface Jupyter Notebook) and the 
**user_files/user_input** directory, which contains the input files for the run.

Our toolbox comes with some initial input files, but these can be supplemented or replaced by any other user-defined input files. 

The output files will be generated in a dedicated output folder under the **user_files/user_output** directory. A new 
folder named with the execution date and time stamp will be created for each run. The prefix of the folder's name 
can be adjusted in the configuration section. The folder will include the execution log, the executed configuration in a .pkl format, the circuit simulation results and exemplary plots showing the currents and voltages for a central and an edge pixel.

## Configuration and Features
The execution of the tool can be configured from the user_interface notebook by changing the different Python dictionary entries. 

A sample section is presented below:

   ```Python
   configuration = {}
   
   configuration["model"]                            = "monopolar"   # model geometry: monopolar or bipolar
   configuration["pixel_size"]                       = 40            # pixel size
   configuration["geometry"]                         = "Flat_human"  # geometry settings: HC/Pillar/Flat devices in pdish/rat/human
   
   # configure input paths (either absolute paths or relative paths within the repository)
   configuration["user_files_path"]                  = None          # If set to None, defaults to inner user_files directory
   configuration["video_sequence_name"]              = "seq15_C-precond2_PS40_30Hz"
   configuration["pixel_label_input_file"]           = f'image_sequence/pixel_label_PS{configuration["pixel_size"]}.pkl'
   
   # define input files for mono-polar arrays
   configuration["monopolar"] = {
       "r_matrix_input_file_diagonal": f'r_matrix/COMSOL_results/{configuration["geometry"]}_PS{configuration["pixel_size"]}_EP_self.csv',
       "r_matrix_input_file_non_diagonal": f'r_matrix/COMSOL_results/{configuration["geometry"]}_PS{configuration["pixel_size"]}_EP_Rmat.csv'
   }
   
   # define input files for bi-polar arrays
   configuration["bipolar"] = {
       "r_matrix_input_file_active": f'r_matrix/COMSOL_results/{configuration["geometry"]}_PS{configuration["pixel_size"]}_UCD_active.csv',
       "r_matrix_input_file_return": f'r_matrix/COMSOL_results/{configuration["geometry"]}_PS{configuration["pixel_size"]}_UCD_return.csv',
       "r_matrix_input_file_return_neighbor": f'r_matrix/COMSOL_results/{configuration["geometry"]}_PS{configuration["pixel_size"]}_UCD_return_neighbor.csv',
       "r_matrix_input_file_px_pos": f'r_matrix/COMSOL_results/PS{configuration["pixel_size"]}-lg_pos.csv'
   }
       
   # configure output paths (either absolute paths or relative paths within the repository)
   configuration["r_matrix_output_file"]             = f'r_matrix/R_{configuration["geometry"]}_PS{configuration["pixel_size"]}.pkl'
   configuration["netlist_output_file"]              = "netlist.sp"
   configuration["output_prefix"]                    = "run"       # this prefix will be added to all generated output directories
   
   
   # R matrix parameters 
   configuration["r_matrix_conductivity"]            = 1
   
   # Pixel layout Parameters
   configuration["photosensitive_area_edge_to_edge"] = 36          # edge-to-edge size of the photosensitive area
   configuration["active_electrode_radius"]          = 9           # radius of the active electrode in um
   configuration["light_to_current_conversion_rate"] = 0.5         # light to current conversion rate in A/W
   configuration["photosensitive_area"]              = None        # total photosensitive area in um. Assign "None" for auto calculation
   
   # Circuit Parameters
   configuration["return_to_active_area_ratio"]      = 5.7525      # monopolar only: ratio between return area and total active area
   configuration["additional_edges"]                 = 142         # bipolar only: edge segments of the return
   configuration["sirof_capacitance"]                = 6           # SIROF capacitance in mF/cm^2
   configuration["initial_Vactive"]                  = .4          # Initial bias of the active electrode in V
   configuration["Ipho_scaling"]                     = 1           # artificial scalar of photocurrent
   configuration["number_of_diodes"]                 = 1           # number of photo diodes per pixel
   configuration["Isat"]                             = 0.3         # diode saturation current in pA
   configuration["ideality_factor"]                  = 1.5         # ideality factor n of the diode
   configuration["shunt_resistance"]                 = 750E3       # shunt resistance in Ohm. Assign "None" if no shunt
   configuration["temperature"]                      = 37
   configuration["nominal_temperature"]              = 25
   configuration["simulation_duration"]              = .7          # simulation duration in seconds
   ```


#### 1. Multiplexing Configurations
Each configuration entry can be either assigned to a single value or a list of values. 

If a list is provided, each value in the list will be treated as a separate configuration entry, and all assembled 
configurations will be executed sequentially in a non-permuting manner. Therefore, all lists must have the same length. This feature enables multiple configurations be launched in one click by providing all requested values in a list form. 

For example, the notation presented below will be interpreted as two different configuration settings, where the 
first run will set the temperature to 25 and simulate for 1 second, and the second run to 37 and simulate for 2 
seconds.

   ```python
   configuration["temperature"]                      = [25, 37]
   configuration["nominal_temperature"]              = 25
   configuration["simulation_duration"]              = [1, 2]
   ```

#### 2. Input and Output Paths
Configuration entries that contains the keywords "input" and "output" will be identified as path-related variables. 
If a given input path is not found as a legitimate absolute path, then the tool will try interpreting the value as a 
relative path under the **user_files/user_input** directory. 
A time-stamped output directory will always be created as explained above. All output files will be placed under 
this directory.

#### 3. Video Sequence Parameters
This section contains the parameters defining the photo-current generated by each pixel (NOT the current injection 
into the electrolyte) as a function of time.

#### 4. Circuit Parameters
This section defines the circuit Xyce would simulate. For detailed explanation of the circuit model and the meaning 
of each parameter, please refer to the [Supplementary Materials](https://www.biorxiv.org/content/10.1101/2021.07.12.452093v2.supplementary-material) of our paper.

## Execution
After proper adjustment of the configuration section, the tool is executed by running the configuration and 
execution section in the notebook:

   ```python
    from RPSim import run_rpsim
    run_rpsim(configuration=configuration, find_similar_runs=True)
   ```

#### 1. Options
**run_stages** - This option enables to control which stages will be executed, defaults to the full flow:
["resistive_mesh", "current_sequence", "circuit", "simulation", "plot_results"]. 
In order to run only the plot_results stage change this variable as follows:

   ```python
    from RPSim import run_rpsim
    run_rpsim(configuration=configuration, run_stages=["plot_results"], find_similar_runs=True)
   ```

As the above flow is initialized midway, the tool will search for the latest available output for the previous 
stages under the user_files/user_output directory. If no previous runs are available, this will result in an error.

The list of stages can be cut at any desired stage, however skipping stages that rely on multiple outputs is 
currently not supported.  

**find_similar_runs** - This option enables the tool to check for existing configuration files in the 
**user_files/user_output** directory in order to issue a warning if the current configuration was already executed. 
It defaults to True.


## Feedback

Please do not hesitate to send us your feedback! 

We are an academic research group, and made this tool available for a collaborative community of scientific research.
If you found a bug, have improvement suggestions, or simply want to connect - please create a new issue or pull 
request, or contact our lab via the provided website!


## Related Publications
Please refer to the following publications for more information:

1. [Wang, Bing-Yi, et al. "Electronic “photoreceptors” enable prosthetic vision with acuity matching the natural 
resolution in rats." *bioRxiv* (2021).](https://doi.org/10.1101/2021.07.12.452093)

## License

MIT License

## Disclaimer
This package was created for our research purposes and distributed as an open source for the benefit of the 
scientific community. It is NOT intended for medical treatment or diagnosis and DOES NOT come with any kind of 
warranty. We hereby disclaim any liability related to the direct or indirect usage of this software.  


