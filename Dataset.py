import os
import numpy as np
from torch.utils.data import Dataset

class Datos_LAB_GFN(Dataset):
    """
    Dataset containing TOF (Time-of-Flight) data obtained at UCM (Spain) by the Group of Nuclear Physics.
    """
    
    def __init__(self, data_dir, positions=None, step_size=0.066):
       """
       Initializes the dataset, setting up placeholders for data storage.
       Parameters:
       - data_dir (str): Path to the dataset directory.
       - positions (list or numpy array, optional): Discrete position values. Default is [0].
       - step_size (float, optional): Conversion factor (cm to TOF in ps). Default is 0.066.
       """
       
       self.data_dir = data_dir
       self.positions = np.array(positions) if positions is not None else np.array([0])
       self.step_size = step_size
       
       # Compute theoretical TOF values based on positions
       self.Theoretical_TOF = self.step_size * self.positions
       
       # Placeholders
       self.TOF = None  
       self.TOF_dict = None  
       self.error_dict = None  
       self.Error = None  
       self.MAE = None  

    def load_data(self):
        """
        Loads dataset files corresponding to each predefined position value.
        
        Returns:
        - numpy array: Concatenated dataset containing all loaded data.
        
        Raises:
        - ValueError: If no valid data files are found.
        """
        data_dict = {}

        for i in range(np.min(self.positions), np.max(self.positions) + 1):   
            filename = f"Na22_norm_pos{i}_test.npz" if i >= 0 else f"Na22_norm_pos_min_{abs(i)}_test.npz"
            filepath = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):  # Load file only if it exists
                data_dict[i] = np.load(filepath, mmap_mode="r")["data"]
            else:
                print(f"Warning: {filepath} not found.")

        if data_dict:
            self.data = np.concatenate(list(data_dict.values()), axis=0)
            return self.data  # Return loaded data
        else:
            raise ValueError("No valid data files found!")

    def get_TOF_slices_train(self, TOF, size):
        """
        Assigns an external TOF array and partitions it into slices corresponding to different positions.

        Parameters:
        - TOF (numpy array): The externally provided TOF array.
        - size (int): The number of data points per position slice.
        
        Returns:
        - dict: Dictionary mapping positions to their respective TOF slices.
        
        Raises:
        - ValueError: If TOF array is not provided.
        """
        if TOF is None:
            raise ValueError("TOF array cannot be None.")

        self.TOF = TOF

        self.TOF_dict = {
            i: TOF[:, (i + np.max(self.positions)) * size : (i + np.max(self.positions) + 1) * size]
            for i in range(np.min(self.positions), np.max(self.positions) + 1)
        }
        return self.TOF_dict

    def get_TOF_slices_eval(self, TOF, size):
        """
        Assigns an external TOF array and partitions it into slices corresponding to different positions.
        Used during evaluation.

        Parameters:
        - TOF (numpy array): The externally provided TOF array.
        - size (int): The number of data points per position slice.
        
        Returns:
        - dict: Dictionary mapping positions to their respective TOF slices.
        
        Raises:
        - ValueError: If TOF array is not provided.
        """
        if TOF is None:
            raise ValueError("TOF array cannot be None.")

        self.TOF = TOF

        self.TOF_dict = {
            i: TOF[(i + np.max(self.positions)) * size : (i + np.max(self.positions) + 1) * size]
            for i in range(np.min(self.positions), np.max(self.positions) + 1)
        }
        return self.TOF_dict

    def compute_error(self, centroid):
        """
        Computes the TOF error for each position by comparing TOF slices with theoretical TOF values.

        Parameters:
        - centroid (numpy array): External centroid array used to compute the error.
        
        Returns:
        - dict: Dictionary mapping positions to error values.
        
        Raises:
        - ValueError: If TOF_dict is not initialized.
        - ValueError: If centroid array is not provided.
        """
        if self.TOF_dict is None:
            raise ValueError("TOF_dict is not set. Call get_TOF_slices_train() or get_TOF_slices_eval() first.")
        if centroid is None:
            raise ValueError("Centroid array cannot be None.")

        self.error_dict = {
            i: abs(self.TOF_dict[i] - centroid - self.Theoretical_TOF[i + np.max(self.positions)])
            for i in range(np.min(self.positions), np.max(self.positions) + 1)
        }

        return self.error_dict

