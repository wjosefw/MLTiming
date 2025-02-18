import os
import numpy as np
from torch.utils.data import Dataset

class Datos_LAB_GFN(Dataset):
    """
    Data obtained at UCM (Spain) by the Group of Nuclear Physics.
    """
    
    positions = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]) 
    step_size = 0.066  # 1 cm corresponds to a TOF difference of 66.6 ps 
    Theoretical_TOF = step_size * positions

    def __init__(self, data_dir, train=True):
        """
        Initializes the dataset.

        Parameters:
        - data_dir (str): Path to dataset directory.
        - train (bool): Whether to use training mode.
        """
        self.data_dir = data_dir
        self.train = train
        self.data = None  # Placeholder for dataset
        self.TOF = None  # Placeholder for external TOF array
        self.TOF_dict = None  # Placeholder for computed TOF slices
        self.error_dict = None  # Placeholder for error per position
        self.Error = None  # Placeholder for concatenated error array
        self.MAE = None  # Placeholder for mean absolute error

    def load_data(self):
        """
        Loads dataset files based on position values.
        Returns:
        - data (numpy array): Concatenated dataset.
        """
        data_dict = {}

        for i in range(np.min(self.positions), np.max(self.positions) + 1):   
            filename = f"Na22_norm_pos{i}_test.npz" if i >= 0 else f"Na22_norm_pos_min_{abs(i)}_test.npz"
            filepath = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):  # Ensure the file exists before loading
                data_dict[i] = np.load(filepath, mmap_mode="r")["data"]
            else:
                print(f"Warning: {filepath} not found.")

        if data_dict:
            self.data = np.concatenate(list(data_dict.values()), axis = 0)
            return self.data  # Now the function also returns the data array
        else:
            raise ValueError("No valid data files found!")

    def get_TOF_slices_train(self, TOF, size):
        """
        Assigns an external TOF array and computes TOF_dict dynamically.

        Parameters:
        - TOF (numpy array): The externally calculated TOF array.
        """
        if TOF is None:
            raise ValueError("TOF array cannot be None.")

        self.TOF = TOF

        # Compute TOF slices dynamically
        self.TOF_dict = {
            i: TOF[:, (i + np.max(self.positions)) * size : (i + np.max(self.positions) + 1) * size]
            for i in range(np.min(self.positions), np.max(self.positions) + 1)
        }
        return self.TOF_dict

    def get_TOF_slices_eval(self, TOF, size):
        """
        Assigns an external TOF array and computes TOF_dict dynamically.

        Parameters:
        - TOF (numpy array): The externally calculated TOF array.
        """
        if TOF is None:
            raise ValueError("TOF array cannot be None.")

        self.TOF = TOF

        # Compute TOF slices dynamically
        self.TOF_dict = {
            i: TOF[(i + np.max(self.positions)) * size : (i + np.max(self.positions) + 1) * size]
            for i in range(np.min(self.positions), np.max(self.positions) + 1)
        }
        return self.TOF_dict

    def compute_error(self, centroid):
        """
        Computes the error per position based on TOF_dict and Theoretical_TOF.

        Parameters:
        - centroid (numpy array): External centroid array used for error computation.
        """
        if self.TOF_dict is None:
            raise ValueError("TOF_dict is not set. Call set_TOF() first.")
        if centroid is None:
            raise ValueError("Centroid array cannot be None.")

        # Compute errors for all positions
        self.error_dict = {
            i: abs(self.TOF_dict[i] - centroid[:, np.newaxis] - self.Theoretical_TOF[i + np.max(self.positions)])
            for i in range(np.min(self.positions), np.max(self.positions) + 1)
        }

        return self.error_dict
