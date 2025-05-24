import os
import numpy as np
from torch.utils.data import Dataset

class Datos_LAB_GFN(Dataset):
    """
    Dataset for Time-of-Flight (TOF) measurements acquired by the Nuclear Physics Group at UCM (Spain).
    Encapsulates loading and processing logic for experimental data acquired at different detector positions.
    """

    def __init__(self, data_dir):
        """
        Initialize the dataset with configuration and placeholders.

        Parameters:
        - data_dir (str): Path to the directory containing the dataset files.
        """
        self.time_step = 0.2  # Sampling time step in ns
        self.positions = np.array([-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5])  # Discrete position indices
        self.step_size = 0.066  # Conversion factor: 1 cm → 0.066 ns TOF difference
        self.Theoretical_TOF = self.step_size * self.positions  # Theoretical TOF shifts in ns based on positions
        self.position_range = np.arange(np.min(self.positions), np.max(self.positions) + 1)
        self.data_dir = data_dir  # Directory with .npz data files

        # Internal data containers
        self.TOF = None
        self.TOF_dict = None
        self.error_dict = None
        
    def load_test_data(self):
        """
        Load test data files for all predefined positions.

        Returns:
        - np.ndarray: Concatenated data array from all available test files.

        Raises:
        - ValueError: If no test files could be loaded from the specified directory.
        """
        data_dict = {}

        for i in range(np.min(self.positions), np.max(self.positions) + 1):
            filename = f"Na22_norm_pos{i}_test.npz" if i >= 0 else f"Na22_norm_pos_min_{abs(i)}_test.npz"
            filepath = os.path.join(self.data_dir, filename)

            if os.path.exists(filepath):
                data_dict[i] = np.load(filepath, mmap_mode="r")["data"]
            else:
                print(f"Warning: {filepath} not found.")

        if data_dict:
            self.data = np.concatenate(list(data_dict.values()), axis=0)
            return self.data
        else:
            raise ValueError("No valid data files found!")

    def load_params(self):
        """
        Return the time step and detector positions used in this dataset.

        Returns:
        - tuple: (time_step, positions)
        """
        return self.time_step, self.positions, self.Theoretical_TOF

    def load_train_data(self):
        """
        Load the training dataset from position 0.

        Returns:
        - np.ndarray: Training data array.

        Note:
        - If the file is not found, a warning is printed and None is returned.
        """
        filename = 'Na22_norm_pos0_train.npz'
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            self.train_data = np.load(filepath, mmap_mode="r")["data"]
            return self.train_data
        else:
            print(f"Warning: {filepath} not found.")

    def load_val_data(self):
        """
        Load the validation dataset from position 0.

        Returns:
        - np.ndarray: Validation data array.

        Note:
        - If the file is not found, a warning is printed and None is returned.
        """
        filename = 'Na22_norm_pos0_val.npz'
        filepath = os.path.join(self.data_dir, filename)
        if os.path.exists(filepath):
            self.val_data = np.load(filepath, mmap_mode="r")["data"]
            return self.val_data
        else:
            print(f"Warning: {filepath} not found.")

    def get_TOF_slices_train(self, TOF):
        """
        Slice the training TOF array by position based on equal-sized segments.

        Parameters:
        - TOF (np.ndarray): Input TOF array.

        Returns:
        - dict: Mapping of position index to corresponding TOF slices.

        Raises:
        - ValueError: If TOF is None.
        """
        if TOF is None:
            raise ValueError("TOF array cannot be None.")

        self.TOF = TOF
        size = int(TOF.shape[1] / self.Theoretical_TOF.shape[0])
        self.TOF_dict = {
            position: TOF[:, (position + np.max(self.positions)) * size : (position + np.max(self.positions) + 1) * size]
            for position in self.position_range
        }
        return self.TOF_dict

    def get_TOF_slices_eval(self, TOF):
        """
        Slice the evaluation TOF array into segments by position.

        Parameters:
        - TOF (np.ndarray): Input TOF array.

        Returns:
        - dict: Mapping of position index to corresponding TOF slices.

        Raises:
        - ValueError: If TOF is None.
        """
        if TOF is None:
            raise ValueError("TOF array cannot be None.")

        self.TOF = TOF
        size = int(TOF.shape[0] / self.Theoretical_TOF.shape[0])
        self.TOF_dict = {
            position: TOF[(position + np.max(self.positions)) * size : (position + np.max(self.positions) + 1) * size]
            for position in self.position_range
        }
        return self.TOF_dict

    def compute_error(self, centroid):
        """
        Compute absolute TOF errors between observed and expected centroids.

        Parameters:
        - centroid (np.ndarray): Estimated centroid value to compare against.

        Returns:
        - dict: Mapping of position index to corresponding TOF error arrays.

        Raises:
        - ValueError: If TOF_dict or centroid is None.
        """
        if self.TOF_dict is None:
            raise ValueError("TOF_dict is not set. Call get_TOF_slices_train() or get_TOF_slices_eval() first.")
        if centroid is None:
            raise ValueError("Centroid array cannot be None.")

        self.error_dict = {
            position: abs(self.TOF_dict[position] - centroid - self.Theoretical_TOF[position + np.max(self.positions)])
            for position in self.position_range
        }

        return self.error_dict
