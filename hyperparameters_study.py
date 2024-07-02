import numpy as np

# Semilla de la red
seed = np.array([42, 123, 321, 456, 0])  
FWHM_results = np.array([150, 149.3, 147.3, 150.3, 145])
bias_results = np.array([11.6, 9.6, 12, 11, 13.6])

#Semilla del dataset
seed = np.array([42, 0, 123, 321, 456])  
FWHM_results = np.array([144.3, 151, 147.6, 148.6, 148.3, 151.6])
bias_results = np.array([12.6, 8.6, 11, 10.3, 12, 12])


#pulso escalon
Num_sims = np.array([1000, 2000, 3000, 4000, 5000]) # For 4795 reals
FWHM_results = np.array([146.6, 145.6, 149.6, 144.6, 149])
bias_results = np.array([8, 13.3, 14.6, 11.6, 12.6])


#Para sims doble exponencial
Num_sims = np.array([0, 1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000]) # For 4795 reals
epochs = np.array([300, 300, 200, 150, 125, 125, 125, 125, 125])
FWHM_results = np.array([160, 155.6, 152.3, 151.3, 144.3, 150, 146.7, 143.6, 145])
bias_results = np.array([5, 9, 8.3, 8.7, 12.6, 12.6, 12, 14, 13.3])


#max delay
max_delays = np.array([10, 15, 20, 25, 30, 35, 40, 45, 50])
FWHM_results = np.array([142.6, 145.7, 144.3, 150, 144.6, 148.3, 147, 146.3, 146.3])
bias_results = np.array([16.3, 12.3, 13, 11.6, 13, 10.6, 11.6, 12.3, 12.6])