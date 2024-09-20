import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

from functions import (interpolate_pulses, create_position, momentos, normalize_given_params, 
                       calculate_gaussian_center_sigma, plot_gaussian, get_gaussian_params)


#Load data
data_dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'
data = np.load(os.path.join(data_dir, 'Na22_test_val.npz'))['data']


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

moments_order = 5       # Max order of moments used
t_shift = 8             # Time steps to move for the new positions
nbins = 91              # Num bins for all histograms  
normalization_method = 'standardization'
EXTRASAMPLING = 8
start_idx = 50
stop_idx = 74
start = start_idx*EXTRASAMPLING 
stop = stop_idx*EXTRASAMPLING 

# -------------------------------------------------------------------------
#----------------------- INTERPOLATE PULSES -------------------------------
# -------------------------------------------------------------------------

new_data, new_time_step =  interpolate_pulses(data, EXTRASAMPLING = EXTRASAMPLING, time_step = 0.2)

# Align the pulses 
align_steps = 20

new_data[:,:,1] = np.roll(new_data[:,:,1], align_steps)
new_data[:,:align_steps,1] = np.random.normal(scale = 1e-6, size = align_steps)


# -------------------------------------------------------------------------
#----------------------- CROP WAVEFORM ------------------------------------
# -------------------------------------------------------------------------

test_data = new_data[:,start:stop,:] 
print('Número de casos de test: ', test_data.shape[0])


# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

TEST_00 = test_data 
TEST_02 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = t_shift, NOISE = False)
TEST_20 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = t_shift, NOISE = False)
TEST_04 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = int(2*t_shift), NOISE = False)
TEST_40 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = int(2*t_shift), NOISE = False)
TEST = np.concatenate((TEST_02, TEST_00, TEST_20, TEST_04, TEST_40), axis = 0)

# Calculate moments 
MOMENTS_TEST = momentos(TEST, order = moments_order)

# Normalize moments
params_dec0 = (np.array([4.33487848, 4.1571882 , 3.84664459, 3.5533875 , 3.29384201]), np.array([3.43156913, 2.69506511, 2.3152794 , 2.05005583, 1.84584942]))
params_dec1 = (np.array([4.86934725, 4.69257392, 4.32931311, 3.98689157, 3.68629918]), np.array([4.03246105, 3.04384752, 2.55958324, 2.23933913, 2.00179458]))


MOMENTS_TEST_norm_dec0 = normalize_given_params(MOMENTS_TEST, params_dec0, channel = 0, method = normalization_method)
MOMENTS_TEST_norm_dec1 = normalize_given_params(MOMENTS_TEST, params_dec1, channel = 1, method = normalization_method)
MOMENTS_TEST = np.stack((MOMENTS_TEST_norm_dec0, MOMENTS_TEST_norm_dec1), axis = -1)


# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------

dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/'
model_dec0_dir = os.path.join(dir, 'model_dec0_300')
model_dec1_dir = os.path.join(dir, 'model_dec1_300')

architecture = [moments_order, 5, 1, 1]    

model_dec0 = KAN(architecture)
model_dec1 = KAN(architecture)

model_dec0.load_state_dict(torch.load(model_dec0_dir))
model_dec1.load_state_dict(torch.load(model_dec1_dir))

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

test_dec0 = np.squeeze(model_dec0(torch.tensor(MOMENTS_TEST[:,:,0]).float()).detach().numpy())
test_dec1 = np.squeeze(model_dec1(torch.tensor(MOMENTS_TEST[:,:,1]).float()).detach().numpy())



# Calculate TOF
TOF_V02 = test_dec0[:TEST_00.shape[0]] - test_dec1[:TEST_00.shape[0]]
TOF_V00 = test_dec0[TEST_00.shape[0] : 2*TEST_00.shape[0]] - test_dec1[TEST_00.shape[0] : 2*TEST_00.shape[0]]
TOF_V20 = test_dec0[2*TEST_00.shape[0] :3*TEST_00.shape[0]] - test_dec1[2*TEST_00.shape[0] :3*TEST_00.shape[0]]
TOF_V04 = test_dec0[3*TEST_00.shape[0] :4*TEST_00.shape[0]] - test_dec1[3*TEST_00.shape[0] :4*TEST_00.shape[0]]
TOF_V40 = test_dec0[4*TEST_00.shape[0]:] - test_dec1[4*TEST_00.shape[0]:]
    

# Calulate Test error
centroid_V00, sigmaN_V00 = calculate_gaussian_center_sigma(TOF_V00[np.newaxis,:], np.zeros((TOF_V00[np.newaxis,:].shape[0])), nbins = nbins) 

error_V02 = abs((TOF_V02 - centroid_V00[:] + 0.2))
error_V00 = abs((TOF_V00 - centroid_V00[:]))
error_V20 = abs((TOF_V20 - centroid_V00[:] - 0.2))
error_V04 = abs((TOF_V04 - centroid_V00[:] + 0.4))
error_V40 = abs((TOF_V40 - centroid_V00[:] - 0.4))

#Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00, error_V04, error_V40))   
MAE = np.mean(Error)
print('MAE: ', MAE)


# Plot
plt.hist(test_dec0, bins = nbins, range = [-1+start_idx*0.2, 1+stop_idx*0.2], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1, bins = nbins, range = [-1+start_idx*0.2, 1+stop_idx*0.2], alpha = 0.5, label = 'Detector 1');
plt.title('Single detector prediction histograms')
plt.xlabel('time (ns)')
plt.ylabel('Counts')
plt.legend()
plt.show()


# Histogram and gaussian fit 
plot_gaussian(TOF_V04, centroid_V00, range = 0.8, label = '-0.4 ns offset', nbins = nbins)
plot_gaussian(TOF_V02, centroid_V00, range = 0.8, label = '-0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V00, centroid_V00, range = 0.8, label = ' 0.0 ns offset', nbins = nbins)
plot_gaussian(TOF_V20, centroid_V00, range = 0.8, label = ' 0.2 ns offset', nbins = nbins)
plot_gaussian(TOF_V40, centroid_V00, range = 0.8, label = ' 0.4 ns offset', nbins = nbins)


params_V04, errors_V04 = get_gaussian_params(TOF_V04, centroid_V00, range = 0.8, nbins = nbins)
params_V02, errors_V02 = get_gaussian_params(TOF_V02, centroid_V00, range = 0.8, nbins = nbins)
params_V00, errors_V00 = get_gaussian_params(TOF_V00, centroid_V00, range = 0.8, nbins = nbins)
params_V20, errors_V20 = get_gaussian_params(TOF_V20, centroid_V00, range = 0.8, nbins = nbins)
params_V40, errors_V40 = get_gaussian_params(TOF_V40, centroid_V00, range = 0.8, nbins = nbins)


print("V40: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V40[2], errors_V40[2], params_V40[3], errors_V40[3]))
print("V20: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V20[2], errors_V20[2], params_V20[3], errors_V20[3]))
print("V00: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V00[2], errors_V00[2], params_V00[3], errors_V00[3]))
print("V02: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V02[2], errors_V02[2], params_V02[3], errors_V02[3]))
print("V04: CENTROID(ns) = %.4f +/- %.5f  FWHM(ns) = %.4f +/- %.5f" % (params_V04[2], errors_V04[2], params_V04[3], errors_V04[3]))

print('')
plt.legend()
plt.xlabel('$\Delta t$ (ns)', fontsize = 14)
plt.ylabel('Counts', fontsize = 14)
plt.show()



