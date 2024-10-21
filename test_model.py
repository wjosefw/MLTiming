import os 
import numpy as np
import matplotlib.pyplot as plt
import torch
from efficient_kan.src.efficient_kan import KAN

from functions import (create_position, momentos, normalize_given_params, 
                       calculate_gaussian_center_sigma, plot_gaussian, get_gaussian_params,
                       continuous_delay)


#Load data
data_dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'
data = np.load(os.path.join(data_dir, 'Na22_test_val.npz'))['data']


# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

moments_order = 5       # Max order of moments used
t_shift = 1             # Time steps to move for the new positions
nbins = 71              # Num bins for all histograms  
normalization_method = 'standardization'
start= 50
stop = 74
time_step = 0.2         # Signal time step

# -------------------------------------------------------------------------
#----------------------- ALIGN PULSES -------------------------------------
# -------------------------------------------------------------------------

align_time = 0.6 # In ns
new_data = continuous_delay(data, time_step = time_step, delay_time = align_time, channel_to_fix = 0, channel_to_move = 1)

# -------------------------------------------------------------------------
#----------------------- CROP WAVEFORM ------------------------------------
# -------------------------------------------------------------------------

test_data = new_data[:,start:stop,:] 
print('Número de casos de test: ', test_data.shape[0])

# -------------------------------------------------------------------------
# ------------------------ PREPROCESS DATA --------------------------------
# -------------------------------------------------------------------------

TEST_00 = test_data 
TEST_02 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = t_shift)
TEST_20 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = t_shift)
TEST_04 = create_position(TEST_00, channel_to_move = 1, channel_to_fix = 0, t_shift = int(2*t_shift))
TEST_40 = create_position(TEST_00, channel_to_move = 0, channel_to_fix = 1, t_shift = int(2*t_shift))
TEST = np.concatenate((TEST_02, TEST_00, TEST_20, TEST_04, TEST_40), axis = 0)

# Calculate moments 
MOMENTS_TEST = momentos(TEST, order = moments_order)

# Normalize moments
params_dec0 = (np.array([0.38123475, 0.38367176, 0.36448884, 0.34427001, 0.32574118]), np.array([0.40531052, 0.3211757 , 0.28093212, 0.25361035, 0.23289862]))
params_dec1 = (np.array([0.34459084, 0.36008492, 0.34417236, 0.32550077, 0.30801989]), np.array([0.43774342, 0.33152184, 0.28260705, 0.25142029, 0.22891419]))

MOMENTS_TEST_norm_dec0 = normalize_given_params(MOMENTS_TEST, params_dec0, channel = 0, method = normalization_method)
MOMENTS_TEST_norm_dec1 = normalize_given_params(MOMENTS_TEST, params_dec1, channel = 1, method = normalization_method)
MOMENTS_TEST = np.stack((MOMENTS_TEST_norm_dec0, MOMENTS_TEST_norm_dec1), axis = -1)


# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------

dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/KAN_models'
model_dec0_dir = os.path.join(dir, 'model_dec0_495')
model_dec1_dir = os.path.join(dir, 'model_dec1_495')

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
TOF = test_dec0 - test_dec1

TOF_V02 = TOF[:TEST_00.shape[0]] 
TOF_V00 = TOF[TEST_00.shape[0] : 2*TEST_00.shape[0]] 
TOF_V20 = TOF[2*TEST_00.shape[0] :3*TEST_00.shape[0]] 
TOF_V04 = TOF[3*TEST_00.shape[0] :4*TEST_00.shape[0]] 
TOF_V40 = TOF[4*TEST_00.shape[0]:] 
    

# Calulate Test error
centroid_V00, sigmaN_V00 = calculate_gaussian_center_sigma(TOF_V00[np.newaxis,:], np.zeros((TOF_V00[np.newaxis,:].shape[0])), nbins = nbins) 

error_V02 = abs((TOF_V02 - centroid_V00 + time_step*t_shift))
error_V00 = abs((TOF_V00 - centroid_V00))
error_V20 = abs((TOF_V20 - centroid_V00 - time_step*t_shift))
error_V04 = abs((TOF_V04 - centroid_V00 + 2*time_step*t_shift))
error_V40 = abs((TOF_V40 - centroid_V00 - 2*time_step*t_shift))


#Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00, error_V04, error_V40))   
MAE = np.mean(Error)
print('MAE: ', MAE)


# Plot
plt.hist(test_dec0, bins = nbins, range = [-1, 4], alpha = 0.5, label = 'Detector 0');
plt.hist(test_dec1, bins = nbins, range = [-1, 4], alpha = 0.5, label = 'Detector 1');
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



# -------------------------------------------------------------------------
#--------------------------- ENERGY DEPENDENCE ----------------------------
# -------------------------------------------------------------------------

dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/'
energy_dec0 = np.load(os.path.join(dir,'pulsos_Na22_energy_dec0_test_val.npz'), allow_pickle = True)['data']
energy_dec1 = np.load(os.path.join(dir,'pulsos_Na22_energy_dec1_test_val.npz'), allow_pickle = True)['data']


plt.plot(energy_dec0 - energy_dec1,  TOF_V00, 'b.', markersize = 1)
plt.xlabel('Moment 0 diff')
plt.ylabel('Time difference (ns)')
plt.show()


plt.plot(energy_dec0 - energy_dec1, error_V00, 'b.', markersize = 1)
plt.xlabel('Energy diff')
plt.ylabel('Error')
plt.show()


# -------------------------------------------------------------------------
#--------------------------- BOOTSTRAPING ---------------------------------
# -------------------------------------------------------------------------

resolution_list = []
for i in range(6):
    a = np.random.choice(np.arange(0, test_data.shape[0]), size = 500, replace = False)
    
    params_V04, errors_V04 = get_gaussian_params(TOF_V04[a], centroid_V00, range = 0.8, nbins = nbins-20)
    params_V02, errors_V02 = get_gaussian_params(TOF_V02[a], centroid_V00, range = 0.8, nbins = nbins-20)
    params_V00, errors_V00 = get_gaussian_params(TOF_V00[a], centroid_V00, range = 0.8, nbins = nbins-20)
    params_V20, errors_V20 = get_gaussian_params(TOF_V20[a], centroid_V00, range = 0.8, nbins = nbins-20)
    params_V40, errors_V40 = get_gaussian_params(TOF_V40[a], centroid_V00, range = 0.8, nbins = nbins-20)

    resolution = np.mean((params_V40[3], params_V20[3], params_V00[3], params_V02[3], params_V04[3]))
    resolution_list.append(resolution)

print('Mean CTR: ', np.mean(np.array(resolution_list))*1000)
print('Std CTR: ', np.std(np.array(resolution_list))*1000)


# -------------------------------------------------------------------------
#------------------------------ AVERAGE -----------------------------------
# -------------------------------------------------------------------------

MOMENTS_TEST = momentos(TEST, order = moments_order)
MOMENTS_TEST_norm_dec00 = normalize_given_params(MOMENTS_TEST, params_dec0, channel = 0, method = normalization_method)
MOMENTS_TEST_norm_dec01 = normalize_given_params(MOMENTS_TEST, params_dec1, channel = 0, method = normalization_method)


MOMENTS_TEST_norm_dec10 = normalize_given_params(MOMENTS_TEST, params_dec0, channel = 1, method = normalization_method)
MOMENTS_TEST_norm_dec11 = normalize_given_params(MOMENTS_TEST, params_dec1, channel = 1, method = normalization_method)



test_dec0_model0 = np.squeeze(model_dec0(torch.tensor(MOMENTS_TEST_norm_dec00).float()).detach().numpy())
test_dec0_model1 = np.squeeze(model_dec1(torch.tensor(MOMENTS_TEST_norm_dec01).float()).detach().numpy())

test_dec1_model0 = np.squeeze(model_dec0(torch.tensor(MOMENTS_TEST_norm_dec10).float()).detach().numpy())
test_dec1_model1 = np.squeeze(model_dec1(torch.tensor(MOMENTS_TEST_norm_dec11).float()).detach().numpy())

test_dec0 = (test_dec0_model0 + test_dec0_model1) / 2
test_dec1 = (test_dec1_model0 + test_dec1_model1) / 2


# Calculate TOF
TOF = test_dec0 - test_dec1

TOF_V02 = TOF[:TEST_00.shape[0]] 
TOF_V00 = TOF[TEST_00.shape[0] : 2*TEST_00.shape[0]] 
TOF_V20 = TOF[2*TEST_00.shape[0] :3*TEST_00.shape[0]] 
TOF_V04 = TOF[3*TEST_00.shape[0] :4*TEST_00.shape[0]] 
TOF_V40 = TOF[4*TEST_00.shape[0]:] 
    

# Calulate Test error
centroid_V00, sigmaN_V00 = calculate_gaussian_center_sigma(TOF_V00[np.newaxis,:], np.zeros((TOF_V00[np.newaxis,:].shape[0])), nbins = nbins) 

error_V02 = abs((TOF_V02 - centroid_V00 + time_step*t_shift))
error_V00 = abs((TOF_V00 - centroid_V00))
error_V20 = abs((TOF_V20 - centroid_V00 - time_step*t_shift))
error_V04 = abs((TOF_V04 - centroid_V00 + 2*time_step*t_shift))
error_V40 = abs((TOF_V40 - centroid_V00 - 2*time_step*t_shift))


#Get MAE
Error = np.concatenate((error_V02, error_V20, error_V00, error_V04, error_V40))   
MAE = np.mean(Error)
print('MAE: ', MAE)

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

