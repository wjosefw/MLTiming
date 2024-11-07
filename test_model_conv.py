import os 
import numpy as np
import matplotlib.pyplot as plt
import torch


from functions import (create_position, calculate_gaussian_center, 
                       plot_gaussian, get_gaussian_params, continuous_delay)
from Models import ConvolutionalModel

#Load data
data_dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/Na22_filtered_data/'
data = np.load(os.path.join(data_dir, 'Na22_train.npz'))['data']
data = data[3000:,:,:]

# -------------------------------------------------------------------------
#----------------------- IMPORTANT DEFINITIONS ----------------------------
# -------------------------------------------------------------------------

t_shift = 1             # Time steps to move for the new positions
nbins = 71              # Num bins for all histograms  
start= 47
stop = 74
time_step = 0.2         # Signal time step
positions = np.array([0.4, 0.2, 0.0, -0.2, -0.4])


# -------------------------------------------------------------------------
#----------------------- ALIGN PULSES -------------------------------------
# -------------------------------------------------------------------------

align_time = 0.5 # In ns
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


# -------------------------------------------------------------------------
#--------------------------- LOAD MODELS ----------------------------------
# -------------------------------------------------------------------------

model_dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/Convolutional'
model_dec0_dir = os.path.join(model_dir, 'Conv_model_dec0')
model_dec1_dir = os.path.join(model_dir, 'Conv_model_dec1')

model_dec0 = ConvolutionalModel(int(stop-start))
model_dec1 = ConvolutionalModel(int(stop-start))

model_dec0.load_state_dict(torch.load(model_dec0_dir))
model_dec1.load_state_dict(torch.load(model_dec1_dir))
model_dec0.eval()
model_dec1.eval()

# -------------------------------------------------------------------------
#--------------------------- GET RESULTS ----------------------------------
# -------------------------------------------------------------------------

test_dec0 = np.squeeze(model_dec0(torch.tensor(TEST[:,None,:,0]).float()).detach().numpy())
test_dec1 = np.squeeze(model_dec1(torch.tensor(TEST[:,None,:,1]).float()).detach().numpy())

# Calculate TOF
TOF = test_dec0 - test_dec1

TOF_V02 = TOF[:TEST_00.shape[0]] 
TOF_V00 = TOF[TEST_00.shape[0] : 2*TEST_00.shape[0]] 
TOF_V20 = TOF[2*TEST_00.shape[0] :3*TEST_00.shape[0]] 
TOF_V04 = TOF[3*TEST_00.shape[0] :4*TEST_00.shape[0]] 
TOF_V40 = TOF[4*TEST_00.shape[0]:] 
    

# Calulate Test error
centroid_V00 = calculate_gaussian_center(TOF_V00[np.newaxis,:], nbins = nbins, limits = 3) 

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

#dir = '/home/josea/DEEP_TIMING/DEEP_TIMING_VS/'
#energy_dec0 = np.load(os.path.join(dir,'pulsos_Na22_energy_dec0_test_val.npz'), allow_pickle = True)['data']
#energy_dec1 = np.load(os.path.join(dir,'pulsos_Na22_energy_dec1_test_val.npz'), allow_pickle = True)['data']
#
#Error = np.concatenate((error_V02, error_V20, error_V00, error_V04, error_V40))   
#MAE = np.mean(Error)
#print('MAE: ', MAE)
#
#plt.plot(energy_dec0 - energy_dec1,  TOF_V00, 'b.', markersize = 1)
#plt.xlabel('Moment 0 diff')
#plt.ylabel('Time difference (ns)')
#plt.show()
#
#
#plt.plot(energy_dec0 - energy_dec1, error_V00, 'b.', markersize = 1)
#plt.xlabel('Energy diff')
#plt.ylabel('Error')
#plt.show()


# -------------------------------------------------------------------------
#--------------------------- BOOTSTRAPING ---------------------------------
# -------------------------------------------------------------------------

resolution_list = []
bias_list = []
MAE_list = []
for i in range(1000):
    a = np.random.choice(np.arange(0, TOF_V00.shape[0]), size = TOF_V00.shape[0], replace = True)
    
    centroid_V00 = calculate_gaussian_center(TOF_V00[None, a], nbins = nbins, limits = 3) 
    params_V04, errors_V04 = get_gaussian_params(TOF_V04[a], centroid_V00, range = 0.8, nbins = nbins)
    params_V02, errors_V02 = get_gaussian_params(TOF_V02[a], centroid_V00, range = 0.8, nbins = nbins)
    params_V00, errors_V00 = get_gaussian_params(TOF_V00[a], centroid_V00, range = 0.8, nbins = nbins)
    params_V20, errors_V20 = get_gaussian_params(TOF_V20[a], centroid_V00, range = 0.8, nbins = nbins)
    params_V40, errors_V40 = get_gaussian_params(TOF_V40[a], centroid_V00, range = 0.8, nbins = nbins)
    
    resolution = np.mean((params_V40[3], params_V20[3], params_V00[3], params_V02[3], params_V04[3]))
    resolution_list.append(resolution)
    
    centroids = np.array([params_V40[2], params_V20[2], params_V00[2], params_V02[2], params_V04[2]])
    bias = np.mean(abs(centroids - positions))
    bias_list.append(bias)

    error_V02 = abs((TOF_V02 - centroid_V00[:, np.newaxis] + time_step*t_shift))
    error_V00 = abs((TOF_V00 - centroid_V00[:, np.newaxis]))
    error_V20 = abs((TOF_V20 - centroid_V00[:, np.newaxis] - time_step*t_shift))
    error_V04 = abs((TOF_V04 - centroid_V00[:, np.newaxis] + 2*time_step*t_shift))
    error_V40 = abs((TOF_V40 - centroid_V00[:, np.newaxis] - 2*time_step*t_shift))

    Error = np.concatenate((error_V02, error_V00, error_V20, error_V04, error_V40), axis = 1)   
    MAE_list.append(np.mean(Error)) 

print('Mean CTR: ', np.mean(np.array(resolution_list))*1000)
print('Std CTR: ', np.std(np.array(resolution_list))*1000)
print('Mean bias: ', np.mean(np.array(bias_list))*1000)
print('Std bias: ', np.std(np.array(bias_list))*1000)
print('Mean MAE: ', np.mean(np.array(MAE_list))*1000)
print('Std MAE: ', np.std(np.array(MAE_list))*1000)

# -------------------------------------------------------------------------
#-------------------------- INFERENCE TIME --------------------------------
# -------------------------------------------------------------------------

import time
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

time_test = np.tile(test_data[0,:,:] , (1000000, 1,1))
model_dec0 = model_dec0.to(device)
time_list_moments = []

time_list_inference = []
# Start timer inference
for i in range(100):
    start_time_inference= time.time()
    with torch.no_grad():
        assert not torch.is_grad_enabled()
        output_time_test = model_dec0(torch.tensor(time_test[:,None,:,0]).float().to(device))
    end_time_inference = time.time()
    elapsed_time_inference = end_time_inference - start_time_inference
    time_list_inference.append(elapsed_time_inference)
time_array_inference = np.array(time_list_inference)

print('Elapsed time inference:', np.mean(time_array_inference), np.std(time_array_inference))