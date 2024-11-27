import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# ------------------------------ Na22 -------------------------------------
# -------------------------------------------------------------------------

#positions = np.array([-200, 0, 200])
#
## KAN
#FWHM_KAN = np.array([222.1, 220.8, 219.8, 221.4, 221.6]) #ps
#err_FWHM_KAN = np.array([1.5, 1.1, 1.2, 1.3, 1.4])
#centroid_KAN = np.array([-401.0, -200.8 , -0.0, 199.8 , 399.9]) #ps
#err_centroid_KAN = np.array([1.4, 1.0, 1.1, 1.2, 1.3])
#parameter_count =  310
#MAE = 0.07834 #ns
#Commit = 'October 17, 2024 at 4:46 PM'
#
#
## MLP
#FWHM_MLP = np.array([222.2, 220.8, 218.6, 219.5, 221.3])
#err_FWHM_MLP = np.array([1.5, 1.1, 0.9, 1.0, 1.2])
#centroid_MLP = np.array([-401.1, -200.8, -0.4, 199.7, 399.7])
#err_centroid_MLP = np.array([1.4, 1.0, 0.8, 1.0, 1.1])
#parameter_count = 657
#MAE = 0.08082
#Commit =  'October 21, 2024 at 9:58 PM'
#
#
## Wave-MLP
#FWHM_WAVEMLP = np.array([240.4, 236.6, 240.6])
#err_FWHM_WAVEMLP = np.array([1.1, 0.4, 0.9])
#centroid_WAVEMLP = np.array([-193.6, -0.0, 185.3])
#err_centroid_WAVEMLP = np.array([1.0, 0.4, 0.9])
#parameter_count = 1009
#MAE = 0.08277
#Commit = 'November 27, 2024 at 3:34 PM'
#
#
## Convolutional
#FWHM_Conv = np.array([231, 235, 232])
#err_FWHM_Conv = np.array([1.0, 1.0, 1.0])
#centroid_Conv = np.array([-189, 0.0, 187])
#err_centroid_Conv = np.array([1.0, 1.0, 1.0])
#parameter_count = 17793
#Mean_CTR_CNN =  232.578
#Std_CTR_CNN =  1.839
#Mean_bias_CNN =  8.191
#Std_bias_CNN =  0.719
#Mean_MAE_CNN =  80.196
#Std_MAE_CNN =  0.015
#
#Commit = 'November 27, 2024 at 2:58 PM'
#model = 'November 26, 2024 at 4:22 PM'
#
#
## CFD
#FWHM_CFD = np.array([237.8, 235.4, 235.8])
#err_FWHM_CFD = np.array([0.5, 0.8, 0.6])
#centroid_CFD = np.array([-194.7, -0.4, 189.5])
#err_centroid_CFD = np.array([0.5, 0.8, 0.6])
#parameter_count = 0 
#Mean_CTR_CFD =  236.215
#Std_CTR_CFD =  1.483
#Mean_bias_CFD =  5.495
#Std_bias_CFD =  0.575
#Mean_MAE_CFD =  82.518
#Std_MAE_CFD =  0.021
#Commit = 'November 22, 2024 at 5:16 PM'
#
#
#
#
#
#print('KAN:')
#print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (Mean_CTR_KAN, Std_CTR_KAN, Mean_bias_KAN, Mean_bias_KAN))
#print('MLP')
#print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (Mean_CTR_MLP, Std_CTR_MLP, Mean_bias_MLP, Mean_bias_MLP))
#print('WAVEMLP')
#print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (Mean_CTR_MLPWAVE, Std_CTR_MLPWAVE, Mean_bias_MLPWAVE, Mean_bias_MLPWAVE))
#print('Convolutional')
#print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (Mean_CTR_CNN, Std_CTR_CNN, Mean_bias_CNN, Mean_bias_CNN))
#print('CFD')
#print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (Mean_CTR_CFD, Std_CTR_CFD, Mean_bias_CFD, Mean_bias_CFD))
#
#
#fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
#
## Adjusting markersize, capsize, and color for better visibility and contrast
#ax.errorbar(positions+20, abs(centroid_KAN - positions), yerr = err_centroid_KAN, label = 'KAN', 
#            marker='o', markersize=10, linestyle='none', capsize=5, color='blue', markerfacecolor='blue', markeredgewidth=2)
#ax.errorbar(positions-20, abs(centroid_MLP - positions), yerr = err_centroid_MLP, label = 'MLP', 
#            marker='s', markersize=10, linestyle='none', capsize=5, color='red', markerfacecolor='red', markeredgewidth=2)
#ax.errorbar(positions+30, abs(centroid_WAVEMLP - positions), yerr = err_centroid_WAVEMLP, label = 'WAVEMLP', 
#            marker='^', markersize=10, linestyle='none', capsize=5, color='green', markerfacecolor='green', markeredgewidth=2)
#ax.errorbar(positions-30, abs(centroid_Conv - positions), yerr = err_centroid_Conv, label = 'CNN', 
#            marker='D', markersize=10, linestyle='none', capsize=5, color='purple', markerfacecolor='purple', markeredgewidth=2)
#ax.errorbar(positions, abs(centroid_CFD - positions), yerr = err_centroid_CFD, label = 'CFD', 
#            marker='D', markersize=10, linestyle='none', capsize=5, color='orange', markerfacecolor='orange', markeredgewidth=2)
#
## Labels and Grid
#ax.set_xlabel('Time difference [ps]', fontsize = 20)
#ax.set_ylabel('Bias [ps]', fontsize = 20)
#ax.tick_params(axis = 'both', which = 'major', labelsize = 16)
#
## Legend with larger font and more contrast
#ax.legend(fontsize = 20)
#
## Save and Show the plot
#plt.savefig('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/figures/bias.png', dpi = 300)
#plt.show()

# -------------------------------------------------------------------------
# ----------------------- Feature selection study -------------------------
# -------------------------------------------------------------------------

positions = np.array([-200, 0, 200])
commit = 'September 22, 2024 at 2:15 PM'

 
#----------------------------------- [NM, 3, 1, 1] -------------------------------------------

# 1 momento
FWHM_13 = np.array([253.8, 241.6, 263.6])  # ps
FWHM_err_13 = np.array([0.9, 0.9, 0.9])  # ps
centroid_13 = np.array([-191.2, -0.0, 192.9])  # ps
centroid_err_13 = np.array([0.8, 0.8, 0.8])  # ps
MAE_13 = 0.0866602  # ps


mean_FWHM_13 = np.mean(FWHM_13)
mean_FWHM_err_13 = np.mean(FWHM_err_13)
mean_bias_13 = np.mean(abs(centroid_13 - positions)) 


# 2 momentos
FWHM_23 = np.array([254.8, 242.2, 261.7])  # ps
FWHM_err_23 = np.array([0.9, 0.7, 1.0])  # ps
centroid_23 = np.array([-190.6, 0.3, 191.9])  # ps
centroid_err_23 = np.array([0.8, 0.6, 0.9])  # ps
MAE_23 = 0.0867586  # ps

mean_FWHM_23 = np.mean(FWHM_23)
mean_FWHM_err_23 = np.mean(FWHM_err_23)
mean_bias_23 = np.mean(abs(centroid_23 - positions))

# 3 momentos
FWHM_33 = np.array([248.4, 242.7, 255.6])  # ps
FWHM_err_33 = np.array([0.7, 0.7, 0.9])  # ps
centroid_33 = np.array([-190.4, 0.2, 193.5])  # ps
centroid_err_33 = np.array([0.7, 0.7, 0.8])  # ps
MAE_33 = 0.0851683  # ps

mean_FWHM_33 = np.mean(FWHM_33)
mean_FWHM_err_33 = np.mean(FWHM_err_33)
mean_bias_33 = np.mean(abs(centroid_33 - positions))


# 4 momentos
FWHM_43 = np.array([254.4, 246.1, 253.0])  # ps
FWHM_err_43 = np.array([1.1, 0.7, 1.2])  # ps
centroid_43 = np.array([-194.4, -0.2, 184.7])  # ps
centroid_err_43 = np.array([1.0, 0.7, 1.1])  # ps
MAE_43 = 0.0875471  # ps


mean_FWHM_43 = np.mean(FWHM_43)
mean_FWHM_err_43 = np.mean(FWHM_err_43)
mean_bias_43 = np.mean(abs(centroid_43 - positions))   


# 5 momentos
FWHM_53 = np.array([252.9, 243.3, 250.7])  # ps
FWHM_err_53 = np.array([1.2, 0.5, 0.8])  # ps
centroid_53 = np.array([-193.6, -0.5, 185.0])  # ps
centroid_err_53 = np.array([1.1, 0.5, 0.7])  # ps
MAE_53 = 0.0869640  # ps


mean_FWHM_53 = np.mean(FWHM_53)
mean_FWHM_err_53 = np.mean(FWHM_err_53)
mean_bias_53 = np.mean(abs(centroid_53 - positions))


# 6 momentos
FWHM_63 = np.array([243.2, 240.9, 239.6])  # ps
FWHM_err_63 = np.array([1.1, 0.8, 0.8])  # ps
centroid_63 = np.array([-195.1, -0.9, 184.0])  # ps
centroid_err_63 = np.array([1.1, 0.7, 0.7])  # ps
MAE_63 = 0.0828567  # ps

mean_FWHM_63 = np.mean(FWHM_63)    
mean_FWHM_err_63 = np.mean(FWHM_err_63)
mean_bias_63 = np.mean(abs(centroid_63 - positions))

# 7 momentos
FWHM_73 = np.array([239.7, 237.0, 236.0])  # ps
FWHM_err_73 = np.array([0.7, 0.6, 0.7])  # ps
centroid_73 = np.array([-192.4, -0.5, 188.5])  # ps
centroid_err_73 = np.array([0.6, 0.6, 0.7])  # ps
MAE_73 = 0.0815512  # ps

mean_FWHM_73 = np.mean(FWHM_73)
mean_FWHM_err_73 = np.mean(FWHM_err_73)
mean_bias_73 = np.mean(abs(centroid_73 - positions))

# 8 momentos
FWHM_83 = np.array([240.9, 238.1, 236.6])  # ps
FWHM_err_83 = np.array([0.8, 0.6, 0.8])  # ps
centroid_83 = np.array([-192.3, 0.3, 188.3])  # ps
centroid_err_83 = np.array([0.7, 0.6, 0.7])  # ps
MAE_83 = 0.0818824  # ps

mean_FWHM_83 = np.mean(FWHM_83)
mean_FWHM_err_83 = np.mean(FWHM_err_83)
mean_bias_83 = np.mean(abs(centroid_83 - positions))


# 9 momentos
FWHM_93 = np.array([239.9, 237.8, 235.1])  # ps
FWHM_err_93 = np.array([0.8, 0.5, 1.0])  # ps
centroid_93 = np.array([-192.9, -0.2, 187.8])  # ps
centroid_err_93 = np.array([0.7, 0.5, 0.9])  # ps
MAE_93 = 0.0816582  # ps

mean_FWHM_93 = np.mean(FWHM_93)
mean_FWHM_err_93 = np.mean(FWHM_err_93)
mean_bias_93 = np.mean(abs(centroid_93 - positions))

# 10 momentos
FWHM_103 = np.array([239.6, 236.9, 235.3])  # ps
FWHM_err_103 = np.array([0.8, 0.7, 0.8])  # ps
centroid_103 = np.array([-191.7, 1.4, 189.8])  # ps
centroid_err_103 = np.array([0.7, 0.6, 0.7])  # ps
MAE_103 = 0.0816503  # ps

mean_FWHM_103 = np.mean(FWHM_103)
mean_FWHM_err_103 = np.mean(FWHM_err_103)
mean_bias_103 = np.mean(abs(centroid_103 - positions))

# 11 momentos
FWHM_113 = np.array([238.2, 236.7, 233.5])  # ps
FWHM_err_113 = np.array([0.8, 0.6, 0.8])  # ps
centroid_113 = np.array([-192.0, 0.4, 188.5])  # ps
centroid_err_113 = np.array([0.7, 0.6, 0.7])  # ps
MAE_113 = 0.0810324  # ps

mean_FWHM_113 = np.mean(FWHM_113)
mean_FWHM_err_113 = np.mean(FWHM_err_113)
mean_bias_113 = np.mean(abs(centroid_113 - positions))

# 12 momentos
FWHM_123 = np.array([238.6, 237.3, 234.4])  # ps
FWHM_err_123 = np.array([1.0, 0.5, 0.7])  # ps
centroid_123 = np.array([-193.3, -0.6, 186.4])  # ps
centroid_err_123 = np.array([0.9, 0.4, 0.6])  # ps
MAE_123 = 0.0831598  # ps

mean_FWHM_123 = np.mean(FWHM_123)
mean_FWHM_err_123 = np.mean(FWHM_err_123)
mean_bias_123 = np.mean(abs(centroid_123 - positions))


parameter_count = np.array([70, 100, 130, 160, 190, 210, 240, 270, 300, 330, 360, 390])
MAE_3_nodos = np.array([MAE_13, MAE_23, MAE_33, MAE_43, MAE_53, MAE_63, MAE_73, MAE_83, MAE_93, MAE_103, MAE_113, MAE_123])
bias_3_nodos = np.array([mean_bias_13, mean_bias_23, mean_bias_33, mean_bias_43, mean_bias_53, mean_bias_63, mean_bias_73, mean_bias_83, mean_bias_93, mean_bias_103, mean_bias_113, mean_bias_123])
FWHM_3_nodos = np.array([mean_FWHM_13, mean_FWHM_23, mean_FWHM_33, mean_FWHM_43, mean_FWHM_53, mean_FWHM_63, mean_FWHM_73, mean_FWHM_83, mean_FWHM_93, mean_FWHM_103, mean_FWHM_113, mean_FWHM_123])

#----------------------------------- [NM, 5, 1, 1] -------------------------------------------

# 1 momento
FWHM_15 = np.array([253.9, 241.5, 260.7])  # ps
FWHM_err_15 = np.array([0.7, 0.8, 1.1])  # ps
centroid_15 = np.array([-191.6, -0.3, 190.3])  # ps
centroid_err_15 = np.array([0.7, 0.8, 1.0])  # ps
MAE_15 = 0.0889940  # ps

mean_FWHM_15 = np.mean(FWHM_15)
mean_FWHM_err_15 = np.mean(FWHM_err_15)
mean_bias_15 = np.mean(abs(centroid_15 - positions))


# 2 momentos
FWHM_25 = np.array([250.9, 244.2, 251.3])  # ps
FWHM_err_25 = np.array([1.0, 0.7, 1.2])  # ps
centroid_25 = np.array([-190.3, 0.3, 191.8])  # ps
centroid_err_25 = np.array([0.9, 0.6, 1.1])  # ps
MAE_25 = 0.0853328  # ps

mean_FWHM_25 = np.mean(FWHM_25)
mean_FWHM_err_25 = np.mean(FWHM_err_25)
mean_bias_25 = np.mean(abs(centroid_25 - positions))

# 3 momentos
FWHM_35 = np.array([249.6, 244.4, 247.2])  # ps
FWHM_err_35 = np.array([0.9, 0.9, 0.9])  # ps
centroid_35 = np.array([-192.0, 0.3, 188.9])  # ps
centroid_err_35 = np.array([0.8, 0.8, 0.8])  # ps
MAE_35 = 0.0845277  # ps

mean_FWHM_35 = np.mean(FWHM_35)
mean_FWHM_err_35 = np.mean(FWHM_err_35)
mean_bias_35 = np.mean(abs(centroid_35 - positions))


# 4 momentos
FWHM_45 = np.array([246.0, 242.5, 244.2])  # ps
FWHM_err_45 = np.array([1.0, 1.1, 0.9])  # ps
centroid_45 = np.array([-192.2, -0.5, 188.4])  # ps
centroid_err_45 = np.array([1.0, 1.0, 0.8])  # ps
MAE_45 = 0.0856754  # ps

mean_FWHM_45 = np.mean(FWHM_45)
mean_FWHM_err_45 = np.mean(FWHM_err_45)
mean_bias_45 = np.mean(abs(centroid_45 - positions))

# 5 momentos
FWHM_55 = np.array([242.2, 239.5, 239.2])  # ps
FWHM_err_55 = np.array([1.0, 0.9, 1.0])  # ps
centroid_55 = np.array([-193.2, -1.1, 187.8])  # ps
centroid_err_55 = np.array([0.9, 0.8, 0.9])  # ps
MAE_55 = 0.0824554  # ps


mean_FWHM_55 = np.mean(FWHM_55)
mean_FWHM_err_55 = np.mean(FWHM_err_55)
mean_bias_55 = np.mean(abs(centroid_55 - positions))


# 6 momentos
FWHM_65 = np.array([243.3, 240.6, 238.7])  # ps
FWHM_err_65 = np.array([1.0, 0.6, 1.1])  # ps
centroid_65 = np.array([-193.9, -1.2, 186.3])  # ps
centroid_err_65 = np.array([0.9, 0.6, 1.0])  # ps
MAE_65 = 0.0846612  # ps


mean_FWHM_65 = np.mean(FWHM_65)
mean_FWHM_err_65 = np.mean(FWHM_err_65)
mean_bias_65 = np.mean(abs(centroid_65 - positions))


# 7 momentos
FWHM_75 = np.array([240.3, 240.9, 235.2])  # ps
FWHM_err_75 = np.array([0.7, 0.6, 0.6])  # ps
centroid_75 = np.array([-193.5, -0.7, 186.7])  # ps
centroid_err_75 = np.array([0.6, 0.6, 0.6])  # ps
MAE_75 = 0.0846164  # ps

mean_FWHM_75 = np.mean(FWHM_75)
mean_FWHM_err_75 = np.mean(FWHM_err_75)
mean_bias_75 = np.mean(abs(centroid_75 - positions))

# 8 momentos
FWHM_85 = np.array([239.9, 240.0, 235.9])  # ps
FWHM_err_85 = np.array([0.7, 0.5, 0.9])  # ps
centroid_85 = np.array([-194.0, -1.1, 186.7])  # ps
centroid_err_85 = np.array([0.6, 0.5, 0.8])  # ps
MAE_85 = 0.0834926  # ps


mean_FWHM_85 = np.mean(FWHM_85)
mean_FWHM_err_85 = np.mean(FWHM_err_85)
mean_bias_85 = np.mean(abs(centroid_85 - positions))

# 9 momentos
FWHM_95 = np.array([239.5, 237.9, 235.1])  # ps
FWHM_err_95 = np.array([0.8, 0.8, 0.9])  # ps
centroid_95 = np.array([-191.7, -0.1, 189.3])  # ps
centroid_err_95 = np.array([0.8, 0.8, 0.8])  # ps
MAE_95 = 0.0830432  # ps


mean_FWHM_95 = np.mean(FWHM_95)
mean_FWHM_err_95 = np.mean(FWHM_err_95)
mean_bias_95 = np.mean(abs(centroid_95 - positions))

# 10 momentos
FWHM_105 = np.array([238.9, 235.9, 234.4])  # ps
FWHM_err_105 = np.array([0.7, 0.6, 0.8])  # ps
centroid_105 = np.array([-192.9, -0.5, 187.7])  # ps
centroid_err_105 = np.array([0.7, 0.6, 0.7])  # ps
MAE_105 = 0.0824306  # ps

mean_FWHM_105 = np.mean(FWHM_105)
mean_FWHM_err_105 = np.mean(FWHM_err_105)
mean_bias_105 = np.mean(abs(centroid_105 - positions))

# 11 momentos
FWHM_115 = np.array([238.6, 236.8, 233.7])  # ps
FWHM_err_115 = np.array([0.9, 0.6, 0.8])  # ps
centroid_115 = np.array([-192.7, -0.3, 187.3])  # ps
centroid_err_115 = np.array([0.8, 0.5, 0.8])  # ps
MAE_115 = 0.0813144  # ps

mean_FWHM_115 = np.mean(FWHM_115)
mean_FWHM_err_115 = np.mean(FWHM_err_115)
mean_bias_115 = np.mean(abs(centroid_115 - positions))

# 12 momentos
FWHM_125 = np.array([237.8, 235.5, 233.8])  # ps
FWHM_err_125 = np.array([0.8, 0.6, 0.8])  # ps
centroid_125 = np.array([-193.4, -1.1, 187.4])  # ps
centroid_err_125 = np.array([0.8, 0.6, 0.8])  # ps
MAE_125 = 0.0822196  # ps

mean_FWHM_125 = np.mean(FWHM_125)
mean_FWHM_err_125 = np.mean(FWHM_err_125)
mean_bias_125 = np.mean(abs(centroid_125 - positions))


parameter_count = np.array([110, 160, 210, 260, 310, 360, 410, 460, 510, 560, 610, 660])
MAE_5_nodos = np.array([MAE_15, MAE_25, MAE_35, MAE_45, MAE_55, MAE_65, MAE_75, MAE_85, MAE_95, MAE_105, MAE_115, MAE_125])
bias_5_nodos = np.array([mean_bias_15, mean_bias_25, mean_bias_35, mean_bias_45, mean_bias_55, mean_bias_65, mean_bias_75, mean_bias_85, mean_bias_95, mean_bias_105, mean_bias_115, mean_bias_125])
FWHM_5_nodos = np.array([mean_FWHM_15, mean_FWHM_25, mean_FWHM_35, mean_FWHM_45, mean_FWHM_55, mean_FWHM_65, mean_FWHM_75, mean_FWHM_85, mean_FWHM_95, mean_FWHM_105, mean_FWHM_115, mean_FWHM_125])


#----------------------------------- [NM, 7, 1, 1] -------------------------------------------


# 1 momento
FWHM_17 = np.array([253.9, 241.7, 261.2])  # ps
FWHM_err_17 = np.array([0.9, 0.9, 0.9])  # ps
centroid_17 = np.array([-190.4, 0.9, 191.8])  # ps
centroid_err_17 = np.array([0.8, 0.9, 0.9])  # ps
MAE_17 = 0.0864545  # ps

mean_FWHM_17 = np.mean(FWHM_17)
mean_FWHM_err_17 = np.mean(FWHM_err_17)
mean_bias_17 = np.mean(abs(centroid_17- positions))

# 2 momentos
FWHM_27 = np.array([252.6, 245.1, 251.9])  # ps
FWHM_err_27 = np.array([0.9, 0.7, 1.0])  # ps
centroid_27 = np.array([-192.1, 0.4, 189.4])  # ps
centroid_err_27 = np.array([0.8, 0.7, 0.9])  # ps
MAE_27 = 0.0855195  # ps

mean_FWHM_27 = np.mean(FWHM_27)
mean_FWHM_err_27 = np.mean(FWHM_err_27)
mean_bias_27 = np.mean(abs(centroid_27 - positions))

# 3 momentos
FWHM_37 = np.array([248.8, 245.2, 246.0])  # ps
FWHM_err_37 = np.array([0.9, 1.0, 1.0])  # ps
centroid_37 = np.array([-192.1, -0.0, 188.4])  # ps
centroid_err_37 = np.array([0.8, 0.9, 0.9])  # ps
MAE_37 = 0.0871516  # ps


mean_FWHM_37 = np.mean(FWHM_37)
mean_FWHM_err_37 = np.mean(FWHM_err_37)
mean_bias_37 = np.mean(abs(centroid_37 - positions))

# 4 momentos
FWHM_47 = np.array([248.0, 244.1, 242.4])  # ps
FWHM_err_47 = np.array([0.8, 0.9, 1.0])  # ps
centroid_47 = np.array([-193.7, -0.1, 186.6])  # ps
centroid_err_47 = np.array([0.7, 0.8, 0.9])  # ps
MAE_47 = 0.0856749  # ps


mean_FWHM_47 = np.mean(FWHM_47)
mean_FWHM_err_47 = np.mean(FWHM_err_47)
mean_bias_47 = np.mean(abs(centroid_47 - positions))

# 5 momentos
FWHM_57 = np.array([243.3, 243.6, 239.5])  # ps
FWHM_err_57 = np.array([1.0, 0.6, 0.8])  # ps
centroid_57 = np.array([-191.8, 0.3, 188.2])  # ps
centroid_err_57 = np.array([1.0, 0.5, 0.7])  # ps
MAE_57 = 0.0828617  # ps

mean_FWHM_57 = np.mean(FWHM_57)
mean_FWHM_err_57 = np.mean(FWHM_err_57)
mean_bias_57 = np.mean(abs(centroid_57 - positions))

# 6 momentos
FWHM_67 = np.array([241.1, 241.0, 236.7])  # ps
FWHM_err_67 = np.array([0.8, 0.7, 1.2])  # ps
centroid_67 = np.array([-190.4, 1.2, 190.1])  # ps
centroid_err_67 = np.array([0.7, 0.6, 1.1])  # ps
MAE_67 = 0.0821874  # ps

mean_FWHM_67 = np.mean(FWHM_67)
mean_FWHM_err_67 = np.mean(FWHM_err_67)
mean_bias_67 = np.mean(abs(centroid_67 - positions))

# 7 momentos
FWHM_77 = np.array([239.9, 239.6, 234.2])  # ps
FWHM_err_77 = np.array([0.8, 0.7, 0.8])  # ps
centroid_77 = np.array([-193.0, -0.9, 187.3])  # ps
centroid_err_77 = np.array([0.8, 0.6, 0.8])  # ps
MAE_77 = 0.0817458  # ps

mean_FWHM_77 = np.mean(FWHM_77)
mean_FWHM_err_77 = np.mean(FWHM_err_77)
mean_bias_77 = np.mean(abs(centroid_77 - positions))

# 8 momentos 
FWHM_87 = np.array([240.7, 240.2, 236.5])  # ps
FWHM_err_87 = np.array([0.9, 0.6, 1.0])  # ps
centroid_87 = np.array([-192.8, -0.1, 186.8])  # ps
centroid_err_87 = np.array([0.8, 0.5, 1.0])  # ps
MAE_87 = 0.0822118  # ps

mean_FWHM_87 = np.mean(FWHM_87)
mean_FWHM_err_87 = np.mean(FWHM_err_87)
mean_bias_87 = np.mean(abs(centroid_87 - positions))

# 9 momentos 
FWHM_97 = np.array([239.5, 237.5, 234.6])  # ps
FWHM_err_97 = np.array([0.9, 0.6, 0.8])  # ps
centroid_97 = np.array([-192.0, 0.1, 188.3])  # ps
centroid_err_97 = np.array([0.8, 0.6, 0.7])  # ps
MAE_97 = 0.0813349  # ps

mean_FWHM_97 = np.mean(FWHM_97)
mean_FWHM_err_97 = np.mean(FWHM_err_97)
mean_bias_97 = np.mean(abs(centroid_97 - positions))

# 10 momentos 
FWHM_107 = np.array([238.5, 236.6, 234.1])  # ps
FWHM_err_107 = np.array([0.7, 0.6, 0.9])  # ps
centroid_107 = np.array([-193.2, -0.3, 187.9])  # ps
centroid_err_107 = np.array([0.7, 0.5, 0.8])  # ps
MAE_107 = 0.0812061  # ps

mean_FWHM_107 = np.mean(FWHM_107)
mean_FWHM_err_107 = np.mean(FWHM_err_107)
mean_bias_107 = np.mean(abs(centroid_107 - positions))

# 11 momentos 
FWHM_117 = np.array([239.4, 238.5, 234.3])  # ps
FWHM_err_117 = np.array([0.9, 0.7, 0.9])  # ps
centroid_117 = np.array([-192.0, 0.9, 187.7])  # ps
centroid_err_117 = np.array([0.8, 0.6, 0.9])  # ps
MAE_117 = 0.0815052  # ps

mean_FWHM_117 = np.mean(FWHM_117)
mean_FWHM_err_117 = np.mean(FWHM_err_117)
mean_bias_117 = np.mean(abs(centroid_117 - positions))

# 12 momentos 
FWHM_127 = np.array([239.2, 237.2, 234.4])  # ps
FWHM_err_127 = np.array([0.8, 0.7, 0.9])  # ps
centroid_127 = np.array([-192.9, -0.0, 187.8])  # ps
centroid_err_127 = np.array([0.8, 0.6, 0.8])  # ps
MAE_127 = 0.0832877  # ps


mean_FWHM_127 = int(np.mean(FWHM_127))
mean_FWHM_err_127 = int(np.mean(FWHM_err_127))
mean_bias_127 = np.mean(abs(centroid_127 - positions))

parameter_count = np.array([150, 220, 290, 360, 430, 500, 570, 640, 710, 780, 850])
MAE_7_nodos = np.array([MAE_17, MAE_27, MAE_37, MAE_47, MAE_57, MAE_67, MAE_77, MAE_87, MAE_97, MAE_107, MAE_117, MAE_127])
bias_7_nodos = np.array([mean_bias_17, mean_bias_27, mean_bias_37, mean_bias_47, mean_bias_57, mean_bias_67, mean_bias_77, mean_bias_87, mean_bias_97, mean_bias_107, mean_bias_117, mean_bias_127])
FWHM_7_nodos = np.array([mean_FWHM_17, mean_FWHM_27, mean_FWHM_37, mean_FWHM_47, mean_FWHM_57, mean_FWHM_67, mean_FWHM_77, mean_FWHM_87, mean_FWHM_97, mean_FWHM_107, mean_FWHM_117, mean_FWHM_127])

#----------------------------------- [NM, 9, 1, 1] -------------------------------------------

# 1 momento 
FWHM_19 = np.array([253.7, 241.7, 261.0])  # ps
FWHM_err_19 = np.array([0.9, 0.8, 0.9])  # ps
centroid_19 = np.array([-191.8, -0.4, 190.9])  # ps
centroid_err_19 = np.array([0.8, 0.7, 0.9])  # ps
MAE_19 = 0.0886695  # ps

mean_FWHM_19 = np.mean(FWHM_19)
mean_FWHM_err_19 = np.mean(FWHM_err_19)
mean_bias_19 = np.mean(abs(centroid_19- positions))


# 2 momentos
FWHM_29 = np.array([250.3, 244.4, 250.4])  # ps
FWHM_err_29 = np.array([1.0, 0.8, 0.9])  # ps
centroid_29 = np.array([-189.6, -0.5, 193.4])  # ps
centroid_err_29 = np.array([0.9, 0.7, 0.8])  # ps
MAE_29 = 0.0850805  # ps

mean_FWHM_29 = np.mean(FWHM_29)
mean_FWHM_err_29 = np.mean(FWHM_err_29)
mean_bias_29 = np.mean(abs(centroid_29- positions))


#3 momentos
FWHM_39 = np.array([248.1, 244.1, 245.2])  # ps
FWHM_err_39 = np.array([1.0, 1.0, 1.2])  # ps
centroid_39 = np.array([-191.7, 0.2, 189.4])  # ps
centroid_err_39 = np.array([0.9, 0.9, 1.1])  # ps
MAE_39 = 0.0841654  # ps

mean_FWHM_39 = np.mean(FWHM_39)
mean_FWHM_err_39 = np.mean(FWHM_err_39)
mean_bias_39 = np.mean(abs(centroid_39- positions))


# 4 momentos
FWHM_49 = np.array([244.7, 243.1, 241.0])  # ps
FWHM_err_49 = np.array([0.8, 1.0, 0.9])  # ps
centroid_49 = np.array([-192.7, -0.2, 187.5])  # ps
centroid_err_49 = np.array([0.8, 0.9, 0.8])  # ps
MAE_49 = 0.0846479  # ps

mean_FWHM_49 = np.mean(FWHM_49)
mean_FWHM_err_49 = np.mean(FWHM_err_49)
mean_bias_49 = np.mean(abs(centroid_49- positions))


# 5 momentos
FWHM_59 = np.array([242.6, 241.5, 240.2])  # ps
FWHM_err_59 = np.array([1.1, 0.7, 0.9])  # ps
centroid_59 = np.array([-193.0, -0.0, 186.9])  # ps
centroid_err_59 = np.array([1.0, 0.7, 0.8])  # ps
MAE_59 = 0.0847548  # ps

mean_FWHM_59 = np.mean(FWHM_59)
mean_FWHM_err_59 = np.mean(FWHM_err_59)
mean_bias_59 = np.mean(abs(centroid_59- positions))

# 6 momentos
FWHM_69 = np.array([242.3, 241.6, 237.3])  # ps
FWHM_err_69 = np.array([1.0, 0.8, 0.9])  # ps
centroid_69 = np.array([-191.0, 1.7, 189.2])  # ps
centroid_err_69 = np.array([0.9, 0.7, 0.9])  # ps
MAE_69 = 0.0825276  # ps

mean_FWHM_69 = np.mean(FWHM_69)
mean_FWHM_err_69 = np.mean(FWHM_err_69)
mean_bias_69 = np.mean(abs(centroid_69- positions))


# 7 momentos
FWHM_79 = np.array([240.7, 240.7, 235.6])  # ps
FWHM_err_79 = np.array([0.9, 0.7, 1.0])  # ps
centroid_79 = np.array([-191.0, 1.7, 188.9])  # ps
centroid_err_79 = np.array([0.8, 0.7, 0.9])  # ps
MAE_79 = 0.0837008  # ps

mean_FWHM_79 = np.mean(FWHM_79)
mean_FWHM_err_79 = np.mean(FWHM_err_79)
mean_bias_79 = np.mean(abs(centroid_79- positions))

#8 momentos
FWHM_89 = np.array([240.0, 238.2, 235.4])  # ps
FWHM_err_89 = np.array([0.7, 0.5, 0.8])  # ps
centroid_89 = np.array([-193.4, 0.2, 187.1])  # ps
centroid_err_89 = np.array([0.6, 0.5, 0.8])  # ps
MAE_89 = 0.0836545  # ps

mean_FWHM_89 = np.mean(FWHM_89)
mean_FWHM_err_89 = np.mean(FWHM_err_89)
mean_bias_89 = np.mean(abs(centroid_89- positions))

#9 momentos
FWHM_99 = np.array([240.2, 238.7, 235.7])  # ps
FWHM_err_99 = np.array([0.9, 0.6, 0.9])  # ps
centroid_99 = np.array([-193.9, -1.2, 186.5])  # ps
centroid_err_99 = np.array([0.8, 0.5, 0.9])  # ps
MAE_99 = 0.0833453  # ps

mean_FWHM_99 = np.mean(FWHM_99)
mean_FWHM_err_99 = np.mean(FWHM_err_99)
mean_bias_99 = np.mean(abs(centroid_99- positions))


#9 momentos
FWHM_99 = np.array([240.2, 238.7, 235.7])  # ps
FWHM_err_99 = np.array([0.9, 0.6, 0.9])  # ps
centroid_99 = np.array([-193.9, -1.2, 186.5])  # ps
centroid_err_99 = np.array([0.8, 0.5, 0.9])  # ps
MAE_99 = 0.0833453  # ps

mean_FWHM_99 = np.mean(FWHM_99)
mean_FWHM_err_99 = np.mean(FWHM_err_99)
mean_bias_99 = np.mean(abs(centroid_99- positions))


#10 momentos
FWHM_109 = np.array([239.0, 236.6, 234.2])  # ps
FWHM_err_109 = np.array([0.7, 0.7, 0.8])  # ps
centroid_109 = np.array([-192.1, -0.1, 188.8])  # ps
centroid_err_109 = np.array([0.6, 0.6, 0.8])  # ps
MAE_109 = 0.0812291  # ps

mean_FWHM_109 = np.mean(FWHM_109)
mean_FWHM_err_109 = np.mean(FWHM_err_109)
mean_bias_109 = np.mean(abs(centroid_109- positions))

#11 momentos
FWHM_119 = np.array([238.8, 237.9, 234.8])  # ps
FWHM_err_119 = np.array([0.9, 0.6, 0.8])  # ps
centroid_119 = np.array([-193.8, -0.8, 186.6])  # ps
centroid_err_119 = np.array([0.8, 0.6, 0.8])  # ps
MAE_119 = 0.0829675  # ps

mean_FWHM_119 = np.mean(FWHM_119)
mean_FWHM_err_119 = np.mean(FWHM_err_119)
mean_bias_119 = np.mean(abs(centroid_119- positions))

#12 momentos
FWHM_129 = np.array([238.6, 235.6, 233.9])  # ps
FWHM_err_129 = np.array([0.9, 0.7, 0.8])  # ps
centroid_129 = np.array([-192.1, 0.3, 188.7])  # ps
centroid_err_129 = np.array([0.8, 0.7, 0.7])  # ps
MAE_129 = 0.0828251  # ps

mean_FWHM_129 = np.mean(FWHM_129)
mean_FWHM_err_129 = np.mean(FWHM_err_129)
mean_bias_129 = np.mean(abs(centroid_129- positions))

MAE_9_nodos = np.array([MAE_19, MAE_29, MAE_39, MAE_49, MAE_59, MAE_69, MAE_79, MAE_89, MAE_99, MAE_109, MAE_119, MAE_129])
bias_9_nodos = np.array([mean_bias_19, mean_bias_29, mean_bias_39, mean_bias_49, mean_bias_59, mean_bias_69, mean_bias_79, mean_bias_89, mean_bias_99, mean_bias_109, mean_bias_119, mean_bias_129])
FWHM_9_nodos = np.array([mean_FWHM_19, mean_FWHM_29, mean_FWHM_39, mean_FWHM_49, mean_FWHM_59, mean_FWHM_69, mean_FWHM_79, mean_FWHM_89, mean_FWHM_99, mean_FWHM_109, mean_FWHM_119, mean_FWHM_129])

#------------------------------------------------------------------ Color matrix ------------------------------------------------
MAE = np.stack((MAE_3_nodos, MAE_5_nodos, MAE_7_nodos, MAE_9_nodos), axis = -1)
BIAS = np.stack((bias_3_nodos, bias_5_nodos, bias_7_nodos, bias_9_nodos), axis = -1)
FWHM = np.stack((FWHM_3_nodos, FWHM_5_nodos, FWHM_7_nodos, FWHM_9_nodos), axis = -1)


plt.subplot(131)
plt.imshow(MAE*1000, cmap = 'inferno_r', interpolation = 'nearest', aspect = 'auto')
plt.colorbar()
plt.title('MAE (ps)', fontsize = 10)
plt.xlabel('Node number', fontsize = 8)
plt.ylabel('Number of features', fontsize = 8)
plt.xticks([0, 1, 2, 3], [3, 5, 7, 9], fontsize = 8)  # X ticks at positions 3, 5, 7, 9
plt.yticks(np.arange(0, len(MAE[:,0]), 1).tolist(), np.arange(1, len(MAE[:,0]) + 1, 1).tolist(), fontsize = 8)  # Y ticks from 1 to 8


plt.subplot(132)
plt.imshow(BIAS, cmap = 'inferno_r', interpolation = 'nearest', aspect = 'auto')
plt.colorbar()
plt.title('BIAS (ps)', fontsize = 10)
plt.xlabel('Node number', fontsize = 8)
plt.ylabel('Number of features', fontsize = 8)
plt.xticks([0, 1, 2, 3], [3, 5, 7, 9], fontsize = 8)
plt.yticks(np.arange(0, len(BIAS[:,0]), 1).tolist(), np.arange(1, len(BIAS[:,0]) + 1, 1).tolist(), fontsize = 8)

plt.subplot(133)
plt.imshow(FWHM, cmap = 'inferno_r', interpolation = 'nearest', aspect = 'auto')
plt.colorbar()
plt.title('FWHM (ps)', fontsize = 10)
plt.xlabel('Node number', fontsize = 8)
plt.ylabel('Number of features', fontsize = 8)
plt.xticks([0, 1, 2, 3], [3, 5, 7, 9], fontsize = 8)
plt.yticks(np.arange(0, len(FWHM[:,0]), 1).tolist(), np.arange(1, len(FWHM[:,0]) + 1, 1).tolist(), fontsize = 8)

plt.tight_layout()

# Save the figure with high DPI for publication
plt.savefig('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/figures/three_metrics.png', dpi = 300, bbox_inches = 'tight')
plt.tight_layout()
plt.show()

# -------------------------------------------------------------------------
# ------------------ MAE singles vs MAE coincidences ----------------------
# -------------------------------------------------------------------------

mean_err_val_dec0 = np.load('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec0_Na22.npz', allow_pickle = True)['data']
mean_err_val_dec1 = np.load('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/mean_err_val_dec1_Na22.npz', allow_pickle = True)['data']
MAE = np.load('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/predictions/MAE_Na22.npz', allow_pickle = True)['data']

plt.plot(mean_err_val_dec0*1000, MAE*1000, 'b.', label = 'MAE singles detector 0')
plt.plot(mean_err_val_dec1*1000, MAE*1000, 'r.', label = 'MAE singles detector 1')
plt.xlabel('MAE singles [ps]')
plt.ylabel('MAE coincidences [ps]')
plt.legend()
plt.savefig('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/figures/MAE_singles_vs_MAE_coincidences_Na22.png')
plt.show()


index = np.where(MAE*1000 < 78)[0]
square_sum = np.sqrt(mean_err_val_dec0.astype(np.float32)**2 + mean_err_val_dec1.astype(np.float32)**2)
plt.plot(square_sum[5:]*1000, MAE[5:]*1000, 'r.', label = 'Regular points')
plt.plot(square_sum[index]*1000, MAE[index]*1000, 'b.', label = 'Time compressed points')
plt.xlabel('MAE singles squared sum [ps]')
plt.ylabel('MAE coincidences [ps]')
plt.legend()
plt.savefig('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/figures/MAE_singles_squared_sums_vs_MAE_coincidences_Na22.png')
plt.show()


# -------------------------------------------------------------------------
# -------------------------- INFERENCE TIME -------------------------------
# -------------------------------------------------------------------------

moments_calculation = 14.35
moments_calculation_std =  0.10

KAN = 0.022
KAN_std = 0.001

MLP = 0.008
MLP_std = 0.004

MLP_Wave = 0.250
MLP_Wave_std = 0.010

CNN = 0.258
CNN_std = 0.004


# -------------------------------------------------------------------------
# -------------------------- Singles MAE ----------------------------------
# -------------------------------------------------------------------------

# KAN

dec0 = 0.00277
dec1 =  0.00207

# MLP-Features

dec0 = 0.02197
dec1 =  0.00205

# Convolutional

dec0 = 0.35387
dec1 =  0.00389