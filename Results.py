import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------------
# ------------------------------ Na22 -------------------------------------
# -------------------------------------------------------------------------

positions = np.array([-400, -200, 0, 200, 400])

# KAN
FWHM_KAN = np.array([223.7, 225.1, 223.4, 223.9, 222.6]) #ps
err_FWHM_KAN = np.array([1.3, 1.4, 1.1, 1.2, 1.3])
centroid_KAN = np.array([-399.8, -199.8 , -0.0, 199.5 , 399.5]) #ps
err_centroid_KAN = np.array([1.2, 1.1, 1.1, 1.3, 1.2])
parameter_count =  310
MAE = 0.07974 #ns
Commit = 'September 22, 2024 at 5:21 PM'


# MLP
FWHM_MLP = np.array([222.5, 223.6, 222.2, 223.7, 220.0])
err_FWHM_MLP = np.array([1.2, 1.4, 1.0, 1.2, 1.3])
centroid_MLP = np.array([-399.0, -199.0, 0.0, 200.8, 400.7])
err_centroid_MLP = np.array([1.2, 1.1, 0.9, 1.3, 1.1])
parameter_count = 8769
MAE = 0.08441
Commit =  'September 22, 2024 at 5:21 PM'


# Wave-MLP
FWHM_WAVEMLP = np.array([232.0, 231.6, 230.6, 230.6, 230.9])
err_FWHM_WAVEMLP = np.array([1.9, 1.3, 1.6, 1.6, 1.3])
centroid_WAVEMLP = np.array([-398.3, -198.2, 0.2, 201.7, 401.8])
err_centroid_WAVEMLP = np.array([1.8, 1.2, 1.5, 1.5, 1.2])
parameter_count = 20737
MAE = 0.08540
Commit = 'September 22, 2024 at 10:17 PM'



# Convolutional
FWHM_Conv = np.array([218.1, 222.0, 218.6, 220.8, 219.0])
err_FWHM_Conv = np.array([1.2, 1.4, 1.3, 0.9, 1.2])
centroid_Conv = np.array([-409.0, -203.5, 0.0, 197.9, 400.1])
err_centroid_Conv = np.array([0.9, 0.8, 1.0, 1.3, 1.0])
parameter_count = 2065 
MAE = 0.08622
Commit = 'September 22, 2024 at 10:50 PM'
model = 'September 22, 2024 at 10:50 PM'



# CFD
FWHM_CFD = np.array([259.2, 259.7, 259.4, 259.1, 259.9])
err_FWHM_CFD = np.array([1.6, 1.7, 1.9, 1.7, 1.6])
centroid_CFD = np.array([-400.6, -200.2, -0.3, 199.6, 400.1])
err_centroid_CFD = np.array([1.5, 1.5, 1.7, 1.5, 1.4])
parameter_count = 0 
MAE = 0.09565
Commit = 'September 8, 2024 at 10:26 PM'


print('KAN:')
print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (np.mean(FWHM_KAN), np.std(FWHM_KAN), np.mean(abs(centroid_KAN-positions)), np.std(abs(centroid_KAN-positions))))
print('MLP')
print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (np.mean(FWHM_MLP), np.std(FWHM_MLP), np.mean(abs(centroid_MLP-positions)), np.std(abs(centroid_MLP-positions))))
print('WAVEMLP')
print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (np.mean(FWHM_WAVEMLP), np.std(FWHM_WAVEMLP), np.mean(abs(centroid_WAVEMLP-positions)), np.std(abs(centroid_WAVEMLP-positions))))
print('Convolutional')
print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (np.mean(FWHM_Conv), np.std(FWHM_Conv), np.mean(abs(centroid_Conv-positions)), np.std(abs(centroid_Conv-positions))))
print('CFD')
print('FWHM: %.3f +/- %.3f Centroid: %.3f +/- %.3f' % (np.mean(FWHM_CFD), np.std(FWHM_CFD), np.mean(abs(centroid_CFD-positions)), np.std(abs(centroid_CFD-positions))))


fig, ax = plt.subplots(figsize=(14, 8), dpi=300)

# Adjusting markersize, capsize, and color for better visibility and contrast
ax.errorbar(positions+20, abs(centroid_KAN - positions), yerr = err_centroid_KAN, label = 'KAN', 
            marker='o', markersize=10, linestyle='none', capsize=5, color='blue', markerfacecolor='blue', markeredgewidth=2)
ax.errorbar(positions-20, abs(centroid_MLP - positions), yerr = err_centroid_MLP, label = 'MLP', 
            marker='s', markersize=10, linestyle='none', capsize=5, color='red', markerfacecolor='red', markeredgewidth=2)
ax.errorbar(positions+30, abs(centroid_WAVEMLP - positions), yerr = err_centroid_WAVEMLP, label = 'WAVEMLP', 
            marker='^', markersize=10, linestyle='none', capsize=5, color='green', markerfacecolor='green', markeredgewidth=2)
ax.errorbar(positions-30, abs(centroid_Conv - positions), yerr = err_centroid_Conv, label = 'CNN', 
            marker='D', markersize=10, linestyle='none', capsize=5, color='purple', markerfacecolor='purple', markeredgewidth=2)
ax.errorbar(positions, abs(centroid_CFD - positions), yerr = err_centroid_CFD, label = 'CFD', 
            marker='D', markersize=10, linestyle='none', capsize=5, color='orange', markerfacecolor='orange', markeredgewidth=2)

# Labels and Grid
ax.set_xlabel('Time difference [ps]', fontsize = 20)
ax.set_ylabel('Bias [ps]', fontsize = 20)
ax.tick_params(axis = 'both', which = 'major', labelsize = 16)

# Legend with larger font and more contrast
ax.legend(fontsize = 20)

# Save and Show the plot
plt.savefig('/home/josea/DEEP_TIMING/DEEP_TIMING_VS/figures/bias.png', dpi = 300)
plt.show()

# -------------------------------------------------------------------------
# ----------------------- Feature selection study -------------------------
# -------------------------------------------------------------------------

positions = np.array([-400, -200, 0, 200, 400])
commit = 'September 22, 2024 at 2:15 PM'

 
#----------------------------------- [NM, 3, 1, 1] -------------------------------------------

# 1 momento
FWHM_13 = np.array([245.9, 249.0, 255.7, 247.5, 243.8])  # ps
FWHM_err_13 = np.array([2.1, 2.6, 2.3, 2.4, 3.1])  # ps
centroid_13 = np.array([-395.3, -197.1, -0.3, 197.8, 396.4])  # ps
centroid_err_13 = np.array([2.0, 2.4, 2.1, 2.2, 2.8])  # ps
MAE_13 = 0.0868175  # ps


mean_FWHM_13 = int(np.mean(FWHM_13))
mean_FWHM_err_13 = int(np.mean(FWHM_err_13))
mean_bias_13 = int(np.mean(abs(centroid_13 - positions)))  


# 2 momentos
FWHM_23 = np.array([235.4, 237.3, 237.6, 237.9, 235.6])  # ps
FWHM_err_23 = np.array([1.8, 2.8, 2.2, 2.6, 2.5])  # ps
centroid_23 = np.array([-398.7, -198.7, -0.0, 199.7, 399.9])  # ps
centroid_err_23 = np.array([1.7, 2.5, 2.1, 2.4, 2.3])  # ps
MAE_23 = 0.0819039  # ps

mean_FWHM_23 = int(np.mean(FWHM_23))
mean_FWHM_err_23 = int(np.mean(FWHM_err_23))
mean_bias_23 = int(np.mean(abs(centroid_23 - positions)))

# 3 momentos
FWHM_33 = np.array([229.5, 230.0, 230.9, 229.9, 229.3])  # ps
FWHM_err_33 = np.array([1.9, 2.6, 1.7, 1.3, 1.8])  # ps
centroid_33 = np.array([-398.9, -198.6, 0.2, 199.8, 399.8])  # ps
centroid_err_33 = np.array([1.7, 2.4, 1.6, 1.2, 1.7])  # ps
MAE_33 = 0.0794947  # ps

mean_FWHM_33 = int(np.mean(FWHM_33))
mean_FWHM_err_33 = int(np.mean(FWHM_err_33))
mean_bias_33 = int(np.mean(abs(centroid_33 - positions)))


# 4 momentos
FWHM_43 = np.array([226.9, 224.7, 225.4, 224.8, 225.5])  # ps
FWHM_err_43 = np.array([1.8, 1.9, 1.5, 1.6, 1.9])  # ps
centroid_43 = np.array([-399.5, -199.0, 0.7, 200.0, 399.8])  # ps
centroid_err_43 = np.array([1.7, 1.7, 1.4, 1.5, 1.8])  # ps
MAE_43 = 0.0783204  # ps


mean_FWHM_43 = int(np.mean(FWHM_43))
mean_FWHM_err_43 = int(np.mean(FWHM_err_43))
mean_bias_43 = int(np.mean(abs(centroid_43 - positions)))     


# 5 momentos
FWHM_53 = np.array([222.0, 222.3, 222.7, 223.3, 222.3])  # ps
FWHM_err_53 = np.array([2.0, 1.5, 1.9, 2.0, 2.1])  # ps
centroid_53 = np.array([-399.7, -199.8, 0.2, 199.1, 399.4])  # ps
centroid_err_53 = np.array([1.9, 1.4, 1.8, 1.9, 2.0])  # ps
MAE_53 = 0.0777420  # ps


mean_FWHM_53 = int(np.mean(FWHM_53))
mean_FWHM_err_53 = int(np.mean(FWHM_err_53))
mean_bias_53 = int(np.mean(abs(centroid_53 - positions)))


# 6 momentos
FWHM_63 = np.array([219.8, 222.8, 221.1, 221.3, 221.6])  # ps
FWHM_err_63 = np.array([1.8, 1.8, 2.1, 2.1, 2.1])  # ps
centroid_63 = np.array([-399.5, -199.4, 0.1, 199.5, 399.2])  # ps
centroid_err_63 = np.array([1.7, 1.7, 1.9, 2.0, 1.9])  # ps
MAE_63 = 0.0776399  # ps

mean_FWHM_63 = int(np.mean(FWHM_63))
mean_FWHM_err_63 = int(np.mean(FWHM_err_63))
mean_bias_63 = int(np.mean(abs(centroid_63 - positions)))


# 7 momentos
FWHM_73 = np.array([218.3, 221.3, 220.2, 220.2, 221.8])  # ps
FWHM_err_73 = np.array([1.8, 2.3, 2.0, 2.1, 2.2])  # ps
centroid_73 = np.array([-399.2, -199.7, -0.3, 199.7, 399.4])  # ps
centroid_err_73 = np.array([1.7, 2.1, 1.9, 1.9, 2.0])  # ps
MAE_73 = 0.0766493  # ps

mean_FWHM_73 = int(np.mean(FWHM_73))
mean_FWHM_err_73 = int(np.mean(FWHM_err_73))
mean_bias_73 = int(np.mean(abs(centroid_73 - positions)))

# 8 momentos
FWHM_83 = np.array([218.5, 220.4, 219.9, 221.1, 221.7])  # ps
FWHM_err_83 = np.array([1.7, 2.1, 2.0, 2.1, 2.4])  # ps
centroid_83 = np.array([-398.9, -199.4, 0.1, 200.0, 399.5])  # ps
centroid_err_83 = np.array([1.6, 2.0, 1.9, 2.0, 2.2])  # ps
MAE_83 = 0.0770747  # ps

mean_FWHM_83 = int(np.mean(FWHM_83))
mean_FWHM_err_83 = int(np.mean(FWHM_err_83))
mean_bias_83 = int(np.mean(abs(centroid_83 - positions)))


parameter_count = np.array([70, 100, 130, 160, 190, 210, 240, 270])
MAE_3_nodos = np.array([MAE_13, MAE_23, MAE_33, MAE_43, MAE_53, MAE_63, MAE_73, MAE_83])
bias_3_nodos = np.array([mean_bias_13, mean_bias_23, mean_bias_33, mean_bias_43, mean_bias_53, mean_bias_63, mean_bias_73, mean_bias_83])
FWHM_3_nodos = np.array([mean_FWHM_13, mean_FWHM_23, mean_FWHM_33, mean_FWHM_43, mean_FWHM_53, mean_FWHM_63, mean_FWHM_73, mean_FWHM_83])

#----------------------------------- [NM, 5, 1, 1] -------------------------------------------

# 1 momento
FWHM_15 = np.array([245.8, 248.8, 255.5, 247.2, 243.6])  # ps
FWHM_err_15 = np.array([2.1, 2.8, 2.4, 2.5, 3.0])  # ps
centroid_15 = np.array([-395.1, -196.8, 0.1, 198.1, 396.7])  # ps
centroid_err_15 = np.array([1.9, 2.6, 2.2, 2.3, 2.7])  # ps
MAE_15 = 0.0872293  # ps

mean_FWHM_15 = int(np.mean(FWHM_15))
mean_FWHM_err_15 = int(np.mean(FWHM_err_15))
mean_bias_15 = int(np.mean(abs(centroid_15 - positions)))


# 2 momentos
FWHM_25 = np.array([232.9, 235.7, 236.0, 236.0, 234.3])  # ps
FWHM_err_25 = np.array([1.8, 2.1, 2.1, 1.7, 1.6])  # ps
centroid_25 = np.array([-398.3, -198.6, 0.1, 199.4, 399.9])  # ps
centroid_err_25 = np.array([1.7, 1.9, 2.0, 1.6, 1.5])  # ps
MAE_25 = 0.0815110  # ps

mean_FWHM_25 = int(np.mean(FWHM_25))
mean_FWHM_err_25 = int(np.mean(FWHM_err_25))
mean_bias_25 = int(np.mean(abs(centroid_25 - positions)))

# 3 momentos
FWHM_35 = np.array([231.7, 230.8, 230.3, 228.8, 229.6])  # ps
FWHM_err_35 = np.array([2.0, 2.4, 1.7, 1.4, 1.5])  # ps
centroid_35 = np.array([-399.2, -198.8, 0.0, 199.3, 399.7])  # ps
centroid_err_35 = np.array([1.9, 2.2, 1.6, 1.3, 1.4])  # ps
MAE_35 = 0.0797603  # ps

mean_FWHM_35 = int(np.mean(FWHM_35))
mean_FWHM_err_35 = int(np.mean(FWHM_err_35))
mean_bias_35 = int(np.mean(abs(centroid_35 - positions)))


# 4 momentos
FWHM_45 = np.array([225.9, 224.5, 224.7, 224.6, 225.5])  # ps
FWHM_err_45 = np.array([1.9, 2.0, 1.8, 1.4, 2.1])  # ps
centroid_45 = np.array([-400.3, -199.6, -0.1, 199.8, 399.2])  # ps
centroid_err_45 = np.array([1.8, 1.9, 1.7, 1.3, 2.0])  # ps
MAE_45 = 0.0771519  # ps

mean_FWHM_45 = int(np.mean(FWHM_45))
mean_FWHM_err_45 = int(np.mean(FWHM_err_45))
mean_bias_45 = int(np.mean(abs(centroid_45 - positions)))

# 5 momentos
FWHM_55 = np.array([220.6, 222.7, 220.6, 222.2, 221.0])  # ps
FWHM_err_55 = np.array([2.1, 1.8, 1.6, 1.9, 1.9])  # ps
centroid_55 = np.array([-400.3, -200.3, -0.3, 199.1, 398.8])  # ps
centroid_err_55 = np.array([2.0, 1.7, 1.5, 1.7, 1.8])  # ps
MAE_55 = 0.0777056  # ps


mean_FWHM_55 = int(np.mean(FWHM_55))
mean_FWHM_err_55 = int(np.mean(FWHM_err_55))
mean_bias_55 = int(np.mean(abs(centroid_55 - positions)))


# 6 momentos
FWHM_65 = np.array([219.1, 222.5, 220.0, 220.0, 221.3])  # ps
FWHM_err_65 = np.array([1.8, 2.2, 1.9, 1.8, 2.2])  # ps
centroid_65 = np.array([-399.9, -200.0, -0.0, 199.5, 399.4])  # ps
centroid_err_65 = np.array([1.7, 2.1, 1.6, 1.7, 2.0])  # ps
MAE_65 = 0.0773823  # ps


mean_FWHM_65 = int(np.mean(FWHM_65))
mean_FWHM_err_65 = int(np.mean(FWHM_err_65))
mean_bias_65 = int(np.mean(abs(centroid_65 - positions)))


# 7 momentos
FWHM_75 = np.array([219.2, 222.9, 221.1, 220.8, 222.6])  # ps
FWHM_err_75 = np.array([1.9, 2.1, 1.9, 2.2, 2.4])  # ps
centroid_75 = np.array([-399.1, -199.7, 0.2, 199.9, 399.2])  # ps
centroid_err_75 = np.array([1.7, 2.0, 1.8, 2.0, 2.2])  # ps
MAE_75 = 0.0768556  # ps

mean_FWHM_75 = int(np.mean(FWHM_75))
mean_FWHM_err_75 = int(np.mean(FWHM_err_75))
mean_bias_75 = int(np.mean(abs(centroid_75 - positions)))

# 8 momentos
FWHM_85 = np.array([218.9, 220.3, 218.7, 220.4, 220.2])  # ps
FWHM_err_85 = np.array([1.9, 2.2, 2.0, 1.9, 2.4])  # ps
centroid_85 = np.array([-399.7, -199.6, -0.1, 200.0, 399.5])  # ps
centroid_err_85 = np.array([1.8, 2.0, 1.8, 1.7, 2.2])  # ps
MAE_85 = 0.0765812  # ps


mean_FWHM_85 = int(np.mean(FWHM_85))
mean_FWHM_err_85 = int(np.mean(FWHM_err_85))
mean_bias_85 = int(np.mean(abs(centroid_85 - positions)))


parameter_count = np.array([110, 160, 210, 260, 310, 360, 410, 460])
MAE_5_nodos = np.array([MAE_15, MAE_25, MAE_35, MAE_45, MAE_55, MAE_65, MAE_75, MAE_85])
bias_5_nodos = np.array([mean_bias_15, mean_bias_25, mean_bias_35, mean_bias_45, mean_bias_55, mean_bias_65, mean_bias_75, mean_bias_85])
FWHM_5_nodos = np.array([mean_FWHM_15, mean_FWHM_25, mean_FWHM_35, mean_FWHM_45, mean_FWHM_55, mean_FWHM_65, mean_FWHM_75, mean_FWHM_85])


#----------------------------------- [NM, 7, 1, 1] -------------------------------------------


# 1 momento
FWHM_17 = np.array([244.7, 248.0, 254.8, 247.1, 244.0])  # ps
FWHM_err_17 = np.array([2.1, 2.6, 2.3, 2.4, 2.8])  # ps
centroid_17 = np.array([-395.6, -197.6, -0.7, 197.4, 395.8])  # ps
centroid_err_17 = np.array([2.0, 2.4, 2.1, 2.2, 2.6])  # ps
MAE_17 = 0.0872856  # ps

mean_FWHM_17 = int(np.mean(FWHM_17))
mean_FWHM_err_17 = int(np.mean(FWHM_err_17))
mean_bias_17 = int(np.mean(abs(centroid_17- positions)))

# 2 momentos
FWHM_27 = np.array([234.5, 235.5, 237.1, 236.4, 234.6])  # ps
FWHM_err_27 = np.array([1.7, 2.2, 2.2, 1.9, 1.7])  # ps
centroid_27 = np.array([-398.3, -198.4, 0.0, 199.3, 399.9])  # ps
centroid_err_27 = np.array([1.6, 2.1, 2.0, 1.7, 1.6])  # ps
MAE_27 = 0.0817935  # ps

mean_FWHM_27 = int(np.mean(FWHM_27))
mean_FWHM_err_27 = int(np.mean(FWHM_err_27))
mean_bias_27 = int(np.mean(abs(centroid_27 - positions)))

# 3 momentos
FWHM_37 = np.array([227.8, 228.8, 229.6, 227.6, 228.2])  # ps
FWHM_err_37 = np.array([1.4, 1.9, 1.8, 1.6, 1.7])  # ps
centroid_37 = np.array([-399.5, -199.0, 0.5, 200.2, 399.4])  # ps
centroid_err_37 = np.array([1.3, 1.8, 1.7, 1.5, 1.6])  # ps
MAE_37 = 0.0787170  # ps


mean_FWHM_37 = int(np.mean(FWHM_37))
mean_FWHM_err_37 = int(np.mean(FWHM_err_37))
mean_bias_37 = int(np.mean(abs(centroid_37 - positions)))

# 4 momentos
FWHM_47 = np.array([227.0, 224.1, 226.2, 224.6, 225.6])  # ps
FWHM_err_47 = np.array([2.1, 2.2, 1.7, 1.5, 2.1])  # ps
centroid_47 = np.array([-399.7, -199.2, 0.6, 199.7, 399.3])  # ps
centroid_err_47 = np.array([2.0, 2.0, 1.6, 1.4, 2.0])  # ps
MAE_47 = 0.0784690  # ps


mean_FWHM_47 = int(np.mean(FWHM_47))
mean_FWHM_err_47 = int(np.mean(FWHM_err_47))
mean_bias_47 = int(np.mean(abs(centroid_47 - positions)))

# 5 momentos
FWHM_57 = np.array([220.9, 222.7, 222.8, 223.0, 221.9])  # ps
FWHM_err_57 = np.array([2.0, 1.7, 1.7, 2.0, 2.1])  # ps
centroid_57 = np.array([-399.9, -199.8, 0.5, 199.4, 399.5])  # ps
centroid_err_57 = np.array([1.9, 1.6, 1.5, 1.9, 2.0])  # ps
MAE_57 = 0.0777065  # ps

mean_FWHM_57 = int(np.mean(FWHM_57))
mean_FWHM_err_57 = int(np.mean(FWHM_err_57))
mean_bias_57 = int(np.mean(abs(centroid_57 - positions)))

# 6 momentos
FWHM_67 = np.array([219.4, 222.6, 220.5, 220.9, 222.1])  # ps
FWHM_err_67 = np.array([1.8, 2.0, 2.0, 2.0, 2.4])  # ps
centroid_67 = np.array([-398.7, -199.1, -0.0, 200.2, 400.2])  # ps
centroid_err_67 = np.array([1.6, 1.9, 2.0, 1.9, 2.2])  # ps
MAE_67 = 0.0771570  # ps

mean_FWHM_67 = int(np.mean(FWHM_67))
mean_FWHM_err_67 = int(np.mean(FWHM_err_67))
mean_bias_67 = int(np.mean(abs(centroid_67 - positions)))

# 7 momentos
FWHM_77 = np.array([218.6, 221.6, 220.1, 219.4, 221.3])  # ps
FWHM_err_77 = np.array([1.7, 2.3, 1.7, 2.1, 2.5])  # ps
centroid_77 = np.array([-398.6, -198.9, 0.6, 200.5, 399.9])  # ps
centroid_err_77 = np.array([1.6, 2.1, 1.6, 2.0, 2.3])  # ps
MAE_77 = 0.0768948  # ps

mean_FWHM_77 = int(np.mean(FWHM_77))
mean_FWHM_err_77 = int(np.mean(FWHM_err_77))
mean_bias_77 = int(np.mean(abs(centroid_77 - positions)))

# 8 momentos 
FWHM_87 = np.array([217.0, 220.0, 218.9, 219.3, 220.0])  # ps
FWHM_err_87 = np.array([1.8, 1.9, 1.8, 1.9, 2.2])  # ps
centroid_87 = np.array([-399.3, -199.4, 0.0, 199.8, 399.3])  # ps
centroid_err_87 = np.array([1.7, 1.8, 1.7, 1.7, 2.1])  # ps
MAE_87 = 0.0767145  # ps

mean_FWHM_87 = int(np.mean(FWHM_87))
mean_FWHM_err_87 = int(np.mean(FWHM_err_87))
mean_bias_87 = int(np.mean(abs(centroid_87 - positions)))



parameter_count = np.array([150, 220, 290, 360, 430, 500, 570, 640])
MAE_7_nodos = np.array([MAE_17, MAE_27, MAE_37, MAE_47, MAE_57, MAE_67, MAE_77, MAE_87])
bias_7_nodos = np.array([mean_bias_17, mean_bias_27, mean_bias_37, mean_bias_47, mean_bias_57, mean_bias_67, mean_bias_77, mean_bias_87])
FWHM_7_nodos = np.array([mean_FWHM_17, mean_FWHM_27, mean_FWHM_37, mean_FWHM_47, mean_FWHM_57, mean_FWHM_67, mean_FWHM_77, mean_FWHM_87])

#----------------------------------- [NM, 9, 1, 1] -------------------------------------------

# 1 momento 
FWHM_19 = np.array([244.5, 248.7, 255.7, 247.1, 243.7])  # ps
FWHM_err_19 = np.array([2.0, 2.5, 2.4, 2.4, 2.8])  # ps
centroid_19 = np.array([-395.7, -197.8, -0.9, 197.4, 395.7])  # ps
centroid_err_19 = np.array([1.9, 2.3, 2.2, 2.2, 2.6])  # ps
MAE_19 = 0.0869315  # ps

mean_FWHM_19 = int(np.mean(FWHM_19))
mean_FWHM_err_19 = int(np.mean(FWHM_err_19))
mean_bias_19 = int(np.mean(abs(centroid_19- positions)))


# 2 momentos
FWHM_29 = np.array([234.6, 236.2, 238.1, 236.7, 234.6])  # ps
FWHM_err_29 = np.array([1.6, 2.5, 2.4, 2.0, 1.8])  # ps
centroid_29 = np.array([-398.0, -198.0, 0.4, 199.6, 400.3])  # ps
centroid_err_29 = np.array([1.5, 2.3, 2.2, 1.8, 1.7])  # ps
MAE_29 = 0.0825217  # ps

mean_FWHM_29 = int(np.mean(FWHM_29))
mean_FWHM_err_29 = int(np.mean(FWHM_err_29))
mean_bias_29 = int(np.mean(abs(centroid_29- positions)))


#3 momentos
FWHM_39 = np.array([229.8, 229.5, 230.4, 228.5, 229.7])  # ps
FWHM_err_39 = np.array([1.9, 2.2, 1.8, 1.6, 1.6])  # ps
centroid_39 = np.array([-399.3, -199.1, -0.1, 199.3, 399.8])  # ps
centroid_err_39 = np.array([1.8, 2.1, 1.7, 1.5, 1.5])  # ps
MAE_39 = 0.0794405  # ps

mean_FWHM_39 = int(np.mean(FWHM_39))
mean_FWHM_err_39 = int(np.mean(FWHM_err_39))
mean_bias_39 = int(np.mean(abs(centroid_39- positions)))


# 4 momentos
FWHM_49 = np.array([224.2, 224.4, 224.7, 224.2, 223.7])  # ps
FWHM_err_49 = np.array([1.6, 1.8, 2.0, 1.5, 1.8])  # ps
centroid_49 = np.array([-399.9, -199.7, -0.1, 199.3, 399.1])  # ps
centroid_err_49 = np.array([1.5, 1.6, 1.8, 1.4, 1.7])  # ps
MAE_49 = 0.0780771  # ps

mean_FWHM_49 = int(np.mean(FWHM_49))
mean_FWHM_err_49 = int(np.mean(FWHM_err_49))
mean_bias_49 = int(np.mean(abs(centroid_49- positions)))


# 5 momentos
FWHM_59 = np.array([221.3, 222.3, 222.4, 221.8, 221.2])  # ps
FWHM_err_59 = np.array([1.8, 1.7, 1.6, 1.6, 1.5])  # ps
centroid_59 = np.array([-399.7, -199.2, 0.3, 199.4, 399.6])  # ps
centroid_err_59 = np.array([1.7, 1.5, 1.5, 1.5, 1.4])  # ps
MAE_59 = 0.0767302  # ps

mean_FWHM_59 = int(np.mean(FWHM_59))
mean_FWHM_err_59 = int(np.mean(FWHM_err_59))
mean_bias_59 = int(np.mean(abs(centroid_59- positions)))

# 6 momentos
FWHM_69 = np.array([219.2, 222.7, 220.7, 220.8, 223.2])  # ps
FWHM_err_69 = np.array([1.7, 2.0, 1.9, 2.0, 2.2])  # ps
centroid_69 = np.array([-399.0, -199.2, 0.3, 200.2, 400.0])  # ps
centroid_err_69 = np.array([1.6, 1.8, 1.8, 1.8, 2.1])  # ps
MAE_69 = 0.0772507  # ps

mean_FWHM_69 = int(np.mean(FWHM_69))
mean_FWHM_err_69 = int(np.mean(FWHM_err_69))
mean_bias_69 = int(np.mean(abs(centroid_69- positions)))


# 7 momentos
FWHM_79 = np.array([219.3, 221.6, 219.4, 219.2, 221.4])  # ps
FWHM_err_79 = np.array([1.7, 2.2, 2.0, 1.7, 2.2])  # ps
centroid_79 = np.array([-399.4, -199.7, 0.1, 199.9, 399.4])  # ps
centroid_err_79 = np.array([1.6, 2.1, 1.9, 1.6, 2.0])  # ps
MAE_79 = 0.0772178  # ps)

mean_FWHM_79 = int(np.mean(FWHM_79))
mean_FWHM_err_79 = int(np.mean(FWHM_err_79))
mean_bias_79 = int(np.mean(abs(centroid_79- positions)))

#8 momentos
FWHM_89 = np.array([219.5, 220.8, 219.4, 219.8, 220.7])  # ps
FWHM_err_89 = np.array([1.7, 2.2, 2.0, 2.1, 2.3])  # ps
centroid_89 = np.array([-399.7, -199.8, -0.0, 199.6, 399.5])  # ps
centroid_err_89 = np.array([1.6, 2.0, 2.0, 2.0, 2.2])  # ps
MAE_89 = 0.0766852  # ps

mean_FWHM_89 = int(np.mean(FWHM_89))
mean_FWHM_err_89 = int(np.mean(FWHM_err_89))
mean_bias_89 = int(np.mean(abs(centroid_89- positions)))


MAE_9_nodos = np.array([MAE_19, MAE_29, MAE_39, MAE_49, MAE_59, MAE_69, MAE_79, MAE_89])
bias_9_nodos = np.array([mean_bias_19, mean_bias_29, mean_bias_39, mean_bias_49, mean_bias_59, mean_bias_69, mean_bias_79, mean_bias_89])
FWHM_9_nodos = np.array([mean_FWHM_19, mean_FWHM_29, mean_FWHM_39, mean_FWHM_49, mean_FWHM_59, mean_FWHM_69, mean_FWHM_79, mean_FWHM_89])

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
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8], fontsize = 8)  # Y ticks from 1 to 8


plt.subplot(132)
plt.imshow(BIAS, cmap = 'inferno_r', interpolation = 'nearest', aspect = 'auto')
plt.colorbar()
plt.title('BIAS (ps)', fontsize = 10)
plt.xlabel('Node number', fontsize = 8)
plt.ylabel('Number of features', fontsize = 8)
plt.xticks([0, 1, 2, 3], [3, 5, 7, 9], fontsize = 8)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8], fontsize = 8)

plt.subplot(133)
plt.imshow(FWHM, cmap = 'inferno_r', interpolation = 'nearest', aspect = 'auto')
plt.colorbar()
plt.title('FWHM (ps)', fontsize = 10)
plt.xlabel('Node number', fontsize = 8)
plt.ylabel('Number of features', fontsize = 8)
plt.xticks([0, 1, 2, 3], [3, 5, 7, 9], fontsize = 8)
plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6, 7, 8], fontsize = 8)

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