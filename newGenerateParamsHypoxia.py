import pandas as pd
import numpy as np
import new_data_processing_hypoxia as dp
errorControl = pd.read_csv("new setonix hypoxia increased growth errors control.csv")
errorControl = list(np.transpose(np.array(errorControl))[0])
errorRT = pd.read_csv("new matched params hypoxia errors RT.csv")
errorRT = list(np.transpose(np.array(errorRT))[0])
param = pd.read_csv("new matched params hypoxia means RT.csv")
param = list(np.transpose(np.array(param))[0])
#print(get_equivalent_bed_treatment(param, 50, 1))
errors = [errorControl, errorRT]
errorMerged = dp.merge_lists(errors)
#copy from non hypoxia fit
errorMerged[6] = 1.4332917616497526e-17
errorMerged[9] = 2.1870296656032597e-22
errorMerged[12] = 3.818502442964274
errorMerged[13] = 0.033750688242235
errorMerged[14] = 0.6929734647608269
errorMerged[15] = 0.1659055513160591
errorMerged[16] = 0.0459942654015681
errorMerged[17] = 0.0045994265401568
errorMerged[19] = 1.4332917616497526e-17
errorMerged[20] = 311459.5233386332
errorMerged[21] = 0.1280491060030706
errorMerged[22] = 1.9332480841721483e-08
errorMerged[23] = 1.7712575793162017e-06
errorMerged[24] = 0
errorMerged[26] = 1.4332917616497526e-17
errorMerged[27] = 5.576293104852031e-08
errorMerged[28] = 0.3057551035393502
errorMerged[29] = 0.8349760556171322
errorMerged[30] = 0.4143561469321419
errorMerged[33] = 0.0124761604376376
errorMerged[36] = 2.866583523299505e-17
errorMerged[37] = 0
errorMerged[38] = 0.1550141870929539
#errorMerged[39] = 1.6174722420261087e+60
num_patients = 500
params = [list(param) for _ in range(num_patients)]
#print(errorMerged)

seeds = range(num_patients)
# Modify the parameters for each patient
for i in range(num_patients):
    rng = np.random.default_rng(seeds[i])
    for j in range(len(param)):
        
        if errorMerged[j] != 0:
            logNormalParams = dp.log_normal_parameters(param[j], errorMerged[j])
            params[i][j] = min(max(rng.lognormal(mean=logNormalParams[0], sigma = logNormalParams[1]), 0.8*param[j]), 1.2*param[j])
        else:
            params[i][j] = param[j]
        if j == 28:
            params[i][j] = min(max(rng.lognormal(mean=logNormalParams[0], sigma = logNormalParams[1]), 0.8*param[j]), 1.2*param[j])

# Assuming 'params' is your list of parameters
df = pd.DataFrame(params)
print(params)
# Save to CSV
df.to_csv('new_matched_hypoxia_parameters.csv', index=False)
