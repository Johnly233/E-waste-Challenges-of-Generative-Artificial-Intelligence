import numpy as np
import xlrd
import pandas as pd
from itertools import starmap as smap

#Data initialization
S1_total = pd.DataFrame()
S2_total = pd.DataFrame()
C1_total = pd.DataFrame()
C2_total = pd.DataFrame()

#Import excel data, parameter initialization
input_path = r'Demo data.xls'
output_path = r'Result.xlsx'

excel = xlrd.open_workbook(input_path)
compute_data = excel.sheet_by_index(1)
Nt_para = compute_data.col_values(colx=0)
Nt_para.pop(0)
Nt_traindata = compute_data.col_values(colx=1)
Nt_traindata.pop(0)
Pt_theo = compute_data.col_values(colx=2)
Pt_theo.pop(0)
Pi_theo = compute_data.col_values(colx=3)
Pi_theo.pop(0)
Ni_user = compute_data.col_values(colx=4)
Ni_user.pop(0)

config_data = excel.sheet_by_index(0)
random_batch = int(config_data.cell(0,1).value)
l_range = int(config_data.cell(1,1).value)
day = int(config_data.cell(2,1).value)
GPU_per_server = int(config_data.cell(3,1).value)
n_model = int(config_data.cell(4,1).value)
upgrade_strategy = int(config_data.cell(5,1).value)

T = len(Pt_theo)

def fcn1(i,P):
    if P[i]==P[i-1]:
        return 1
    else:
        return 0

def randomize(n,sigma):  #Randomize the parameter
        return np.random.normal(n,sigma)
    
def calc_sigma_002(n):  #Calculate 1-sigma with ±2%'s deviation
    return n*0.02

def calc_sigma_005(n):  #Calculate 1-sigma with ±5%'s deviation
    return n*0.05

def calc_Rt(nt_para,nt_traindata,nt_model):
    return nt_para*nt_traindata*nt_model*125/1800/day

def calc_Ri(nt_para,ni_user,ni_query_amount):
    return nt_para*ni_user*ni_query_amount/43200

l = l_range
for r in range(random_batch):
        S1_t = np.zeros(T)
        S2_t = np.zeros(T)
        S1_i = np.zeros(T)
        S2_i = np.zeros(T)
        
        nt_model = round(randomize(n_model,1),0)  
        eff_t = randomize(0.3,0.005)  #Computing efficiency for training
        eff_i = randomize(0.46,0.01)  #Computing efficiency for inference
        ni_query_amount = randomize(10000,500)  #Token requirements per capital per day
        nt_para = list(smap(randomize, zip(Nt_para, list(map(calc_sigma_002,Nt_para)))))
        nt_traindata = list(smap(randomize, zip(Nt_traindata, list(map(calc_sigma_002,Nt_traindata)))))
        ni_user = list(smap(randomize, zip(Ni_user, list(map(calc_sigma_005,Ni_user)))))
        pt_theo = list(smap(randomize, zip(Pt_theo, list(map(calc_sigma_002,Pt_theo)))))
        pi_theo = list(smap(randomize, zip(Pi_theo, list(map(calc_sigma_002,Pi_theo)))))
        
        
        R_t = list(smap(calc_Rt,zip(nt_para,nt_traindata,[nt_model]*T)))
        R_i = list(smap(calc_Ri,zip(nt_para,ni_user,[ni_query_amount]*T)))
        R_t.insert(0,0)
        R_i.insert(0,0)
        P_t = [e*eff_t*GPU_per_server for e in pt_theo]
        P_i = [e*eff_i*GPU_per_server for e in pi_theo]
        
        
        for i in range(T):
            if i>=l:  #for train
                S1_t[i] = np.ceil((R_t[i+1]-R_t[i]+P_t[i-l]*S1_t[i-l])/P_t[i])
            else:
                S1_t[i] = np.ceil((R_t[i+1]-R_t[i])/P_t[i])
            S2_t[i] = np.ceil((R_t[i+1]-fcn1(i,P_t)*R_t[i])/P_t[i])
            
            if i>=l:  #for infer
                S1_i[i] = np.ceil((R_i[i+1]-R_i[i]+P_i[i-l]*S1_i[i-l])/P_i[i])
            else:
                S1_i[i] = np.ceil((R_i[i+1]-R_i[i])/P_i[i])
            S2_i[i] = np.ceil((R_i[i+1]-fcn1(i,P_i)*R_i[i])/P_i[i])
        
        S1 = [S1_t[i]+S1_i[i] for i in range(T)]
        S2 = [S2_t[i]+S2_i[i] for i in range(T)]
        
        C1 = np.zeros(T)
        C2 = np.zeros(T)
        C1[0] = S1[0]
        for i in range(T):
            if i<l:
                C1[i] = C1[i-1]+S1[i]
            else:
                C1[i] = C1[i-1]+S1[i]-S1[i-l]
            C2[i] = np.ceil(R_t[i+1]/P_t[i]+R_i[i+1]/P_i[i])
        
      # Store result in DataFrame
        S1_total[r] = S1
        S2_total[r] = S2
        C1_total[r] = C1
        C2_total[r] = C2

#Choose Stepwise Upgrade Strategy result or Continuous Upgrading Strategy result
if upgrade_strategy == 1:
	S_final = S1_total
elif upgrade_strategy == 2:
	S_final = S2_total
	
Exp = pd.DataFrame()

#Calculate mean value and standard deviation
mean_list = []
std_list = []
std_cum_list = []
for i in range(T):
    mean_list.append(np.mean(S_final.loc[i,:]))
    std_list.append(np.std(S_final.loc[i,:], ddof=1))
    std_cum_list.append(np.std([np.sum(S_final.loc[0:i,j]) for j in range(random_batch)]))
mean_list.append(np.mean([np.sum(S_final.loc[:,i]) for i in range(S_final.shape[1]-1)]))
std_list.append(np.std([np.sum(S_final.loc[:,i]) for i in range(S_final.shape[1]-1)], ddof=1))
std_cum_list.append(0)

Exp['mean'] = mean_list
Exp['std'] = std_list
Exp['std_cum'] = std_cum_list
with pd.ExcelWriter(output_path) as writer:
    Exp.to_excel(writer, sheet_name = 'S')
    
