import numpy as np
import xlrd
import pandas as pd
from itertools import starmap as smap

S1_total = pd.DataFrame()
S2_total = pd.DataFrame()
C1_total = pd.DataFrame()
C2_total = pd.DataFrame()
R_total = pd.DataFrame()
Out_total = pd.DataFrame()

#Import excel data
input_path = r'Demo data.xls'
output_path = r'Result.xlsx'
excel = xlrd.open_workbook(input_path)
sheet = excel.sheet_by_index(1)
Nt_para = sheet.col_values(colx=0)
Nt_para.pop(0)
Nt_traindata = sheet.col_values(colx=1)
Nt_traindata.pop(0)
Pt_theo = sheet.col_values(colx=2)
Pt_theo.pop(0)
Pi_theo = sheet.col_values(colx=3)
Pi_theo.pop(0)
Ni_user = sheet.col_values(colx=4)
Ni_user.pop(0)
Eff_t = sheet.col_values(colx=5)
Eff_t.pop(0)
Eff_i = sheet.col_values(colx=6)
Eff_i.pop(0)
Sp_rate = sheet.col_values(colx=7)
Sp_rate.pop(0)
N_model = sheet.col_values(colx=8)
N_model.pop(0)
sheet2 = excel.sheet_by_index(2)
lifespan_distribution = sheet2.col_values(colx=0)
lifespan_distribution.pop(0)

config_data = excel.sheet_by_index(0)
random_batch = int(config_data.cell(0,1).value)
l_range = int(config_data.cell(1,1).value)
day = int(config_data.cell(2,1).value)
GPU_per_server = int(config_data.cell(3,1).value)
upgrade_strategy = int(config_data.cell(4,1).value)
n_token = int(config_data.cell(5,1).value)

lifespan_distribution = [0]*int(l_range-len(lifespan_distribution)/2) + lifespan_distribution + [0]*int(l_range-len(lifespan_distribution)/2)

T = len(Pt_theo)
len_lifespan = len(lifespan_distribution)

def cum_p_outflow(time_index, P, S_out):
    cum_result = 0
    index = min(time_index, len_lifespan)
    for j in range(index):
        cum_result = cum_result + P[time_index-j] * S_out[time_index-j]
    return cum_result

def cum_s_outflow(time_index, S_in):
    cum_result = 0
    index = min(time_index, len_lifespan)
    for j in range(index):
        cum_result = cum_result + S_in[time_index-j-1] * float(lifespan_distribution[j])
    return cum_result

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

def calc_Rt(nt_para,nt_traindata,nt_model,sp_rate):
    return nt_para*nt_traindata*nt_model/86.4/day/sp_rate

def calc_Ri(nt_para,ni_user,ni_query_amount,sp_rate):
    return nt_para*ni_user*ni_query_amount/86.4*1.25/sp_rate

PDF = [0.03, 0.137, 2.14, 13.59, 34.13, 34.13, 13.59, 2.14, 0.137, 0.003]

l = l_range
for r in range(random_batch):
        S1_t = np.zeros(T)
        S2_t = np.zeros(T)
        S1_i = np.zeros(T)
        S2_i = np.zeros(T)
        Out_t = np.zeros(T)
        Out_i = np.zeros(T)
        
        eff_t = [randomize(x,0.005) for x in Eff_t]  #Computing efficiency for training
        eff_i = [randomize(x,0.01) for x in Eff_i]  #Computing efficiency for inference
        if Sp_rate[0] == 1:
            sp_rate = [x for x in Sp_rate]
        else:
            sp_rate = [randomize(x,0.25) for x in Sp_rate]  #Sparsity rate of model
        ni_query_amount = randomize(n_token,100)  #Token requirements per capital per day
        nt_para = list(smap(randomize, zip(Nt_para, list(map(calc_sigma_002,Nt_para)))))
        nt_traindata = list(smap(randomize, zip(Nt_traindata, list(map(calc_sigma_002,Nt_traindata)))))
        ni_user = list(smap(randomize, zip(Ni_user, list(map(calc_sigma_005,Ni_user)))))
        pt_theo = list(smap(randomize, zip(Pt_theo, list(map(calc_sigma_002,Pt_theo)))))
        pi_theo = list(smap(randomize, zip(Pi_theo, list(map(calc_sigma_002,Pi_theo)))))
        
        R_t = list(smap(calc_Rt, zip(nt_para, nt_traindata, N_model, sp_rate)))
        R_i = list(smap(calc_Ri,zip(nt_para,ni_user,[ni_query_amount]*T,sp_rate)))
        R_t.insert(0,0)
        R_i.insert(0,0)
        P_t = [x*eff_t[i]*GPU_per_server for i,x in enumerate(pt_theo)]
        P_i = [x*eff_i[i]*GPU_per_server for i,x in enumerate(pi_theo)]
        
        S1_t[0] = np.ceil((R_t[1]-R_t[0])/P_t[0])
        S1_i[0] = np.ceil((R_i[1]-R_i[0])/P_i[0])
        for i in range(1,T):
            
            Out_t[i] = cum_s_outflow(i, S1_t)
            S1_t[i] = np.ceil((R_t[i+1]-R_t[i]+cum_p_outflow(i, P_t, Out_t))/P_t[i])
            S2_t[i] = np.ceil((R_t[i+1]-fcn1(i,P_t)*R_t[i])/P_t[i])
            
            Out_i[i] = cum_s_outflow(i, S1_i)
            S1_i[i] = np.ceil((R_i[i+1]-R_i[i]+cum_p_outflow(i, P_i, Out_i))/P_i[i])
            S2_i[i] = np.ceil((R_i[i+1]-fcn1(i,P_i)*R_i[i])/P_i[i])
        
        S1 = [S1_t[i]+S1_i[i] for i in range(T)]
        S2 = [S2_t[i]+S2_i[i] for i in range(T)]
        Out = [Out_t[i]+Out_i[i] for i in range(T)]
        
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
        R_total[r] = list(np.add(R_t, R_i))
        Out_total[r] = Out
        
if upgrade_strategy==1:
    S_final = S1_total  #This is used to choose Stepwise Upgrade Strategy result or Continuous Upgrading Strategy result
else:
    S_final = S2_total

Exp = pd.DataFrame()

#Calculate mean value and standard deviation
mean_list = []
std_list = []
std_cum_list = []
mean_R = []
mean_out = []
std_out_list = []
std_cum_out_list = []
for i in range(T):
    mean_list.append(np.mean(S_final.loc[i,:]))
    std_list.append(np.std(S_final.loc[i,:], ddof=1))
    std_cum_list.append(np.std([np.sum(S_final.loc[0:i,j]) for j in range(random_batch)]))
    mean_R.append(np.mean(R_total.loc[i,:]))
    mean_out.append(np.mean(Out_total.loc[i,:]))
    std_out_list.append(np.std(Out_total.loc[i,:], ddof=1))
    std_cum_out_list.append(np.std([np.sum(Out_total.loc[0:i,j]) for j in range(random_batch)]))
mean_list.append(np.mean([np.sum(S_final.loc[:,i]) for i in range(S_final.shape[1])]))
std_list.append(np.std([np.sum(S_final.loc[:,i]) for i in range(S_final.shape[1])], ddof=1))
std_cum_list.append(0)
mean_R.append(np.mean([np.sum(R_total.loc[:,i]) for i in range(R_total.shape[1])]))
mean_out.append(np.mean([np.sum(Out_total.loc[:,i]) for i in range(Out_total.shape[1])]))
std_out_list.append(np.std([np.sum(Out_total.loc[:,i]) for i in range(Out_total.shape[1])], ddof=1))
std_cum_out_list.append(0)

Exp['mean'] = mean_list
Exp['std'] = std_list
Exp['std_cum'] = std_cum_list
Exp['Out'] = mean_out
Exp['std_out'] = std_out_list
Exp['std_cum_out'] = std_cum_out_list
with pd.ExcelWriter(output_path) as writer:
    Exp.to_excel(writer, sheet_name = 'S')

print('done')
    