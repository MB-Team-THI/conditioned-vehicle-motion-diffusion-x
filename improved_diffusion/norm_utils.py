import torch as th
MAX_ACC = 4
MAX_dPSI = 0.7
MAX_X = 200.0
MAX_Y = 10.0

## NORM Vehicle Motion Model
#-------------------------------------------------------------------------------------
def vmm_norm(acc, d_psi):
    # keys =  ['ego_x', 'ego_y', 'ego_psi', 'ego_vx', 'ego_vy', 'ego_dpsi', 'ego_ax', 'ego_ay']
    flag = False
    
    if th.max((d_psi)) > MAX_dPSI:
        flag = True
    if th.min(d_psi) < -MAX_dPSI:
        flag = True
    
    norm_acc = (acc / MAX_ACC).clamp(-1, 1)
    norm_dpsi = (d_psi / MAX_dPSI).clamp(-1,1)
    batch_norm = th.cat((norm_acc, norm_dpsi),dim=1)
    return batch_norm, flag

def inv_vmm_norm(x):                 # x: (B,2,T) in normalized space
    scale = x.new_tensor([MAX_ACC, MAX_dPSI]).view(1,2,1)
    return x * scale                 # returns NEW tensor
    
## NORM XY Prediction
#-------------------------------------------------------------------------------------
def xy_norm(x, y):    
    # keys = [ .... ]
    norm_x = (x/MAX_X).clamp(-1, 1)
    norm_y = (y/MAX_Y).clamp(-1, 1)
    batch_norm = th.cat((norm_x, norm_y),dim=1)
    return batch_norm, None

def inv_xy_norm(batch):
    scale = batch.new_tensor([MAX_X, MAX_Y]).view(1,2,1)
    return batch * scale  


