import numpy as np

"""Constants"""
ITER_TOLERANCE=0.01
ITER_LIMIT    =200
cl           =np.ones(int(r/2))*cl_alpha*alpha_wing
cl           =np.append(cl,0)       
cl           =cl[::-1]
c_b          =np.ones(int(r/2))*c/b
c_b          =np.append(c_b,0)
c_b          =c_b[::-1]
#c            =c*np.ones((n+1)) 
def pad_arrays(arr):
    arr=np.append(arr,0)
    arr=arr[::-1]
    return arr

def get_eta_m(m):
        eta_m=(np.pi/(2*r))*np.sin(m*np.pi/r)
        if (m==int(r/2)):
            eta_ms=eta_m
        else:
            eta_ms=2*eta_m
        return eta_ms
        
def get_lambda_mk(m,k):
    if (k+m)%2!=0:
        if m==int(r/2):
            lambda_mk=-180/(np.pi*r*((np.cos(2*k*np.pi/r)+1)))
        else:
            a=np.pi*(k+m)/r
            b=np.pi*(k-m)/r             
            lambda_mk=((1/(np.tan(a)*np.sin(a)))-(1/(np.tan(b)*np.sin(b))))*(180/(2*np.pi*r*np.sin(k*np.pi/r)))    
        
    elif k==m:
        lambda_mk=(180*r)/(8*np.pi*np.sin(k*np.pi/r))
         
    else:
        lambda_mk=0
    return lambda_mk

"""Non Linear LLT Code Based on NACA TN 1269
b          =Wing span(m)
c          =array of chord lengths (m) array length=r/2
cl_alpha   =slopes for different reynolds nos
alpha_wing =Operating angle
alpha_0    =Zero-lift angle (array)
A          =Aspect Ratio 
r          =No of sections (same as k from NACA TN 1269)"""    
    
def nonlinear_llt(b,c,cl_alpha,alpha_wing,r,A,alpha_0):
    c_b=np.ones(int(r/2))*c/b
    cl=np.ones(int(r/2))*cl_alpha*alpha_wing
    cl=pad_arrays(cl)
    c_b=pad_arrays(c_b)
    cl_c_b =cl*c_b
    for iter in range(ITER_LIMIT):
        alpha_i=np.array([])   
        for k in range (int(r/2),0,-1):
            alpha_i_temp=0
            for m in np.arange(int(r/2),0,-1):
                lambda_mk=get_lambda_mk(m,k)
                alpha_i_temp+=cl_c_b[m]*lambda_mk
            alpha_i=np.append(alpha_i,alpha_i_temp)        
        alpha_eff=alpha_wing-alpha_i
        alpha_eff=pad_arrays(alpha_eff)
        cl=cl_alpha*(alpha_eff-foil_alpha_0)
        cl_c_b_new=cl*(c_b)
        if (np.fabs(np.sum(cl_c_b_new-cl_c_b))<ITER_TOLERANCE):
            print(iter)
            break
        else:
            cl_c_b=cl_c_b+0.05*(cl_c_b_new-cl_c_b)
    CL=0
    CD_i=0
    for m in range(int(r/2),0,-1):
        CL+=cl_c_b[m]*get_eta_m(m)
        CD_i+=cl_c_b[m]*alpha_eff[m]*get_eta_m(m)
    CL=A*CL
    CD_i=(np.pi/180)*A*CD_i
    print("CL:" +str(CL))
    print("CD_i: "+str(CD_i))

