import numpy as np

"""Constants"""
ITER_TOLERANCE=0.0001
ITER_LIMIT    =300


"""Non Linear LLT Code Based on NACA TN 1269
b          =Wing span(m)
c          =array of chord lengths (m) array length=r/2
cl_alpha   =slopes for different reynolds nos
alpha_wing =Operating angle
alpha_0    =Zero-lift angle (array)
A          =Aspect Ratio 
r          =No of sections (same as k from NACA TN 1269)"""    

#For testing
# r=20
# b=2
# c=np.array([ 0.3   ,  0.2833,  0.2667,  0.25  ,  0.2333,  0.2167,  0.2   ,
        # 0.1833,  0.1667,  0.15  ])
# alpha_wing=3
# alpha_0=np.array([-1.8209, -1.7901, -1.7593, -1.7285, -1.6977, -1.6669 ,-1.6361 ,-1.6053, -1.5746, -1.5438])
# A=b**2/(b*0.233)
# cl_alpha=np.array([ 0.1061,  0.107 ,  0.1079,  0.1088,  0.1096,  0.1105 , 0.1114 , 0.1123 , 0.1132,  0.114 ])
def pad_arrays(arr):
    arr=np.append(arr,0)
    arr=arr[::-1]
    return arr

def get_eta_m(m,r):
        eta_m=(np.pi/(2*r))*np.sin(m*np.pi/r)
        if (m==int(r/2)):
            eta_ms=eta_m
        else:
            eta_ms=2*eta_m
        return eta_ms
        
def get_lambda_mk(m,k,r):
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

    
def LLT(b,c,cl_alpha,alpha_wing,r,wing_area,alpha_0,cd_0,cd_coeff):
    
    
    c=pad_arrays(c)
    alpha_0=pad_arrays(alpha_0)
    cl_alpha=pad_arrays(cl_alpha)
    cd_0=pad_arrays(cd_0)
    cl=cl_alpha*alpha_wing
    c_b=c/b
    cl_c_b =cl*c_b
    A=(b**2)/wing_area
    for iter in range(ITER_LIMIT):
        alpha_i=np.array([])   
        for k in range (int(r/2),0,-1):
            alpha_i_temp=0
            for m in np.arange(int(r/2),0,-1):
                lambda_mk=get_lambda_mk(m,k,r)
                alpha_i_temp+=cl_c_b[m]*lambda_mk
            alpha_i=np.append(alpha_i,alpha_i_temp)        
        alpha_eff=alpha_wing-alpha_i
        alpha_eff=pad_arrays(alpha_eff)
        alpha_eff-=alpha_0
        cl=cl_alpha*(alpha_eff)
        cd=(cd_coeff[2]*cl**2)+(cd_coeff[1]*cl)+cd_0
        cl_c_b_new=cl*(c_b)
        if (np.fabs(np.sum(cl_c_b_new-cl_c_b))<ITER_TOLERANCE):
            #print(iter)
            break
        else:
            cl_c_b=cl_c_b+0.01*(cl_c_b_new-cl_c_b)
    CL=0
    CD_i=0
    CD_0=0
    c_bar=wing_area/b
    cd_c_cbar=cd*c/c_bar
    alpha_i=pad_arrays(alpha_i)
    for m in range(int(r/2),0,-1):
        CL+=cl_c_b[m]*get_eta_m(m,r)
        CD_i+=cl_c_b[m]*alpha_i[m]*get_eta_m(m,r)
        CD_0+=cd_c_cbar[m]*get_eta_m(m,r)
    CL=A*CL
    CD_i=(np.pi/180)*A*CD_i
    CD=np.fabs(CD_i)+CD_0
    # print("CL:" +str(CL))
    # print("CD_i: "+str(CD_i))
    # print("CD_0 "+str(CD_0))
    return CL,CD

