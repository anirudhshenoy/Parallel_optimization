import os
import glob

os.chdir(r'C:\Users\Aniru_000\Desktop\TD-1\Airfoil\s1223\airfoil\Python Code\tempfiles')
while True:
        try:
            [os.remove(x) for x in glob.glob("session_Thread*.txt")]
            [os.remove(x) for x in glob.glob("airfoil_Thread*.dat")]
            [os.remove(x) for x in glob.glob("output_Thread*.txt")]
        except PermissionError:
            print("Unable To delete")
        else:
            break
print("Completed")