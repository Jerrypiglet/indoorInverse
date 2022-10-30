import os,sys,inspect
pwd_path = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, pwd_path)
sys.path.insert(0, os.path.join(pwd_path, 'train'))
print(sys.path)
