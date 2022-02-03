import os,sys,inspect
pwdpath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, pwdpath)
sys.path.insert(0, os.path.join(pwdpath, 'train'))
print(sys.path)
