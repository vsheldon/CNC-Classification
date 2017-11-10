import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
	
	S1 = np.loadtxt('S1.txt')
	X1 = np.loadtxt('X1.txt')
	Y1 = np.loadtxt('Y1.txt')
	Z1 = np.loadtxt('Z1.txt')
	print X1.shape
	print Y1.shape
	print Z1.shape
	#plt.plot(X1[:,0],'r--',X1[:,3],'b^')
	#plt.plot(X1[:,1],'g--',X1[:,4],'k^')
	#plt.plot(X1[:,2],'y--',X1[:,5],'r^')
	#plt.show()



