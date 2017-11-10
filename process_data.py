import glob
import numpy as np
import sys

# to get start, make sure you install following packages:
# glob, numpy

# to install using pip (Linux):
# sudo pip install glob
# sudo pip install numpy

# to use this script:
# first create a folder called "data" under the same directory as this file
# put all the raw files into "data" folder
# then run the following command in terminal
#	 python process_data.py  data

# data will be saving into 4 files:
# X1.txt, Y1.txt, Z1.txt, S1.txt

# each file contains 10-12 columns 
# (X,Y has 11 col, Z has 10 col, S has 12 col)
# represents [ActualPosition, ActualVelocity, 
#			  ActualAcceleration, CommandPosition
#             CommandVelocity, CommandAcceleration, 
#			  CurrentFeedback, DCBusVoltage,
#			  OutputCurrent, OutputVoltage,
#             OutputPower, SystemInertia]



def data_processing(directory):
	X1, Y1, Z1, S1=[],[],[],[]
	MCPN, MSN, MCF = [],[],[]
	MCPN_ = 'M1_CURRENT_PROGRAM_NUMBER'
	MSN_ = 'M1_sequence_number'
	MCF_ = 'M1_CURRENT_FEEDRATE'
	for file in glob.glob(directory+"/*.dat"):
		cnt = 0
		tmp_x = np.arange(12,dtype='f')
		tmp_y = np.arange(12,dtype='f')
		tmp_z = np.arange(12,dtype='f')
		tmp_s = np.arange(12,dtype='f')

		for line in open(file,'r').read().split("},{"):

			tag = line[line.find("TagValue")+11:]
			val = float(tag[0:tag.find('"')])
			#print val
			if 'X1' in line:
				tmp_x[cnt] = val
			if 'Y1' in line:
				tmp_y[cnt] = val
			if 'Z1' in line:
				tmp_z[cnt] = val
			if 'S1' in line:
				tmp_s[cnt] = val

			if MCPN_ in line:
				MCPN.append(val)
				cnt = 0
			elif MSN_ in line:
				MSN.append(val)
				cnt = 0
			elif MCF_ in line:
				MCF.append(val)
				cnt = 0
			else:
				cnt += 1

			if 'SystemInertia' in line:
				S1.append(tmp_s)
				cnt = 0
			elif 'OutputPower' in line:
				if 'X1' in line:
					X1.append(tmp_x[:11])
				elif 'Y1' in line:
					Y1.append(tmp_y[:11])
				elif 'Z1' in line:
					Z1.append(tmp_z[:10])
				if 'S1' not in line:
					cnt = 0

	np.savetxt('X1.txt',np.asarray(X1,dtype='f'))
	np.savetxt('Y1.txt',np.asarray(Y1,dtype='f'))
	np.savetxt('Z1.txt',np.asarray(Z1,dtype='f'))
	np.savetxt('S1.txt',np.asarray(S1,dtype='f'))
	
if __name__ == '__main__':
	if (len(sys.argv) < 2):
		print("not enough input argument!")
		
	data_processing(sys.argv[1])
	



