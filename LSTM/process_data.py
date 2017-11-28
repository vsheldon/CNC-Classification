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
# (X,Y.txt has 12 col, Z.txt has 11 col, S.txt has 13 col)
# represents [TimeStamp, ActualPosition, ActualVelocity, 
#			  ActualAcceleration, CommandPosition
#             CommandVelocity, CommandAcceleration, 
#			  CurrentFeedback, DCBusVoltage,
#			  OutputCurrent, OutputVoltage,
#             OutputPower, SystemInertia]
# M1.txt has 4 col [timestamp, M1_CURRENT_PROGRAM_NUMBER, M1_sequence_number, M1_CURRENT_FEEDRATE]



def data_processing(directory):
	X1, Y1, Z1, S1, M1=[],[],[],[],[]
	MCPN_ = 'M1_CURRENT_PROGRAM_NUMBER'
	MSN_ = 'M1_sequence_number'
	MCF_ = 'M1_CURRENT_FEEDRATE'
	for file in glob.glob(directory+"/*.dat"):
		cnt = 1
		tmp_x = np.arange(13,dtype='f')
		tmp_y = np.arange(13,dtype='f')
		tmp_z = np.arange(13,dtype='f')
		tmp_s = np.arange(13,dtype='f')
		tmp_m = np.arange(4,dtype='f')

		for line in open(file,'r').read().split("},{"):

			tag = line[line.find("TagValue")+11:]
			val = float(tag[0:tag.find('"')])
			#print val
			time_start = line.find('Date(') + len('Data(')
			time_end = line.find('-',time_start)
			time = line[time_start:time_end]

			if 'X1' in line:
				tmp_x[0] = time
				tmp_x[cnt] = val
			if 'Y1' in line:
				tmp_y[0] = time
				tmp_y[cnt] = val
			if 'Z1' in line:
				tmp_z[0] = time
				tmp_z[cnt] = val
			if 'S1' in line:
				tmp_s[0] = time
				tmp_s[cnt] = val

			if MCPN_ in line:
				tmp_m[0] = time
				tmp_m[1] = val
				cnt = 1
			elif MSN_ in line:
				tmp_m[0] = time
				tmp_m[2] = val
				cnt = 1
			elif MCF_ in line:
				tmp_m[0] = time
				tmp_m[3] = val
				M1.append(tmp_m)
				cnt = 1
			else:
				cnt += 1

			if 'SystemInertia' in line:
				S1.append(tmp_s)
				cnt = 1
			elif 'OutputPower' in line:
				if 'X1' in line:
					X1.append(tmp_x[:12])
				elif 'Y1' in line:
					Y1.append(tmp_y[:12])
				elif 'Z1' in line:
					Z1.append(tmp_z[:11])
				if 'S1' not in line:
					cnt = 1

	np.savetxt('X1.txt',np.asarray(X1,dtype='f'))
	np.savetxt('Y1.txt',np.asarray(Y1,dtype='f'))
	np.savetxt('Z1.txt',np.asarray(Z1,dtype='f'))
	np.savetxt('S1.txt',np.asarray(S1,dtype='f'))
	np.savetxt('M1.txt',np.asarray(M1,dtype='f'))

if __name__ == '__main__':
	if (len(sys.argv) < 2):
		print("not enough input argument!")
	#data_processing('data')

	data_processing(sys.argv[1])
	



