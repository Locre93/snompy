from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from scipy.optimize import curve_fit

import os, csv, scipy.io, re, struct, h5py
import matplotlib.pyplot as plt
import numpy as np

__version__ = "0.3.0"
__authors__ = ["Lorenzo Orsini","Matteo Ceccanti"]

# NOTES
# The variable k is the light wavenumber (1/λ) in cm⁻¹

# REFERENCES

# ----------------------------------------------------------------------------------------------------- #
#                                    Functions and class description                                    #
# ----------------------------------------------------------------------------------------------------- #

# ----------------------------------------------------------------------------------------------------- #

def load_folder(root,i):
	measurements = [os.listdir(root)[i] for i in range(len(os.listdir(root))) if not re.compile(r".*\.png").match(os.listdir(root)[i])]
	return root + "\\" + measurements[i]

def load_gsf(file_name):
	with open(file_name,'rb') as gsf_file:
		gsf_data = gsf_file.read()

		i = 26
		while gsf_data[i:i+1] != b'\n':
			i = i + 1

		X_res = int(re.findall(r'\d+', str(gsf_data[26:i]))[0])

		i = i + 1
		j = i

		while gsf_data[i:i+1] != b'\n':
			i = i + 1

		Y_res = int(re.findall(r'\d+', str(gsf_data[j:i]))[0])

		last = len(gsf_data)
		first = last - X_res*Y_res*4

		data = np.empty(int((last-first)/4))
		i = 0
		for value in struct.iter_unpack('f',gsf_data[first:last]):
			data[i] = value[0]
			i=i+1

	return np.reshape(data,(Y_res,X_res))

def load_dump(file_name):
    with open(file_name,'rb') as dump_file:
        dump_data = dump_file.read()

    X_res = int(re.search(r'xres=(\d+)',str(dump_data)).group(1))
    Y_res = int(re.search(r'yres=(\d+)',str(dump_data)).group(1))

    X_span = float(re.search(r'xres=(\d+)',str(dump_data)).group(1))
    Y_span = float(re.search(r'yres=(\d+)',str(dump_data)).group(1))

    last = len(dump_data) - 3
    first = last - X_res*Y_res*8 

    data = np.empty(int((last-first)/8))
    i = 0
    for value in struct.iter_unpack('d',dump_data[first:last]):
        data[i] = value[0]
        i=i+1

    return np.reshape(data,(Y_res,X_res))

def load_scan(file_name):

	with open(file_name+'.npy', 'rb') as file:
		X = np.load(file)
		Y = np.load(file)
		Z = np.load(file)

	return X,Y,Z

def load_scan_HDF5(file_name):

	file = h5py.File(file_name + ".hdf5", "r")

	X = file.get("x")[:]
	Y = file.get("y")[:]
	Z = file.get("map")[:]

	file.close()

	return X,Y,Z

def save_scan(X,Y,Z,file_name):

	with open(file_name + '.npy', 'wb') as file:
		np.save(file,X)
		np.save(file,Y)
		np.save(file,Z)

def save_scan_HDF5(X,Y,Z,file_name):

	file = h5py.File(file_name + ".hdf5", "w")
	file.create_dataset("x",data=X)
	file.create_dataset("y",data=Y)
	file.create_dataset("map",data=Z)
	file.close()

def find_x(y,p1,p2):

	m = (p2[1]-p1[1])/(p2[0]-p1[0])
	q = p2[1] - m*p2[0]

	x = int((y-q)/m)

	if x < 0:
		return 0
	else:
		return int(np.floor(x))

def complex_fit(fitfun,x,y,**kwargs):
	d_real=np.real(y)
	d_imag=np.imag(y)
	yBoth = np.hstack([d_real, d_imag])
	n_el=fitfun.__code__.co_argcount-1

	if n_el==1:
		def funcBoth(x,a):
			N = len(x)
			x_real = x[:N//2]
			x_imag = x[N//2:]
			y_real = np.real(fitfun(x_real,a))
			y_imag = np.imag(fitfun(x_imag,a))
			return np.hstack([y_real, y_imag])
	elif n_el==2:
		def funcBoth(x,a,b):
			N = len(x)
			x_real = x[:N//2]
			x_imag = x[N//2:]
			y_real = np.real(fitfun(x_real,a,b))
			y_imag = np.imag(fitfun(x_imag,a,b))
			return np.hstack([y_real, y_imag])
	elif n_el==3:
		def funcBoth(x,a,b,c):
			N = len(x)
			x_real = x[:N//2]
			x_imag = x[N//2:]
			y_real = np.real(fitfun(x_real,a,b,c))
			y_imag = np.imag(fitfun(x_imag,a,b,c))
			return np.hstack([y_real, y_imag])
	elif n_el==4:
		def funcBoth(x,a,b,c,d):
			N = len(x)
			x_real = x[:N//2]
			x_imag = x[N//2:]
			y_real = np.real(fitfun(x_real,a,b,c,d))
			y_imag = np.imag(fitfun(x_imag,a,b,c,d))
			return np.hstack([y_real, y_imag])
	elif n_el==5:
		def funcBoth(x,a,b,c,d,e):
			N = len(x)
			x_real = x[:N//2]
			x_imag = x[N//2:]
			y_real = np.real(fitfun(x_real,a,b,c,d,e))
			y_imag = np.imag(fitfun(x_imag,a,b,c,d,e))
			return np.hstack([y_real, y_imag])
	elif n_el==6:
		def funcBoth(x,a,b,c,d,e,f):
			N = len(x)
			x_real = x[:N//2]
			x_imag = x[N//2:]
			y_real = np.real(fitfun(x_real,a,b,c,d,e,f))
			y_imag = np.imag(fitfun(x_imag,a,b,c,d,e,f))
			return np.hstack([y_real, y_imag])
	elif n_el==7:
		def funcBoth(x,a,b,c,d,e,f,g):
			N = len(x)
			x_real = x[:N//2]
			x_imag = x[N//2:]
			y_real = np.real(fitfun(x_real,a,b,c,d,e,f,g))
			y_imag = np.imag(fitfun(x_imag,a,b,c,d,e,f,g))
			return np.hstack([y_real, y_imag])
	elif n_el==8:
		def funcBoth(x,a,b,c,d,e,f,g,h):
			N = len(x)
			x_real = x[:N//2]
			x_imag = x[N//2:]
			y_real = np.real(fitfun(x_real,a,b,c,d,e,f,g,h))
			y_imag = np.imag(fitfun(x_imag,a,b,c,d,e,f,g,h))
			return np.hstack([y_real, y_imag])
	else:
		raise TypeError("Sei stronzo se fitti con più di otto parametri una funzione complessa, diocane pensaci un attimo magari riesci a toglierne qualcuno, altrimenti chiama Matteo")

	poptBoth, pcovBoth = curve_fit(funcBoth, np.hstack([x, x]), yBoth,**kwargs)

	return poptBoth, pcovBoth

def new_map(Colors,N):
	MAP=np.zeros([np.sum(N),3])
	column1=np.linspace(Colors[0,0],Colors[1,0],N[0]);

	column2=np.linspace(Colors[0,1],Colors[1,1],N[0]);
	column3=np.linspace(Colors[0,2],Colors[1,2],N[0]);

	for i in range(1,len(N)):
		column1=np.concatenate((column1,np.linspace(Colors[i,0],Colors[i+1,0],N[i])));
		column2=np.concatenate((column2,np.linspace(Colors[i,1],Colors[i+1,1],N[i])));
		column3=np.concatenate((column3,np.linspace(Colors[i,2],Colors[i+1,2],N[i])));

	MAP[:,0]=column1/255
	MAP[:,1]=column2/255
	MAP[:,2]=column3/255
	return ListedColormap(CMAP);

def Lorentz(k,BG1,BG2,kT,epsIx,Gx):
	return BG1+1j*BG2 + epsIx/(kT**2 - k**2 - 1j*k*Gx)

def Gauss(x,A,B,mu,sigma):
	return A*np.exp(-0.5*np.abs((x-mu)/sigma)**2) + B

def Fringes(x,A,B,C,q,k,phi):
	return A*np.sin(2*q*x + phi)*np.exp(-2*k*x)/(np.sqrt(x)) + B*np.sin(q*x + phi)*np.exp(-k*x)/(np.sqrt(x**3)) + C

# --------------------------------------------- CLASS SNOM -------------------------------------------- #

class snom():

	# Notation: The wavenumber has to be the last integer number written in the name of the scan

	def __init__(self,path):
		self.path = path
		self.folder = os.path.split(path)[-1]
		self.date = re.findall(r'\d+-\d+-\d+',self.folder)

		with open(self.path + "\\" + self.folder + ".txt",'r') as txt_file:
			lines = txt_file.readlines()

		self.x_max,self.y_max,_ = re.findall(r"[-+]?(?:\d*\.*\d+)",lines[7])
		self.Nx,self.Ny,_ = re.findall(r'\d+',lines[8])

		if not(float(self.y_max) == 0):
			self.type = "Spatial"
			self.y_min = 0
		else:
			sweep = re.findall(r"[-+]?(?:\d*\.*\d+to[-+]*\d*\.*\d+)",self.folder)
			y_min,y_max = sweep[0].split("to")

			self.y_min = float(y_min)
			self.y_max = float(y_max)

			if self.y_min > 600:
				self.type = "Frequency Sweep"
				self.wavelength = None
			else:
				self.type = "Voltage Sweep"
				self.voltage = None

		if self.type == "Spatial":
			self.wavelength = re.findall(r"[-+]?(?:\d*\.*\d+)",lines[13])
			if len(self.wavelength) == 0:
				self.wavelength = int(re.findall(r'\d+',self.folder)[-1])
			else:
				self.wavelength = round(10000/float(self.wavelength[0]),3)

			self.voltage = re.findall(r"[-+]?(?:\d*\.*\d+)V",self.folder)
			if len(self.voltage) == 0:
				self.voltage = None
			else:
				self.voltage = float(self.voltage[0][0:-1])

		elif self.type == "Frequency Sweep":
			self.voltage = re.findall(r"[-+]?(?:\d*\.*\d+)V",self.folder)
			if len(self.voltage) == 0:
				self.voltage = None
			else:
				self.voltage = float(self.voltage[0][0:-1])

		elif self.type == "Voltage Sweep":
			self.wavelength = re.findall(r"[-+]?(?:\d*\.*\d+)",lines[13])
			if len(self.wavelength) == 0:
				self.wavelength = int(re.findall(r'\d+',self.folder)[-1])
			else:
				self.wavelength = 10000/float(self.wavelength[0][0:-1])

		# Detection scheme used
		regexp = re.compile(r'2D \(PsHet\)')
		if regexp.search(lines[1]):
			self.scheme = "Pseudo-heterodyne"
		else:
			self.scheme = "Homodyne"

		# Default initialization
		self.channel_name = 'O4'
		self.map = load_gsf(self.path + "\\" + self.folder + " R-O4A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-O4P raw.gsf"))
	
		self.x = np.linspace(0,float(self.x_max),num=int(self.Nx))
		self.y = np.linspace(float(self.y_min),float(self.y_max),num=int(self.Ny))
		self.X,self.Y = np.meshgrid(self.x,self.y)

		self.sections = []

		self.fft_flag = False
		self.plot_flag = False

		self.fig = None
		self.axs = None

	def channel(self,channel_name="O4",direction="backward"):

		self.channel_name = channel_name

		if direction == "backward":
			if channel_name == "O0":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-O0A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-O0P raw.gsf"))

			elif channel_name == "O1":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-O1A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-O1P raw.gsf"))

			elif channel_name == "O2":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-O2A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-O2P raw.gsf"))

			elif channel_name == "O3":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-O3A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-O3P raw.gsf"))

			elif channel_name == "O4":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-O4A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-O4P raw.gsf"))

			elif channel_name == "O5":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-O5A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-O5P raw.gsf"))

			elif channel_name == "M0":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-M0A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-M0P raw.gsf"))

			elif channel_name == "M1":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-M1A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-M1P raw.gsf"))

			elif channel_name == "M2":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-M2A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-M2P raw.gsf"))

			elif channel_name == "M3":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-M3A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-M3P raw.gsf"))

			elif channel_name == "M4":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-M4A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-M4P raw.gsf"))

			elif channel_name == "M5":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-M5A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-M5P raw.gsf"))

			elif channel_name == "A0":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-A0A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-A0P raw.gsf"))

			elif channel_name == "A1":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-A1A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-A1P raw.gsf"))

			elif channel_name == "A2":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-A2A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-A2P raw.gsf"))

			elif channel_name == "A3":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-A3A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-A3P raw.gsf"))

			elif channel_name == "A4":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-A4A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-A4P raw.gsf"))

			elif channel_name == "A5":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-A5A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-A5P raw.gsf"))

			elif channel_name == "B0":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-B0A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-B0P raw.gsf"))

			elif channel_name == "B1":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-B1A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-B1P raw.gsf"))

			elif channel_name == "B2":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-B2A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-B2P raw.gsf"))

			elif channel_name == "B3":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-B3A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-B3P raw.gsf"))

			elif channel_name == "B4":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-B4A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-B4P raw.gsf"))

			elif channel_name == "B5":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-B5A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-B5P raw.gsf"))

			elif channel_name == "Z":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-Z raw.gsf")

			elif channel_name == "ZC":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-Z C.gsf")

			elif channel_name == "E":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-EA raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " R-EP raw.gsf"))

			elif channel_name == "M":
				self.map = load_gsf(self.path + "\\" + self.folder + " R-M raw.gsf")
		else:
			if channel_name == "O0":
				self.map = load_gsf(self.path + "\\" + self.folder + " O0A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " O0P raw.gsf"))

			elif channel_name == "O1":
				self.map = load_gsf(self.path + "\\" + self.folder + " O1A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " O1P raw.gsf"))

			elif channel_name == "O2":
				self.map = load_gsf(self.path + "\\" + self.folder + " O2A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " O2P raw.gsf"))

			elif channel_name == "O3":
				self.map = load_gsf(self.path + "\\" + self.folder + " O3A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " O3P raw.gsf"))

			elif channel_name == "O4":
				self.map = load_gsf(self.path + "\\" + self.folder + " O4A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " O4P raw.gsf"))

			elif channel_name == "O5":
				self.map = load_gsf(self.path + "\\" + self.folder + " O5A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " O5P raw.gsf"))

			elif channel_name == "M0":
				self.map = load_gsf(self.path + "\\" + self.folder + " M0A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " M0P raw.gsf"))

			elif channel_name == "M1":
				self.map = load_gsf(self.path + "\\" + self.folder + " M1A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " M1P raw.gsf"))

			elif channel_name == "M2":
				self.map = load_gsf(self.path + "\\" + self.folder + " M2A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " M2P raw.gsf"))

			elif channel_name == "M3":
				self.map = load_gsf(self.path + "\\" + self.folder + " M3A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " M3P raw.gsf"))

			elif channel_name == "M4":
				self.map = load_gsf(self.path + "\\" + self.folder + " M4A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " M4P raw.gsf"))

			elif channel_name == "M5":
				self.map = load_gsf(self.path + "\\" + self.folder + " M5A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " M5P raw.gsf"))

			elif channel_name == "A0":
				self.map = load_gsf(self.path + "\\" + self.folder + " A0A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " A0P raw.gsf"))

			elif channel_name == "A1":
				self.map = load_gsf(self.path + "\\" + self.folder + " A1A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " A1P raw.gsf"))

			elif channel_name == "A2":
				self.map = load_gsf(self.path + "\\" + self.folder + " A2A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " A2P raw.gsf"))

			elif channel_name == "A3":
				self.map = load_gsf(self.path + "\\" + self.folder + " A3A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " A3P raw.gsf"))

			elif channel_name == "A4":
				self.map = load_gsf(self.path + "\\" + self.folder + " A4A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " A4P raw.gsf"))

			elif channel_name == "A5":
				self.map = load_gsf(self.path + "\\" + self.folder + " A5A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " A5P raw.gsf"))

			elif channel_name == "B0":
				self.map = load_gsf(self.path + "\\" + self.folder + " B0A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " B0P raw.gsf"))

			elif channel_name == "B1":
				self.map = load_gsf(self.path + "\\" + self.folder + " B1A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " B1P raw.gsf"))

			elif channel_name == "B2":
				self.map = load_gsf(self.path + "\\" + self.folder + " B2A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " B2P raw.gsf"))

			elif channel_name == "B3":
				self.map = load_gsf(self.path + "\\" + self.folder + " B3A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " B3P raw.gsf"))

			elif channel_name == "B4":
				self.map = load_gsf(self.path + "\\" + self.folder + " B4A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " B4P raw.gsf"))

			elif channel_name == "B5":
				self.map = load_gsf(self.path + "\\" + self.folder + " B5A raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " B5P raw.gsf"))

			elif channel_name == "Z":
				self.map = load_gsf(self.path + "\\" + self.folder + " Z raw.gsf")

			elif channel_name == "ZC":
				self.map = load_gsf(self.path + "\\" + self.folder + " Z C.gsf")

			elif channel_name == "E":
				self.map = load_gsf(self.path + "\\" + self.folder + " EA raw.gsf")*np.exp(1j*load_gsf(self.path + "\\" + self.folder + " EP raw.gsf"))

			elif channel_name == "M":
				self.map = load_gsf(self.path + "\\" + self.folder + " M raw.gsf")

		self.x = np.linspace(0,float(self.x_max),num=int(self.Nx))
		self.y = np.linspace(float(self.y_min),float(self.y_max),num=int(self.Ny))
		self.X,self.Y = np.meshgrid(self.x,self.y)

		self.sections = []

		self.fft_flag = False
		self.plot_flag = False

		self.fig = None
		self.axs = None

		return self 

	def cut(self,x_range=[0,None],x_reset=True,y_range=[0,None],y_reset=False):

		self.map = self.map[y_range[0]:y_range[1],x_range[0]:x_range[1]]

		self.X = self.X[y_range[0]:y_range[1],x_range[0]:x_range[1]]
		self.Y = self.Y[y_range[0]:y_range[1],x_range[0]:x_range[1]]

		if x_reset and y_reset:
			self.X = self.X - self.x[x_range[0]]
			self.Y = self.Y - self.y[y_range[0]]

			self.x = self.x[x_range[0]:x_range[1]] - self.x[x_range[0]]
			self.y = self.y[y_range[0]:y_range[1]] - self.y[y_range[0]]

		elif x_reset and not(y_reset):
			self.X = self.X - self.x[x_range[0]]
			self.x = self.x[x_range[0]:x_range[1]] - self.x[x_range[0]]

		elif not(x_reset) and y_reset:
			self.Y = self.Y - self.y[y_range[0]]
			self.y = self.y[y_range[0]:y_range[1]] - self.y[y_range[0]]

		else:
			self.x = self.x[x_range[0]:x_range[1]]
			self.y = self.y[y_range[0]:y_range[1]]

		return self

	def plot(self,fun='abs',cres=200,cmap='viridis',vmin=None,vmax=None,xlim=None,ylim=None,figsize=(8,6),save=False,show=True,pixel=False,colorbar=True):
		
		if pixel:

			if type(fun) is list:

				self.fig, self.axs = plt.subplots(1,2,figsize=(10,5.5))
				self.fig.suptitle(self.folder + "  " + self.channel_name)

				if fun[0] == 'abs' and fun[1] == 'phase':
					self.axs[0].contourf(np.abs(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.adjust_phase(),cres,cmap=cmap)
					self.axs[0].set_title('Absolute Value')	
					self.axs[1].set_title('Phase')
				elif fun[0] == 'abs' and fun[1] == 'real':
					self.axs[0].contourf(np.abs(self.map),cres,cmap=cmap)
					self.axs[1].contourf(np.real(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Absolute Value')	
					self.axs[1].set_title('Real Part')
				elif fun[0] == 'abs' and fun[1] == 'imag':
					self.axs[0].contourf(np.abs(self.map),cres,cmap=cmap)
					self.axs[1].contourf(np.imag(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Absolute Value')	
					self.axs[1].set_title('Imaginary Part')
				elif fun[0] == 'phase' and fun[1] == 'abs':
					self.axs[0].contourf(self.adjust_phase(),cres,cmap=cmap)
					self.axs[1].contourf(np.abs(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Phase')	
					self.axs[1].set_title('Absolute Value')
				elif fun[0] == 'phase' and fun[1] == 'real':
					self.axs[0].contourf(self.adjust_phase(),cres,cmap=cmap)
					self.axs[1].contourf(np.real(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Phase')	
					self.axs[1].set_title('Real Part')
				elif fun[0] == 'phase' and fun[1] == 'imag':
					self.axs[0].contourf(self.adjust_phase(),cres,cmap=cmap)
					self.axs[1].contourf(np.imag(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Phase')	
					self.axs[1].set_title('Imaginary Part')
				elif fun[0] == 'real' and fun[1] == 'abs':
					self.axs[0].contourf(np.real(self.map),cres,cmap=cmap)
					self.axs[1].contourf(np.abs(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Real Part')	
					self.axs[1].set_title('Absolute Value')
				elif fun[0] == 'real' and fun[1] == 'phase':
					self.axs[0].contourf(np.real(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.adjust_phase(),cres,cmap=cmap)
					self.axs[0].set_title('Real Part')	
					self.axs[1].set_title('Phase')
				elif fun[0] == 'real' and fun[1] == 'imag':
					self.axs[0].contourf(np.real(self.map),cres,cmap=cmap)
					self.axs[1].contourf(np.imag(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Real Part')	
					self.axs[1].set_title('Imaginary Part')
				elif fun[0] == 'imag' and fun[1] == 'abs':
					self.axs[0].contourf(np.imag(self.map),cres,cmap=cmap)
					self.axs[1].contourf(np.abs(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Imaginary Part')	
					self.axs[1].set_title('Absolute Value')
				elif fun[0] == 'imag' and fun[1] == 'phase':
					self.axs[0].contourf(np.imag(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.adjust_phase(),cres,cmap=cmap)
					self.axs[0].set_title('Imaginary Part')	
					self.axs[1].set_title('Phase')
				elif fun[0] == 'imag' and fun[1] == 'real':
					self.axs[0].contourf(np.imag(self.map),cres,cmap=cmap)
					self.axs[1].contourf(np.real(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Imaginary Part')	
					self.axs[1].set_title('Real Part')
				
				self.axs[0].set_ylabel(ylabel='Slow axis, pixel',fontsize=18)
				self.axs[0].set_xlabel(xlabel='Fast axis, pixel',fontsize=18)
				self.axs[0].tick_params(axis='both',which='major',labelsize=16)

				self.axs[1].set_xlabel(xlabel='Fast axis, pixel',fontsize=18)
				self.axs[1].tick_params(axis='both',which='major',labelsize=16)

			else:
				if fun == 'abs':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(np.abs(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)
					self.fig = plt.xlabel('Fast axis, pixel',fontsize=18)
					self.fig = plt.ylabel('Slow axis, pixel',fontsize=18)

					if colorbar:
						plt.colorbar()

				elif fun == 'phase':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(self.adjust_phase(),cres,cmap=cmap,vmin=vmin,vmax=vmax)
					self.fig = plt.xlabel('Fast axis, pixel',fontsize=18)
					self.fig = plt.ylabel('Slow axis, pixel',fontsize=18)

					if colorbar:
						plt.colorbar()

				elif fun == 'real':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(np.real(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)
					self.fig = plt.xlabel('Fast axis, pixel',fontsize=18)
					self.fig = plt.ylabel('Slow axis, pixel',fontsize=18)

					if colorbar:
						plt.colorbar()

				elif fun == 'imag':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(np.imag(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)
					self.fig = plt.xlabel('Fast axis, pixel',fontsize=18)
					self.fig = plt.ylabel('Slow axis, pixel',fontsize=18)

					if colorbar:
						plt.colorbar()

				elif fun == 'all':
					self.fig, self.axs = plt.subplots(2,2,figsize=(10,8))

					self.fig.suptitle(self.folder + "  " + self.channel_name)

					self.axs[0, 0].set_title('Absolute Value')
					self.axs[0, 0].contourf(np.abs(self.map),cres,cmap=cmap)
					self.axs[0, 0].set_ylabel(ylabel='Slow axis, pixel',fontsize=16)

					self.axs[0, 1].set_title('Phase')
					self.axs[0, 1].contourf(self.adjust_phase(),cres,cmap=cmap)

					self.axs[1, 0].set_title('Real Part')
					self.axs[1, 0].contourf(np.real(self.map),cres,cmap=cmap)
					self.axs[1, 0].set_ylabel(ylabel='Slow axis, pixel',fontsize=16)
					self.axs[1, 0].set_xlabel(xlabel='Fast axis, pixel',fontsize=16)

					self.axs[1, 1].set_title('Imaginary Part')
					self.axs[1, 1].contourf(np.imag(self.map),cres,cmap=cmap)
					self.axs[1, 1].set_xlabel(xlabel='Fast axis, pixel',fontsize=16)
			
		else:

			if type(fun) is list:

				self.fig, self.axs = plt.subplots(1,2,figsize=(10,5.5))
				self.fig.suptitle(self.folder + "  " + self.channel_name)

				if fun[0] == 'abs' and fun[1] == 'phase':
					self.axs[0].contourf(self.X,self.Y,np.abs(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,self.adjust_phase(),cres,cmap=cmap)
					self.axs[0].set_title('Absolute Value')	
					self.axs[1].set_title('Phase')
				elif fun[0] == 'abs' and fun[1] == 'real':
					self.axs[0].contourf(self.X,self.Y,np.abs(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,np.real(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Absolute Value')	
					self.axs[1].set_title('Real Part')
				elif fun[0] == 'abs' and fun[1] == 'imag':
					self.axs[0].contourf(self.X,self.Y,np.abs(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,np.imag(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Absolute Value')	
					self.axs[1].set_title('Imaginary Part')
				elif fun[0] == 'phase' and fun[1] == 'abs':
					self.axs[0].contourf(self.X,self.Y,self.adjust_phase(),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,np.abs(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Phase')	
					self.axs[1].set_title('Absolute Value')
				elif fun[0] == 'phase' and fun[1] == 'real':
					self.axs[0].contourf(self.X,self.Y,self.adjust_phase(),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,np.real(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Phase')	
					self.axs[1].set_title('Real Part')
				elif fun[0] == 'phase' and fun[1] == 'imag':
					self.axs[0].contourf(self.X,self.Y,self.adjust_phase(),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,np.imag(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Phase')	
					self.axs[1].set_title('Imaginary Part')
				elif fun[0] == 'real' and fun[1] == 'abs':
					self.axs[0].contourf(self.X,self.Y,np.real(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,np.abs(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Real Part')	
					self.axs[1].set_title('Absolute Value')
				elif fun[0] == 'real' and fun[1] == 'phase':
					self.axs[0].contourf(self.X,self.Y,np.real(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,self.adjust_phase(),cres,cmap=cmap)
					self.axs[0].set_title('Real Part')	
					self.axs[1].set_title('Phase')
				elif fun[0] == 'real' and fun[1] == 'imag':
					self.axs[0].contourf(self.X,self.Y,np.real(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,np.imag(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Real Part')	
					self.axs[1].set_title('Imaginary Part')
				elif fun[0] == 'imag' and fun[1] == 'abs':
					self.axs[0].contourf(self.X,self.Y,np.imag(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,np.abs(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Imaginary Part')	
					self.axs[1].set_title('Absolute Value')
				elif fun[0] == 'imag' and fun[1] == 'phase':
					self.axs[0].contourf(self.X,self.Y,np.imag(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,self.adjust_phase(),cres,cmap=cmap)
					self.axs[0].set_title('Imaginary Part')	
					self.axs[1].set_title('Phase')
				elif fun[0] == 'imag' and fun[1] == 'real':
					self.axs[0].contourf(self.X,self.Y,np.imag(self.map),cres,cmap=cmap)
					self.axs[1].contourf(self.X,self.Y,np.real(self.map),cres,cmap=cmap)
					self.axs[0].set_title('Imaginary Part')	
					self.axs[1].set_title('Real Part')


				if self.type == "Spatial":
					self.axs[0].set_ylabel(ylabel='Y, μm',fontsize=18)

				elif self.type == "Voltage Sweep":
					self.axs[0].set_ylabel(ylabel='Voltage, V',fontsize=18)

				elif self.type == "Frequency Sweep":
					self.axs[0].set_ylabel(ylabel='Wavenumber, cm⁻¹',fontsize=18)

				if self.fft_flag:
					self.axs[0].set_xlabel(xlabel='Q, x10⁴ cm⁻¹',fontsize=18)
					self.axs[1].set_xlabel(xlabel='Q, x10⁴ cm⁻¹',fontsize=18)
				else:
					self.axs[0].set_xlabel(xlabel='X, μm',fontsize=18)
					self.axs[1].set_xlabel(xlabel='X, μm',fontsize=18)

				self.axs[0].tick_params(axis='both',which='major',labelsize=16)
				self.axs[1].tick_params(axis='both',which='major',labelsize=16)

			else:
				if fun == 'abs':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(self.X,self.Y,np.abs(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)

					if colorbar:
						plt.colorbar()

				elif fun == 'phase':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(self.X,self.Y,self.adjust_phase(),cres,cmap=cmap,vmin=vmin,vmax=vmax)

					if colorbar:
						plt.colorbar()

				elif fun == 'real':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(self.X,self.Y,np.real(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)

					if colorbar:
						plt.colorbar()

				elif fun == 'imag':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(self.X,self.Y,np.imag(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)

					if colorbar:
						plt.colorbar()			

				elif fun == 'all':
					self.fig, self.axs = plt.subplots(2,2,figsize=(10,8))

					self.fig.suptitle(self.folder + "  " + self.channel_name)

					self.axs[0, 0].set_title('Absolute value')
					self.axs[0, 0].contourf(self.X,self.Y,np.abs(self.map),cres,cmap=cmap)

					self.axs[0, 1].set_title('Phase')
					self.axs[0, 1].contourf(self.X,self.Y,self.adjust_phase(),cres,cmap=cmap)

					self.axs[1, 0].set_title('Real part')
					self.axs[1, 0].contourf(self.X,self.Y,np.real(self.map),cres,cmap=cmap)

					self.axs[1, 1].set_title('Imaginary part')
					self.axs[1, 1].contourf(self.X,self.Y,np.imag(self.map),cres,cmap=cmap)
					
					if self.type == "Spatial":
						self.axs[0, 0].set(ylabel='Y, μm')
						self.axs[1, 0].set(ylabel='Y, μm')

					elif self.type == "Voltage Sweep":
						self.axs[0, 0].set(ylabel='Voltage, V')
						self.axs[1, 0].set(ylabel='Voltage, V')

					elif self.type == "Frequency Sweep":
						self.axs[0, 0].set(ylabel='Wavenumber, cm⁻¹')
						self.axs[1, 0].set(ylabel='Wavenumber, cm⁻¹')

					if self.fft_flag:
						self.axs[1, 0].set(xlabel='Q, x10⁴ cm⁻¹')
						self.axs[1, 1].set(xlabel='Q, x10⁴ cm⁻¹')
					else:
						self.axs[1, 0].set(xlabel='X, μm')
						self.axs[1, 1].set(xlabel='X, μm')

				if not fun == 'all':
					self.fig = plt.tick_params(axis='both',which='major',labelsize=16)
					self.fig = plt.tick_params(axis='both',which='minor',labelsize=16)
					self.fig = plt.title(self.folder + "  " + self.channel_name + "  " + fun)
					self.fig = plt.xlim(xlim)
					self.fig = plt.ylim(ylim)

					if self.type == "Spatial":
						self.fig = plt.ylabel('Y, μm',fontsize=18)
					elif self.type == "Voltage Sweep":
						self.fig = plt.ylabel('Voltage, V',fontsize=18)
					elif self.type == "Frequency Sweep":
						self.fig = plt.ylabel('Wavenumber, cm⁻¹',fontsize=18)

					if self.fft_flag:
						self.fig = plt.xlabel('Q, x10⁴ cm⁻¹',fontsize=18)	# Check the units
					else:
						self.fig = plt.xlabel('X, μm',fontsize=18)

		if save:
			Tk().withdraw()
			file_name = asksaveasfilename(filetypes=[("Portable Network Graphic", ".png")], defaultextension=".png")
			self.fig = plt.savefig(file_name,dpi=192,transparent=True,bbox_inches='tight')

		if show:
			self.fig = plt.show()

		return self	

	def fft(self):

		FFT_resolution = 1/(self.x[1] - self.x[0])
		q_max = FFT_resolution/2

		self.map = np.fft.fft(np.abs(self.map), axis = 1)
		self.map = abs(np.fft.fftshift(self.map,axes=1))

		self.x = np.linspace(-q_max,q_max,np.shape(self.map)[1])
		self.X,_ = np.meshgrid(self.x,self.y)

		self.fft_flag = True

		return self

	def normalize(self,pixel=-10,data=None):
		if type(data) == type(None):
			self.map = self.map/np.repeat(np.expand_dims(self.map[:,pixel],axis=1),self.map.shape[1],axis=1)
		else:
			self.map = self.map/np.repeat(np.expand_dims(data,axis=1),self.map.shape[1],axis=1)
		return self

	def adjust_phase(self):

		PHASE_MAP = np.angle(self.map)
		theta = np.angle(self.map)

		for i in range(len(self.y)):
			for j in range(0,len(self.x)-1):

				theta = (PHASE_MAP[i,j] - PHASE_MAP[i,j+1])

				if theta > 5:
					PHASE_MAP[i,j+1] = PHASE_MAP[i,j+1] + 2*np.pi
				elif theta < -5:
					PHASE_MAP[i,j+1] = PHASE_MAP[i,j+1] - 2*np.pi

		return PHASE_MAP

	def filter_std(self,threshold=None,pixel=range(10,40)):

		std = np.std(np.abs(self.map[:,pixel]),axis=1)

		if threshold == None:
			threshold = np.mean(std)*2
			print("threshold = " + str(threshold))

		for i in range(-len(self.y),0):
			if std[i] > threshold:
				self.map = np.delete(self.map,i,0)
				self.X = np.delete(self.X,i,0)
				self.Y = np.delete(self.Y,i,0)
				self.y = np.delete(self.y,i,0)

		return self

	def filter(self,threshold,pixel=range(10,40),fun='abs'):

		if fun == 'abs':
			mean = np.mean(np.abs(self.map[:,pixel]),axis=1)
		elif fun=='phase':
			mean = np.mean(self.adjust_phase()[:,pixel],axis=1)
		elif fun == 'real':
			mean = np.mean(np.real(self.map[:,pixel]),axis=1)
		elif fun == 'imag':
			mean = np.mean(np.imag(self.map[:,pixel]),axis=1)

		for i in range(-len(self.y),0):
			if mean[i] > threshold[0]:
				self.map = np.delete(self.map,i,0)
				self.X = np.delete(self.X,i,0)
				self.Y = np.delete(self.Y,i,0)
				self.y = np.delete(self.y,i,0)

		return self

	def section(self,pixel,direction='Vertical',fun='abs',figsize=(8,6),xlim=None,ylim=None,plot=False,save=False,show=True,plot_type="plot",s=25):

		if direction == 'Vertical':
			self.sections.append([self.y,self.map[:,pixel][:]])
		elif direction == 'Horizontal':
			self.sections.append([self.x,self.map[pixel,:][:]])

		if plot == True:
			if plot_type == 'plot':
				if not self.plot_flag:
					self.fig = plt.figure(figsize=figsize)
					self.plot_flag = True

				if fun == 'abs':
					self.fig = plt.plot(self.sections[-1][0],np.abs(self.sections[-1][1]))
				elif fun == 'phase':
					self.fig = plt.plot(self.sections[-1][0],np.angle(self.sections[-1][1]))
				elif fun == 'real':
					self.fig = plt.plot(self.sections[-1][0],np.real(self.sections[-1][1]))
				elif fun == 'imag':
					self.fig = plt.plot(self.sections[-1][0],np.imag(self.sections[-1][1]))

				if direction == 'Vertical':
					if self.type == "Spatial":
						self.fig = plt.xlabel('Y, μm',fontsize=18)
					elif self.type == "Voltage Sweep":
						self.fig = plt.xlabel('Voltage, V',fontsize=18)
					elif self.type == "Frequency Sweep":
						self.fig = plt.xlabel('Wavenumber, cm⁻¹',fontsize=18)

				elif direction == 'Horizontal':
					if self.fft_flag:
						self.fig = plt.xlabel('Q, x10⁴ cm⁻¹',fontsize=18)	# Check the units
					else:
						self.fig = plt.xlabel('X, μm',fontsize=18)

				self.fig = plt.tick_params(axis='both',which='major',labelsize=16)
				self.fig = plt.tick_params(axis='both',which='minor',labelsize=16)
				self.fig = plt.xlim(xlim)
				self.fig = plt.ylim(ylim)

				self.fig = plt.ylabel(self.channel_name + "  " + fun,fontsize=18)

			elif plot_type == 'scatter':

				if not self.plot_flag:
					self.fig = plt.figure(figsize=figsize)
					self.plot_flag = True

				if fun == 'abs':
					self.fig = plt.scatter(self.sections[-1][0],np.abs(self.sections[-1][1]),s=s)
				elif fun == 'phase':
					self.fig = plt.scatter(self.sections[-1][0],np.angle(self.sections[-1][1]),s=s)
				elif fun == 'real':
					self.fig = plt.scatter(self.sections[-1][0],np.real(self.sections[-1][1]),s=s)
				elif fun == 'imag':
					self.fig = plt.scatter(self.sections[-1][0],np.imag(self.sections[-1][1]),s=s)

				if direction == 'Vertical':
					if self.type == "Spatial":
						self.fig = plt.xlabel('Y, μm',fontsize=18)
					elif self.type == "Voltage Sweep":
						self.fig = plt.xlabel('Voltage, V',fontsize=18)
					elif self.type == "Frequency Sweep":
						self.fig = plt.xlabel('Wavenumber, cm⁻¹',fontsize=18)

				elif direction == 'Horizontal':
					if self.fft_flag:
						self.fig = plt.xlabel('Q, x10⁴ cm⁻¹',fontsize=18)	# Check the units
					else:
						self.fig = plt.xlabel('X, μm',fontsize=18)

				self.fig = plt.tick_params(axis='both',which='major',labelsize=16)
				self.fig = plt.tick_params(axis='both',which='minor',labelsize=16)
				self.fig = plt.xlim(xlim)
				self.fig = plt.ylim(ylim)

				self.fig = plt.ylabel(self.channel_name + "  " + fun,fontsize=18)

			elif plot_type == 'plot_and_scatter':

				if not self.plot_flag:
					self.fig = plt.figure(figsize=figsize)
					self.plot_flag = True

				if fun == 'abs':
					self.fig = plt.scatter(self.sections[-1][0],np.abs(self.sections[-1][1]),s=s)
					self.fig = plt.plot(self.sections[-1][0],np.abs(self.sections[-1][1]))
				elif fun == 'phase':
					self.fig = plt.scatter(self.sections[-1][0],np.angle(self.sections[-1][1]),s=s)
					self.fig = plt.plot(self.sections[-1][0],np.angle(self.sections[-1][1]))
				elif fun == 'real':
					self.fig = plt.scatter(self.sections[-1][0],np.real(self.sections[-1][1]),s=s)
					self.fig = plt.plot(self.sections[-1][0],np.real(self.sections[-1][1]))
				elif fun == 'imag':
					self.fig = plt.scatter(self.sections[-1][0],np.imag(self.sections[-1][1]),s=s)
					self.fig = plt.plot(self.sections[-1][0],np.imag(self.sections[-1][1]))

				if direction == 'Vertical':
					if self.type == "Spatial":
						self.fig = plt.xlabel('Y, μm',fontsize=18)
					elif self.type == "Voltage Sweep":
						self.fig = plt.xlabel('Voltage, V',fontsize=18)
					elif self.type == "Frequency Sweep":
						self.fig = plt.xlabel('Wavenumber, cm⁻¹',fontsize=18)

				elif direction == 'Horizontal':
					if self.fft_flag:
						self.fig = plt.xlabel('Q, x10⁴ cm⁻¹',fontsize=18)	# Check the units
					else:
						self.fig = plt.xlabel('X, μm',fontsize=18)

				self.fig = plt.tick_params(axis='both',which='major',labelsize=16)
				self.fig = plt.tick_params(axis='both',which='minor',labelsize=16)
				self.fig = plt.xlim(xlim)
				self.fig = plt.ylim(ylim)

				self.fig = plt.ylabel(self.channel_name + "  " + fun,fontsize=18)

			if save:
				Tk().withdraw()
				file_name = asksaveasfilename(filetypes=[("Portable Network Graphic", ".png")], defaultextension=".png")
				self.fig = plt.savefig(file_name,dpi=192,transparent=True,bbox_inches='tight')

			if show:
				self.fig = plt.show()
				self.plot_flag = False

		return self

	def drift_corr(self,v):

		v = np.array(v)

		v[0,1] = 0
		v[-1,1] = len(self.y)

		x_min = np.amin(v[:,0])
		x_max = np.amax(v[:,0])

		aux_map = np.zeros(shape=(len(self.y),len(self.x)-x_max+x_min),dtype=complex)

		j = 0
		for i in range(len(self.y)):
			if i > v[j+1,1]:
				j=j+1

			if i == v[j+1,1]:
				x_edge = v[j+1,0]

			else:
				x_edge = find_x(i,v[j,:],v[j+1,:])

			
			aux_map[i,:] = self.map[i,x_edge-x_min:x_edge+len(self.x)-x_max]


		self.x = self.x[0:len(self.x)-x_max+x_min]

		self.map = aux_map
		self.X,self.Y = np.meshgrid(self.x,self.y)

		return self

	def print_details(self):
		print(self.folder)
		print(self.scheme)
		print(self.type)
		print(self.wavelength)
		print(self.voltage)

		return self

# --------------------------------------------- TEST CODE --------------------------------------------- #

if __name__ == '__main__':
	pass