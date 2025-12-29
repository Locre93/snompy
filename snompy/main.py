import os, re, struct

import numpy as np

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import rotate

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, LinearSegmentedColormap

from tkinter import Tk
from tkinter.filedialog import asksaveasfilename

__version__ = "1.0.0"
__authors__ = ["Lorenzo Orsini","Elisa Mendels","Matteo Ceccanti", "Bianca Turini"]

# NOTES
# The variable k is the light wavenumber (1/λ) in cm⁻¹

# REFERENCES

# ----------------------------------------------------------------------------------------------------- #
#                                    Functions and class description                                    #
# ----------------------------------------------------------------------------------------------------- #

# ---------------------------------------------- LOADING ---------------------------------------------- #

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

def load_folder(root, idx):
    # List all entries
    entries = os.listdir(root)
    # Sort them alphabetically to match Finder/ls order
    entries.sort()
    # Keep only directories (ignore files, including .png)
    measurements = [
        entry for entry in entries
        if os.path.isdir(os.path.join(root, entry)) and not re.search(r"\.png$", entry)
    ]
    # Return the idx-th folder
    return os.path.join(root, measurements[idx])

# ---------------------------------------------- ANALYSIS --------------------------------------------- #

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

def Lorentz(k,BG1,BG2,kT,epsIx,Gx):
	return BG1+1j*BG2 + epsIx/(kT**2 - k**2 - 1j*k*Gx)

def Gauss(x,A,B,mu,sigma):
	return A*np.exp(-0.5*np.abs((x-mu)/sigma)**2) + B

# --------------------------------------------- CLASS SNOM -------------------------------------------- #

class snom():

	# Notation: The wavenumber has to be the last integer number written in the name of the scan

	def __init__(self,path,flip=False):
		self.path = path
		self.flip = flip
		self.folder = os.path.split(path)[-1]
		self.date = re.findall(r'\d+-\d+-\d+',self.folder)

		with open(os.path.join(self.path, self.folder + ".txt"),'r', encoding="utf-8") as txt_file:
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
		self.map = load_gsf(os.path.join(self.path, self.folder + " R-O4A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-O4P raw.gsf")))
	
		self.x = np.linspace(0,float(self.x_max),num=int(self.Nx))
		self.y = np.linspace(float(self.y_min),float(self.y_max),num=int(self.Ny))
		self.X,self.Y = np.meshgrid(self.x,self.y)

		if self.type != "Frequency Sweep":
			if self.flip:
				self.map = np.flip(self.map,0)

		self.sections = []

		self.fft_flag = False
		self.plot_flag = False

		self.fig = None
		self.axs = None

	def channel(self,channel_name="O4",direction="backward"):

		self.channel_name = channel_name

		if direction == "backward":
			if channel_name == "O0":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-O0A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-O0P raw.gsf")))

			elif channel_name == "O1":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-O1A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-O1P raw.gsf")))

			elif channel_name == "O2":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-O2A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-O2P raw.gsf")))

			elif channel_name == "O3":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-O3A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-O3P raw.gsf")))

			elif channel_name == "O4":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-O4A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-O4P raw.gsf")))

			elif channel_name == "O5":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-O5A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-O5P raw.gsf")))

			elif channel_name == "M0":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-M0A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-M0P raw.gsf")))

			elif channel_name == "M1":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-M1A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-M1P raw.gsf")))

			elif channel_name == "M2":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-M2A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-M2P raw.gsf")))

			elif channel_name == "M3":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-M3A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-M3P raw.gsf")))

			elif channel_name == "M4":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-M4A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-M4P raw.gsf")))

			elif channel_name == "M5":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-M5A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-M5P raw.gsf")))

			elif channel_name == "A0":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-A0A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-A0P raw.gsf")))

			elif channel_name == "A1":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-A1A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-A1P raw.gsf")))

			elif channel_name == "A2":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-A2A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-A2P raw.gsf")))

			elif channel_name == "A3":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-A3A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-A3P raw.gsf")))

			elif channel_name == "A4":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-A4A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-A4P raw.gsf")))

			elif channel_name == "A5":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-A5A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-A5P raw.gsf")))

			elif channel_name == "B0":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-B0A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-B0P raw.gsf")))

			elif channel_name == "B1":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-B1A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-B1P raw.gsf")))

			elif channel_name == "B2":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-B2A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-B2P raw.gsf")))

			elif channel_name == "B3":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-B3A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-B3P raw.gsf")))

			elif channel_name == "B4":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-B4A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-B4P raw.gsf")))

			elif channel_name == "B5":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-B5A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-B5P raw.gsf")))

			elif channel_name == "Z":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-Z raw.gsf"))

			elif channel_name == "ZC":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-Z C.gsf"))

			elif channel_name == "E":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-EA raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " R-EP raw.gsf")))

			elif channel_name == "M":
				self.map = load_gsf(os.path.join(self.path, self.folder + " R-M raw.gsf"))
		else:
			if channel_name == "O0":
				self.map = load_gsf(os.path.join(self.path, self.folder + " O0A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " O0P raw.gsf")))

			elif channel_name == "O1":
				self.map = load_gsf(os.path.join(self.path, self.folder + " O1A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " O1P raw.gsf")))

			elif channel_name == "O2":
				self.map = load_gsf(os.path.join(self.path, self.folder + " O2A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " O2P raw.gsf")))

			elif channel_name == "O3":
				self.map = load_gsf(os.path.join(self.path, self.folder + " O3A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " O3P raw.gsf")))

			elif channel_name == "O4":
				self.map = load_gsf(os.path.join(self.path, self.folder + " O4A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " O4P raw.gsf")))

			elif channel_name == "O5":
				self.map = load_gsf(os.path.join(self.path, self.folder + " O5A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " O5P raw.gsf")))

			elif channel_name == "M0":
				self.map = load_gsf(os.path.join(self.path, self.folder + " M0A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " M0P raw.gsf")))

			elif channel_name == "M1":
				self.map = load_gsf(os.path.join(self.path, self.folder + " M1A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " M1P raw.gsf")))

			elif channel_name == "M2":
				self.map = load_gsf(os.path.join(self.path, self.folder + " M2A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " M2P raw.gsf")))

			elif channel_name == "M3":
				self.map = load_gsf(os.path.join(self.path, self.folder + " M3A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " M3P raw.gsf")))

			elif channel_name == "M4":
				self.map = load_gsf(os.path.join(self.path, self.folder + " M4A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " M4P raw.gsf")))

			elif channel_name == "M5":
				self.map = load_gsf(os.path.join(self.path, self.folder + " M5A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " M5P raw.gsf")))

			elif channel_name == "A0":
				self.map = load_gsf(os.path.join(self.path, self.folder + " A0A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " A0P raw.gsf")))

			elif channel_name == "A1":
				self.map = load_gsf(os.path.join(self.path, self.folder + " A1A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " A1P raw.gsf")))

			elif channel_name == "A2":
				self.map = load_gsf(os.path.join(self.path, self.folder + " A2A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " A2P raw.gsf")))

			elif channel_name == "A3":
				self.map = load_gsf(os.path.join(self.path, self.folder + " A3A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " A3P raw.gsf")))

			elif channel_name == "A4":
				self.map = load_gsf(os.path.join(self.path, self.folder + " A4A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " A4P raw.gsf")))

			elif channel_name == "A5":
				self.map = load_gsf(os.path.join(self.path, self.folder + " A5A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " A5P raw.gsf")))

			elif channel_name == "B0":
				self.map = load_gsf(os.path.join(self.path, self.folder + " B0A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " B0P raw.gsf")))

			elif channel_name == "B1":
				self.map = load_gsf(os.path.join(self.path, self.folder + " B1A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " B1P raw.gsf")))

			elif channel_name == "B2":
				self.map = load_gsf(os.path.join(self.path, self.folder + " B2A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " B2P raw.gsf")))

			elif channel_name == "B3":
				self.map = load_gsf(os.path.join(self.path, self.folder + " B3A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " B3P raw.gsf")))

			elif channel_name == "B4":
				self.map = load_gsf(os.path.join(self.path, self.folder + " B4A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " B4P raw.gsf")))

			elif channel_name == "B5":
				self.map = load_gsf(os.path.join(self.path, self.folder + " B5A raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " B5P raw.gsf")))

			elif channel_name == "Z":
				self.map = load_gsf(os.path.join(self.path, self.folder + " Z raw.gsf"))

			elif channel_name == "ZC":
				self.map = load_gsf(os.path.join(self.path, self.folder + " Z C.gsf"))

			elif channel_name == "E":
				self.map = load_gsf(os.path.join(self.path, self.folder + " EA raw.gsf"))*np.exp(1j*load_gsf(os.path.join(self.path, self.folder + " EP raw.gsf")))

			elif channel_name == "M":
				self.map = load_gsf(os.path.join(self.path, self.folder + " M raw.gsf"))

		self.x = np.linspace(0,float(self.x_max),num=int(self.Nx))
		self.y = np.linspace(float(self.y_min),float(self.y_max),num=int(self.Ny))
		self.X,self.Y = np.meshgrid(self.x,self.y)

		if self.type != "Frequency Sweep":
			if self.flip:
				self.map = np.flip(self.map,0)

		self.sections = []

		self.fft_flag = False
		self.plot_flag = False

		self.fig = None
		self.axs = None

		return self 

	def cut(self, x_range=None, x_reset=True, y_range=None, y_reset=False):
		# Default ranges: full extent
		if x_range is None:
			x_range = (0, None)
		if y_range is None:
			y_range = (0, None)

		xslice = slice(x_range[0], x_range[1])
		yslice = slice(y_range[0], y_range[1])

		# Cut arrays
		self.map = self.map[yslice, xslice]
		self.X   = self.X[yslice, xslice]
		self.Y   = self.Y[yslice, xslice]

		# Update coordinates
		self.x = self.x[xslice]
		self.y = self.y[yslice]

		if x_reset:
			self.X = self.X - self.x[0]
			self.x = self.x - self.x[0]

		if y_reset:
			self.Y = self.Y - self.y[0]
			self.y = self.y - self.y[0]

		return self

	def plot(self,fun='abs',cres=200,cmap='viridis',vmin=None,vmax=None,xlim=None,ylim=None,figsize=(8,6),save=False,show=True,pixel=False,colorbar=True, savedir = "Figures", data_type = ""):
			
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

				elif fun == 'phase':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(self.adjust_phase(),cres,cmap=cmap,vmin=vmin,vmax=vmax)
					self.fig = plt.xlabel('Fast axis, pixel',fontsize=18)
					self.fig = plt.ylabel('Slow axis, pixel',fontsize=18)

				elif fun == 'real':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(np.real(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)
					self.fig = plt.xlabel('Fast axis, pixel',fontsize=18)
					self.fig = plt.ylabel('Slow axis, pixel',fontsize=18)

				elif fun == 'imag':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(np.imag(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)
					self.fig = plt.xlabel('Fast axis, pixel',fontsize=18)
					self.fig = plt.ylabel('Slow axis, pixel',fontsize=18)

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
				elif fun == 'phase':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(self.X,self.Y,self.adjust_phase(),cres,cmap=cmap,vmin=vmin,vmax=vmax)
				elif fun == 'real':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(self.X,self.Y,np.real(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)
				elif fun == 'imag':
					self.fig = plt.figure(figsize=figsize)
					self.fig = plt.contourf(self.X,self.Y,np.imag(self.map),cres,cmap=cmap,vmin=vmin,vmax=vmax)

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
			# Extract the folder name
			folder_name = os.path.basename(os.path.normpath(self.path))

			# Split into tokens
			tokens = folder_name.split()

			# First token = date (YYYY-MM-DD)
			date = tokens[0]

			# Second token = measurement number (skip it)
			# Remaining tokens = user-chosen description
			description_tokens = tokens[2:] if len(tokens) > 2 else []

			# Join description with underscores
			description = "_".join(description_tokens)

			# Clean description (remove unwanted characters)
			description = re.sub(r'[^A-Za-z0-9_-]+', '_', description)

			# Build filename
			file_name = f"{date}_{description}_{self.channel_name}_{data_type}.png"

			# Full path
			file_path = os.path.join(savedir, file_name)

			# Ensure directory exists
			os.makedirs(os.path.dirname(file_path), exist_ok=True)

			# Save figure
			self.fig = plt.savefig(file_path, dpi=300, transparent=True, bbox_inches='tight')
			print(f"Plot saved as: {file_path}")

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
					print(self.sections[-1][0],np.abs(self.sections[-1][1]))
				elif fun == 'phase':
					self.fig = plt.plot(self.sections[-1][0],np.angle(self.sections[-1][1]))
				elif fun == 'real':
					self.fig = plt.plot(self.sections[-1][0],np.real(self.sections[-1][1]))
				elif fun == 'imag':
					self.fig = plt.plot(self.sections[-1][0],np.imag(self.sections[-1][1]))

				if direction == 'Vertical':
					if self.type == "Spatial":
						self.fig = plt.xlabel(r'Y  ($\mu$m)',fontsize=18)
					elif self.type == "Voltage Sweep":
						self.fig = plt.xlabel('Voltage, V',fontsize=18)
					elif self.type == "Frequency Sweep":
						self.fig = plt.xlabel(r'Wavenumber  (cm$^{-1})$',fontsize=18)

				elif direction == 'Horizontal':
					if self.fft_flag:
						self.fig = plt.xlabel(r'q, $\times 10^4$  (cm$^{-1})$',fontsize=18)	# Check the units
					else:
						self.fig = plt.xlabel(r'X  ($\mu$m)',fontsize=18)

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
						self.fig = plt.xlabel(r'Y  ($\mu$m)',fontsize=18)
					elif self.type == "Voltage Sweep":
						self.fig = plt.xlabel('Voltage, V',fontsize=18)
					elif self.type == "Frequency Sweep":
						self.fig = plt.xlabel(r'Wavenumber  (cm$^{-1})$',fontsize=18)

				elif direction == 'Horizontal':
					if self.fft_flag:
						self.fig = plt.xlabel(r'q, $\times 10^4$  (cm$^{-1})$',fontsize=18)	# Check the units
					else:
						self.fig = plt.xlabel(r'X  ($\mu$m)',fontsize=18)

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
						self.fig = plt.xlabel(r'Y  ($\mu$m)',fontsize=18)
					elif self.type == "Voltage Sweep":
						self.fig = plt.xlabel('Voltage, V',fontsize=18)
					elif self.type == "Frequency Sweep":
						self.fig = plt.xlabel(r'Wavenumber  (cm$^{-1})$',fontsize=18)

				elif direction == 'Horizontal':
					if self.fft_flag:
						self.fig = plt.xlabel(r'q, $\times 10^4$  (cm$^{-1})$',fontsize=18)	# Check the units
					else:
						self.fig = plt.xlabel(r'X  ($\mu$m)',fontsize=18)

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

	# DoS computaton functions

	def interpolate(self, interpolationFactor: int, method: str = 'linear'):
		# interpolate the sSNOM signal such us the map has more points.
		# the amount of which depends on the interpolationFactor:
		#
		# A Map (NxM) -> (p*N x p*M)
		#
		# Here, p is the interpolationFactor

		Interpolatior_Real_part = RegularGridInterpolator((self.y, self.x), np.real(self.map), method=method)
		Interpolatior_Imag_part = RegularGridInterpolator((self.y, self.x), np.imag(self.map), method=method)

		x = np.linspace(0,np.max(self.x), int(interpolationFactor*len(self.x)))
		y = np.linspace(0,np.max(self.y), int(interpolationFactor*len(self.y)))
		X_mesh, Y_mesh = np.meshgrid(y, x, indexing='ij')

		points_fine = np.vstack([X_mesh.ravel(), Y_mesh.ravel()]).T

		# Perform the interpolation
		Real_part = Interpolatior_Real_part(points_fine).reshape(Y_mesh.shape)
		Imag_part = Interpolatior_Imag_part(points_fine).reshape(Y_mesh.shape)

		# Update the class instance
		self.map = Real_part + 1j*Imag_part
		self.x = x
		self.y = y
		self.X, self.Y  = np.meshgrid(x, y)

		return self
	
	def extract_coordinates(self, i ,pixel):

		self.plot(pixel=pixel, show=False)

		plt.title(f'Select {i} points from the plot')
		coordinates = plt.ginput(i, timeout=-1)
		plt.close()

		return np.array(coordinates, dtype=int) if pixel else np.array(coordinates, dtype=float)
	
	def rotation(self, load: bool, save: bool, coordinates = None, path = ".\\Analysis Output\\"):
		# The rotation is performed by select two coordinates that identify a line.
		# This line will be aligned to the horizontal axis.

		coordinates = np.loadtxt(path + "Rotation\\" + self.folder + ".txt", delimiter=',') if load else coordinates

		if coordinates is None:
			coordinates = self.extract_coordinates(i=2,pixel=False)
			formatted_coordinates = np.array2string(coordinates, precision=8, separator=',', suppress_small=True, max_line_width=np.inf)
			print(f"The extracted coordinates to calculate the rotation are: {formatted_coordinates}")

		if save:
			os.makedirs(path + "Rotation\\") if not os.path.exists(path + "Center\\") else None 
			np.savetxt(path + "Rotation\\" + self.folder + ".txt", coordinates, delimiter=',', fmt='%.8f') 

		# Angle extraction
		angle = np.arctan((coordinates[1,1]-coordinates[0,1]) / (coordinates[1,0]-coordinates[0,0]))*180/np.pi

		# Scan rotation
		Real_part = rotate(np.real(self.map), angle, reshape=False)
		Imag_part = rotate(np.imag(self.map), angle, reshape=False)

		# Update the class instance
		self.map = Real_part + 1j*Imag_part

		return self
	
	def rectangle_cut(self, Lx, Ly, load: bool, save: bool, coordinates = None, path = ".\\Analysis Output\\"):

		dx = int(Lx/(self.x[1]-self.x[0]))
		dy = int(Ly/(self.y[1]-self.y[0]))

		coordinates = np.loadtxt(path + "Center\\" + self.folder + ".txt", delimiter=',', dtype=int).reshape(1, 2) if load else coordinates

		if coordinates is None:
			coordinates = self.extract_coordinates(i=1,pixel=True)
			formatted_coordinates = np.array2string(coordinates, precision=8, separator=',', suppress_small=True, max_line_width=np.inf)
			print(f"The extracted coordinates of the unit cell center are: {formatted_coordinates}")

		if save:
			os.makedirs(path + "Center\\") if not os.path.exists(path + "Center\\") else None 
			np.savetxt(path + "Center\\" + self.folder + ".txt", coordinates, delimiter=',', fmt='%d')

		self.cut(x_range=[coordinates[0,0]-dx//2,coordinates[0,0]+dx//2],y_range=[coordinates[0,1]-dy//2,coordinates[0,1]+dy//2],y_reset=True)

		return self
	
	def symmetrization(self):

		if self.map.shape[0] == self.map.shape[1]:
			Real_part = (np.real(self.map) + rotate(np.real(self.map), 90) + rotate(np.real(self.map), 180) + rotate(np.real(self.map), 270))/4
			Imag_part = (np.imag(self.map) + rotate(np.imag(self.map), 90) + rotate(np.imag(self.map), 180) + rotate(np.imag(self.map), 270))/4

		# Update the class instance
		self.map = Real_part + 1j*Imag_part

		return self

	def DoS_computation(self, Lx, Ly, load: bool, save: bool, BG_range = range(10,100), interpolationFactor = 5, coordinates_rotation = None, coordinates_center = None, symmetrization = True , plot = True):

		# Background extraction
		background = self.map
		Real_part = np.mean(np.real(background[:,BG_range]),axis=1)		# Average linewise (real part)
		Imag_part = np.mean(np.imag(background[:,BG_range]),axis=1)		# Average linewise (imaginary part)

		mean_background = np.mean(Real_part) + 1j*np.mean(Imag_part)

		Std_Real_part = np.std(np.real(background[:,BG_range]),axis=1)		# Standard Deviation linewise (real part)
		Std_Imag_part = np.std(np.imag(background[:,BG_range]),axis=1)		# Standard Deviation linewise (imaginary part)

		std_background = np.mean(Std_Real_part) + 1j*np.mean(Std_Imag_part)

		self.normalize(data = Real_part + 1j*Imag_part)

		# Unit cell extraction
		self.interpolate(interpolationFactor = interpolationFactor)										# interpolate for better accuracy during the geometrical transfomation step
		self.rotation(load = load, save = save, coordinates = coordinates_rotation)						# correct the scan misalignment with respect to the lattice
		self.rectangle_cut(Lx=Lx ,Ly=Ly ,load = load , save = save, coordinates=coordinates_center)		# cut the unit cell
		if plot: self.plot()

		if symmetrization:
			self.symmetrization()		# symmetrization
			if plot: self.plot()
		
		return	self.wavelength, np.mean(np.mean(self.map, axis=0)), mean_background, std_background, 

# --------------------------------------------- FUNCTIONS --------------------------------------------- #
# Normalization

def show_normalization_data(power_data, snom_scan, save = False, savedir = "Figures/normalization_data", add_save = ""): 
	# power_data is the intensities
	# snom_scan is the snom object created by the scan 
	# -------- goal : show the full power map

	x = snom_scan.x
	freq = snom_scan.y

	x_trimmed = x[:power_data.shape[1]]
	freq_trimmed = freq[:power_data.shape[0]]

	plot_maps(x_trimmed, freq_trimmed, power_data, save, savedir, f"normalization_data_{add_save}")

	#power_data_normalized = power_data/power_data[]


def I_avg_freq_dependancy(power_data, snom_scan, num_pixel=1,
                          list_SNOM_signal=["O1", "O2", "O3", "O4"], normalized_signal = True,
                          save=False, savedir="Figures/normalization_data",
                          add_save=""):

	x = snom_scan.x
	freq = snom_scan.y

	x_trimmed = x[:num_pixel]
	freq_trimmed = freq[:power_data.shape[0]]
	power_data_trimmed = power_data[:, :num_pixel]

	I_means = np.mean(power_data_trimmed, axis=1)
	I_stds  = np.std(power_data_trimmed, axis=1)

	if normalized_signal:
		I_means = I_means/np.max(I_means)

	all_I_means = [I_means]
	all_I_stds  = [I_stds]

	I_plot_percentage_variation(freq_trimmed, I_means, I_stds, label="Power Data")


	# Loop over SNOM channels
	for sig in list_SNOM_signal:
		I_Oi = np.abs(snom_scan.channel(sig).map)
		I_Oi_trimmed = I_Oi[:power_data.shape[0], :num_pixel]

		I_means_Oi = np.mean(I_Oi_trimmed, axis=1)
		I_stds_Oi  = np.std(I_Oi_trimmed, axis=1)

		if normalized_signal: 
			I_means_Oi = I_means_Oi/np.max(I_means_Oi)

		all_I_means.append(I_means_Oi)
		all_I_stds.append(I_stds_Oi)

		plots_I_avg_freq_dependancy(x_trimmed, freq_trimmed,[I_means_Oi], [I_stds_Oi], labels = [sig], std_activated= False, save = save ,name = sig + " mean intensity in function of the frequency "+f"{add_save}")

		plot_I_std_freq(x_trimmed, freq_trimmed, I_means_Oi, I_stds_Oi)

	plots_I_avg_freq_dependancy(x_trimmed, freq_trimmed,
								[I_means], [I_stds], std_activated= False, labels = ["power data"],save = save ,name= "Power mean intensity in function of the frequency "+f"{add_save}")

	# Convert lists to 2D numpy arrays: shape = (n_signals, n_freqs)
	all_I_means = np.vstack(all_I_means)
	all_I_stds  = np.vstack(all_I_stds)

	plots_I_avg_freq_dependancy(x_trimmed, freq_trimmed,
								all_I_means, all_I_stds, labels = ["power data "]+ list_SNOM_signal, std_activated=False, name =  "Comparison of mean intesities "+f"{add_save}"+" ".join(list_SNOM_signal)+" power data")

	return freq_trimmed, all_I_means, all_I_stds
	

# maps division	
def Oi_power_data_maps_division(power_data, snom_scan, num_pixel = 1,list_SNOM_signal=["O1", "O2", "O3", "O4"], save = False, savedir = "Figures", add_save = ""):
	x = snom_scan.x
	freq = snom_scan.y

	x_trimmed = x[:num_pixel]
	freq_trimmed = freq[:power_data.shape[0]]
	power_data_trimmed = power_data[:, :num_pixel]

	plot_maps(x_trimmed, freq_trimmed, power_data_trimmed)

	I_means = np.mean(power_data_trimmed, axis=1)
	I_means = I_means[:, None] # to enable the division of each scan by the I mean at each frequency

	for sig in list_SNOM_signal:
		I_Oi = np.abs(snom_scan.channel(sig).map)
		I_Oi_trimmed = I_Oi[:power_data.shape[0], :num_pixel]

		plot_maps(x_trimmed, freq_trimmed, I_Oi_trimmed)

		print((I_means))

		I_Oi_over_power = I_Oi_trimmed/I_means # indeed as std of 3 percent in the I mean, then can use it like this (consider variation negligable)
		I_Oi_full_matrix_over_power = I_Oi_trimmed/power_data_trimmed

		plot_maps(x_trimmed, freq_trimmed, I_Oi_over_power, save= save ,savedir = savedir ,name= sig+" signal divided by mean power "+f"{add_save}")
		plot_maps(x_trimmed, freq_trimmed, I_Oi_full_matrix_over_power , save= save, savedir = savedir , name = sig +" full matrix division by power data "+f"{add_save}")

# plot functions
def plot_maps(x, f, I_map, save = False, savedir = "Figures", name = ""):

	##### plots 
	plt.figure(figsize=(8,6))
	extent = [x[0], x[-1], f[0], f[-1]]
	im = plt.imshow(I_map, cmap='hot', origin='lower', aspect='auto', extent=extent)

	# Add a colorbar to show intensity values
	cbar = plt.colorbar(im)
	cbar.set_label("Intensity", fontsize=16)   # colorbar label fontsize
	cbar.ax.tick_params(labelsize=18)   

	# Label axes with custom font sizes
	plt.xlabel("$x$ ($\\mu$m)", fontsize=16)   # x-axis label size 16
	plt.ylabel("$f$ (cm$^{-1}$)", fontsize=16) # y-axis label size 16
	plt.title(f"{name}", fontsize=12) # title size 12

	# Set tick label font sizes
	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)

	# saving the files
	if save:
		name_save = name.replace(" ", "_")	

		file_name = f"{name_save}.png"
		file_path = os.path.join(savedir, file_name)
		os.makedirs(os.path.dirname(file_path), exist_ok=True)
		plt.savefig(file_path, dpi=300, transparent=True, bbox_inches='tight')

	plt.show()


def plots_I_avg_freq_dependancy(x,f, I_means_list, I_stds_list, labels,std_activated = True, save = False , name = "", savedir = "Figures"):

	plt.figure(figsize=(5,8))
	for i in range(len(I_means_list)):
		plt.plot(I_means_list[i],f, label="Mean "+labels[i])
		if std_activated :
			plt.fill_between(
				f,
				I_means_list[i] - I_stds_list[i],
				I_means_list[i] + I_stds_list[i],
				color="blue",
				alpha=0.3,   # transparency
				label="± std "+labels[i]
			)

	plt.ylabel("Frequency", fontsize = 16)
	plt.xlabel("Intensity", fontsize = 16)
	plt.legend()
	plt.title(f"{name}", fontsize = 14)
	plt.grid(True, linestyle="--", alpha=0.6)  # optional, makes it easier to read

	plt.xticks(fontsize=18)
	plt.yticks(fontsize=18)

	plt.tight_layout()

	if save:
		name_save = name.replace(" ", "_")	

		file_name = f"{name_save}.png"
		file_path = os.path.join(savedir, file_name)
		os.makedirs(os.path.dirname(file_path), exist_ok=True)
		plt.savefig(file_path, dpi=300, transparent=True, bbox_inches='tight')

	plt.show()



def plot_I_std_freq(x,f,I_means, I_stds):

	plt.figure(figsize=(8,5))
	plt.plot(f, I_stds)

	plt.xlabel("Frequency")
	plt.ylabel("Std. deviation")
	plt.title("Laser Signal Variability")
	plt.grid(True, linestyle="--", alpha=0.6)  # optional, makes it easier to read
	plt.tight_layout()
	plt.show()

def I_plot_percentage_variation(freq, I_means, I_stds, label="Signal"):
    percentage_variation = (I_stds / I_means) * 100

    plt.figure(figsize=(8,5))
    plt.plot(freq, percentage_variation, label=label)
    plt.xlabel("Frequency")
    plt.ylabel("Percentage Variation ($\%$)")
    plt.title(f"{label} - Percentage Variation vs Frequency")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# sSNOM Data analysis
def peak_spacing_in_SNOM(snom_data, mesure_position ,pixel_range = [1], plot = False, saveindex = [], savedir = "Figures"):
	peak_distances = []
	for i in pixel_range:
		section_at_i = [snom_data.x, np.abs(snom_data.map[i, :][:])] # gives x and Oi signal
		peaks, properties = find_peaks(savgol_filter(section_at_i[1], window_length=11, polyorder=3), height=0 )
		if plot:
			plt.figure()
			plt.plot(section_at_i[0], section_at_i[1])
			plt.plot(section_at_i[0][peaks], section_at_i[1][peaks], '.')
			plt.xlim(mesure_position)
			plt.xlabel("$x$ ($\mu$m)")
			plt.ylabel("I (u.a.)")
			name_save = f"{i}, {snom_data.y[i]}"+"cm$^{-1}$ oscillations in hBN"
			plt.title(name_save)
			plt.grid(True, linestyle="--", alpha=0.6)  # optional, makes it easier to read

			
			if i in saveindex:
				name_save = name_save.replace(" ", "_")	
				file_name = f"{name_save}.png"
				file_path = os.path.join(savedir, file_name)
				os.makedirs(os.path.dirname(file_path), exist_ok=True)
				plt.savefig(file_path, dpi=300, transparent=True, bbox_inches='tight')
			plt.show()


		peaks_in_range = [p for p in peaks if mesure_position[0] <= section_at_i[0][p] <= mesure_position[1]]

		diff_1a2 = (section_at_i[0][peaks_in_range[-1]]-section_at_i[0][peaks_in_range[-2]])
		diff_2a3 = (section_at_i[0][peaks_in_range[-2]]-section_at_i[0][peaks_in_range[-3]])

		if section_at_i[1][peaks_in_range[-1]] < section_at_i[1][peaks_in_range[-2]]:
			mean_peak_distance = (diff_2a3 +(section_at_i[0][peaks_in_range[-3]]-section_at_i[0][peaks_in_range[-4]]))/2
			#mean_peak_distance = diff_2a3
			#print(f"peak {i} ")
		else:
			mean_peak_distance = (diff_1a2 + diff_2a3)/2
			#mean_peak_distance = diff_1a2

		peak_distances = peak_distances + [mean_peak_distance]

	return peak_distances	


def plot_peak_spacing(peak_distances, snom_data, good_index_list, thickness,n , plot = True, save = False):
	if plot:
		plt.figure(figsize=(8,6))
		plt.plot(peak_distances,( snom_data.y[good_index_list]), '.') 
		plt.xlabel("Mean peak difference ($\mu$m)", fontsize = 16)
		plt.ylabel("Wavenumber (cm$^{-1}$)", fontsize = 16)
		plt.xticks(fontsize=18)
		plt.yticks(fontsize=18)
		plt.grid(True, linestyle="--", alpha=0.6)  # optional, makes it easier to read
		plt.tight_layout()
		plt.show()
	
	q = np.array([1/x for x in peak_distances])*2*np.pi/2 # to have actual wavevector
	new_freqs = np.array(snom_data.y[good_index_list])

	m_fit, b_fit = np.polyfit(q, new_freqs, 1)

	# fit part in h11BN for thickness
	epsilon_xx = 5.32 * (1.+ (((1608.8)**2 -1359.8**2)/(1359.8**2 - new_freqs**2. - 1j * new_freqs* 2.1)))
	epsilon_zz = 3.15 * (1.+ (((814)**2 -785.**2)/(785.**2 - new_freqs**2. - 1j * new_freqs* 1.)))

	epsilon_minus = -10000. #(gold)
	epsilon_plus = 1. # air

	phi = np.sqrt(- epsilon_xx/epsilon_zz +0j)

	r_plus = (epsilon_xx - 1j * epsilon_plus * phi)/(epsilon_xx + 1j * epsilon_plus * phi)
	r_minus = (epsilon_xx - 1j * epsilon_minus * phi)/(epsilon_xx + 1j * epsilon_minus * phi)

	rho_plus = np.unwrap(np.angle(r_plus), discont=np.pi) * (1/np.pi)
	rho_minus = np.unwrap(np.angle(r_minus), discont=np.pi) * (1/np.pi)

	k_z_real = (np.pi / (2*thickness)) * (2*n + rho_plus + rho_minus)
	#k_z_imag = (1j / (2*thickness)) * np.log(np.abs(r_plus) * np.abs(r_minus))
	k_z = k_z_real #+ k_z_imag
	wavenumber_k_x =( k_z / phi )* 1e-4
	k_x_theoretical = np.real(wavenumber_k_x)   # Convert cm⁻¹ to μm⁻¹

	if plot:
		plt.figure(figsize=(8,6))
		plt.plot(q,new_freqs, '.', label = "data") 
		#plt.plot(q, m_fit*q + b_fit, '-', label = f"fit: ${m_fit:.3f}x + {b_fit:.3f}$")

		plt.plot( k_x_theoretical, new_freqs , label = f"Computed t={(thickness*1e7):.1f} nm")

		plt.xlabel("Fringes Wavevector (\mum^{-1} \cdot 2 \pi /2)", fontsize = 16)
		plt.ylabel("Wavenumber (cm^{-1})", fontsize = 16)
		plt.xticks(fontsize=18)
		plt.yticks(fontsize=18)
		plt.legend()
		plt.grid(True, linestyle="--", alpha=0.6)  # optional, makes it easier to read
		plt.tight_layout()
		plt.show()

	return m_fit, b_fit, wavenumber_k_x, new_freqs


def prepare_data(i, data, measure_position=(2, 3.01)):
    """Extract, slice, flip, and normalize data for index i."""
    print(f"Raw y[{i}]: {data.y[i]}")
    x, y = data.x, data.map[i, :]

    # Select measurement window
    mask = (x >= measure_position[0]) & (x <= measure_position[1])
    x, y = x[mask], y[mask]

    # Flip and normalize
    y = np.flip(y)
    x = x - x[0] + 1e-4
    return x, y

def fit_fringes(x, y, qr, qi, plot=True):
    """Fit the fringes model and optionally plot results."""

    def fringes(x, A, B, C, loss, phi):
        return (A * np.sin(2*qr*x + phi) * np.exp(-2*qi*loss*x) / np.sqrt(x)
              + B * np.sin(qr*x + phi) * np.exp(-qi*loss*x) / np.sqrt(x**3)
              + C)

    popt, _ = curve_fit(
        fringes, x, np.real(y),
        p0=[1, 0.1, 0.5, 1, 0.5],
        bounds=([-100., -10000., -10., 1, -2*np.pi],
                [100., 10000., 10., 6., 2*np.pi]),
        maxfev=1_000_000
    )

    print(f"A:    {popt[0]:.1f}")
    print(f"B:    {popt[1]:.1f}")
    print(f"C:    {popt[2]:.1f}")
    print(f"loss: {popt[3]:.1f}")
    print(f"phi:  {popt[4]*180/np.pi:.1f}")

    if plot:
        y_fit = fringes(x, *popt)
        plt.figure(figsize=(10, 5))
        plt.plot(x, np.real(y), '.', label='Data (Real)', alpha=0.5)
        plt.plot(x, np.real(y_fit), '-', label='Fit (Real)', linewidth=2)
        plt.xlabel("x")
        plt.ylabel("Re(y)")
        plt.legend()
        plt.tight_layout()
        plt.show()

    return popt


def fit_peaks_model(x, y, plot=True):
	"""Detect peaks, fit sqrt(x) model, and optionally plot results."""

	peaks, _ = find_peaks(savgol_filter(y, window_length=11, polyorder=3), height=0)

	def model(x, a, b):
		return a / np.sqrt(x**3) + b

	params, _ = curve_fit(model, x[peaks], np.real(y[peaks]), p0=[1, 0])
	a_fit, b_fit = params
	print(f"Fitted parameters: a = {a_fit:.3f}, b = {b_fit:.3f}")

	if plot:
		plt.figure(figsize=(8, 6))
		plt.plot(x, np.real(y), '.', label="Data")
		plt.plot(x[peaks], np.real(y[peaks]), '.', label="Peaks")
		plt.plot(x, model(x, a_fit, b_fit), color="red", label=r"$1/\sqrt{x^3}$ Fit")
		plt.xlabel("x")
		#plt.xlim([0.1,1])
		plt.ylim([0.75,1.25])
		plt.ylabel("y")
		plt.legend()
		plt.tight_layout()
		plt.show()

	return a_fit, b_fit

def analyze_fringes(i, data, k_x_theoretical, measure_position=(2, 3.01), plot=True):
	"""Main wrapper: prepares data, fits fringes, and fits peaks model."""
	x, y = prepare_data(i, data, measure_position)
	qr, qi = np.real(k_x_theoretical[i]), np.imag(k_x_theoretical[i])

	print("/n qr :",qr, " qi :", qi, "/n")

	popt_fringes = fit_fringes(x, y, qr, qi, plot=plot)
	popt_model = fit_peaks_model(x, y, plot=plot)

	return popt_fringes, popt_model


# --------------------------------------------- TEST CODE --------------------------------------------- #

if __name__ == '__main__':

	Data_path = "T:\\dataSNOM\\Cavity QED\\Patterned hBN\\PBN06\\High_resolution_scans_DoS"

	i = 0

	example = snom(load_folder(Data_path,i))	# select scan
	example.print_details()						# select channel

	example.plot()
