import os, re, struct

from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit
from scipy.signal import find_peaks, savgol_filter
from scipy.ndimage import rotate
from scipy.stats import linregress

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, LinearSegmentedColormap
from mpl_toolkits import axes_grid1

from tkinter import Tk
from tkinter.filedialog import asksaveasfilename

import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift

from skimage.feature import peak_local_max
from skimage.restoration import denoise_wavelet

__version__ = "1.1.0"
__authors__ = ["Lorenzo Orsini","Elisa Mendels","Matteo Ceccanti", "Bianca Turini"]

# NOTES
# The variable k is the light wavenumber (1/λ) in cm⁻¹

# REFERENCES

# ----------------------------------------------------------------------------------------------------- #
#                                    Functions and class description                                    #
# ----------------------------------------------------------------------------------------------------- #

# ---------------------------------------------- LOADING ---------------------------------------------- #

def add_colorbar(im, aspect=20, pad_fraction=0.5, **kwargs):
	"""Add a vertical color bar to an image plot."""
	divider = axes_grid1.make_axes_locatable(im.axes)
	width = axes_grid1.axes_size.AxesY(im.axes, aspect=1./aspect) # type: ignore
	pad = axes_grid1.axes_size.Fraction(pad_fraction, width)	  # type: ignore
	current_ax = plt.gca()
	cax = divider.append_axes("right", size=width, pad=pad)
	plt.sca(current_ax)
	return im.axes.figure.colorbar(im, cax=cax, **kwargs)

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
		self.fft2D_flag = False
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

	def plot(self, fun='abs', cres=200, cmap='viridis', vmin=None, vmax=None,
             xlim=None, ylim=None, figsize=(8,6), save=False, show=True,
             pixel=False, colorbar=True, aspect='auto', savedir="Figures",
             data_type="", logscale=False, use_imshow=False):
			 
		# --- axis labels ---
		def axis_labels(fft_flag=False, fft2D_flag=False, pixel= False):
			if fft2D_flag:
				if pixel: 
					xlabel = r'$q_x$ (pixel)'
					ylabel = r'$q_y$ (pixel)'
				else:
					xlabel = r'$q_x$ ($\mu$m$^{-1}$)'
					ylabel = r'$q_y$ ($\mu$m$^{-1}$)'
			else:
				if pixel:
					if fft_flag:
						xlabel = r'q (pixel)$'
					else:
						xlabel = r'X  (pixel)'

					if self.type == "Spatial":
						ylabel = r'Y  (pixel)'
					elif self.type == "Voltage Sweep":
						ylabel = 'Voltage (pixel)'
					elif self.type == "Frequency Sweep":
						ylabel = r'Wavenumber  (pixel)$'
					else:
						ylabel = r'Y  (pixel)'

				else: 
					if fft_flag:
						xlabel = r'q, $\times 10^4$  (cm$^{-1})$'
					else:
						xlabel = r'X  ($\mu$m)'

					if self.type == "Spatial":
						ylabel = r'Y  ($\mu$m)'
					elif self.type == "Voltage Sweep":
						ylabel = 'Voltage, V'
					elif self.type == "Frequency Sweep":
						ylabel = r'Wavenumber  (cm$^{-1})$'
					else:
						ylabel = r'Y  ($\mu$m)'    

			return xlabel, ylabel

		# --- extract data by type ---
		def fun_data(fun_type, logscale=False):
			if fun_type == 'abs':
				data = np.abs(self.map)
				if logscale:
					data = np.log1p(data)
				return data
			elif fun_type == 'phase':
				return self.adjust_phase()
			elif fun_type == 'real':
				return np.real(self.map)
			elif fun_type == 'imag':
				return np.imag(self.map)
			else:
				raise ValueError(f"Unknown fun_type: {fun_type}")

		# --- Local helper: draw one subplot ---
		def one_plot(ax, X, Y, data, pixel_mode=False, aspect='auto',
								cres=200, cmap='viridis', vmin=None, vmax=None,
								use_imshow=False, colorbar=True, fft_flag=False,
								fft2D_flag=False, fun=None, n_fun = 1):

			if use_imshow:
				if pixel_mode:
					im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
									origin='lower', aspect='auto')
					ax.xlim([self.x.min(),self.x.max()])
					ax.ylim([self.y.min(), self.y.max()])
				else:
					
					extent = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]
					im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax,
									extent=extent, origin='lower',
									interpolation='nearest',
									aspect='equal' if aspect == 'equal' else 'auto')
			else:
				if pixel_mode:
					im = ax.contourf(data, cres, cmap=cmap, vmin=vmin, vmax=vmax)
				else:
					im = ax.contourf(X, Y, data, cres, cmap=cmap, vmin=vmin, vmax=vmax)

			xlabel, ylabel = axis_labels(
				fft_flag=fft_flag, fft2D_flag=fft2D_flag, pixel = pixel_mode
			)
			ax.set_xlabel(xlabel, fontsize=18)
			ax.set_ylabel(ylabel, fontsize=18)
			ax.tick_params(axis='both', which='major', labelsize=16)
			ax.tick_params(axis='both', which='minor', labelsize=16)

			if aspect == 'equal':
				ax.set_aspect('equal')

			if aspect == "equal":
				add_colorbar(im=im)
			elif colorbar:
				plt.colorbar(im, ax=ax)
				

			title_dict = {"abs" : "Absolute value", "phase": "Phase", "real": "Real part", "imag": "Imaginary part" }

			if fun is not None:
				if n_fun == 1:
					ax.set_title(f"{self.folder}  {self.channel_name}  {fun}", fontsize = 12)
				else: 
					ax.set_title(f"{title_dict[fun]}", fontsize = 12)

			return im

		# nb of plot
		if fun == 'all':
			fun_list = ['abs', 'phase', 'real', 'imag']
		elif isinstance(fun, list):
			fun_list = fun
		else:
			fun_list = [fun]

		nplots = len(fun_list)

		if nplots == 1: 
			n_vert_subplots = 1 
			n_hor_subplots = 1 
		else: 
			n_hor_subplots = 2 
			n_vert_subplots = (nplots + 1) // 2

		# --- Create figure and axes ---
		fig, axs = plt.subplots(
			n_vert_subplots, n_hor_subplots , figsize=(figsize[0] * n_hor_subplots, figsize[1]*n_vert_subplots), squeeze=False
		)
		axs = axs[0]

		# --- Coordinates ---
		if pixel:
			X = Y = None
		else:
			X = self.X
			Y = self.Y

		images = []
		for ax, fun_type in zip(axs, fun_list):
			data = fun_data(fun_type, logscale=logscale)

			im = one_plot(
				ax, X, Y, data,
				pixel_mode=pixel,
				aspect=aspect,
				cres=cres,
				cmap=cmap,
				vmin=vmin, vmax=vmax,
				use_imshow=use_imshow,
				colorbar=colorbar,
				fft_flag=self.fft_flag,
				fft2D_flag= self.fft2D_flag,
				fun=fun_type, 
				n_fun= nplots,
			)
			images.append(im)

			if xlim is not None:
				ax.set_xlim(xlim)
			if ylim is not None:
				ax.set_ylim(ylim)

		if nplots !=1: 
			fig.suptitle(self.folder + "  " + self.channel_name, fontsize = 16)

		# --- Attach to self for later use ---
		if nplots == 1:
			self.fig = fig
			self.axs = axs[0]
			self._last_image = images[0]
		else:
			self.fig = fig
			self.axs = axs
			self._last_images = images

		# --- Save and show ---
		if save:
			self._save_plot(self.fig, savedir=savedir, data_type=data_type)

		if show:
			plt.show()

		return self

	def _save_plot(self, fig, savedir="Figures", data_type=""):
			"""Save the plot to a file.""" 
			folder_name = os.path.basename(os.path.normpath(self.path)) 
			tokens = folder_name.split() 
			date = tokens[0] if tokens else "unknown" 
			description_tokens = tokens[2:] if len(tokens) > 2 else [] 
			description = "_".join(description_tokens) 
			description = re.sub(r'[^A-Za-z0-9_-]+', '_', description) 
			file_name = f"{date}_{description}_{self.channel_name}_{data_type}.png" 
			file_path = os.path.join(savedir, file_name) 
			os.makedirs(os.path.dirname(file_path), exist_ok=True) 
			fig.savefig(file_path, dpi=300, transparent=True, bbox_inches='tight') 
			print(f"Plot saved as: {file_path}")

	def fft(self):

		FFT_resolution = 1/(self.x[1] - self.x[0])
		q_max = FFT_resolution/2

		self.map = np.fft.fft(np.abs(self.map), axis = 1)
		self.map = abs(np.fft.fftshift(self.map,axes=1))

		self.x = np.linspace(-q_max,q_max,np.shape(self.map)[1]) # spatial frequency in cycles/length (e.g., μm^-1)
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
	
	def extract_coordinates(self, i, pixel, save_path=None):
		"""
		Select points from a plot.
		i: number of points
		pixel: True=int coords, False=float
		save_path: optional file to save coords
		"""
		self.plot(pixel=pixel, show=False)
		plt.title(f'Select {i} points')
		coordinates = plt.ginput(i, timeout=-1)
		plt.close()

		coords_array = np.array(coordinates, dtype=int if pixel else float)

		if save_path:
			np.savetxt(save_path, coords_array, fmt='%d' if pixel else '%.6f')

		return coords_array
	
	def rotation(self, load: bool, save: bool, coordinates=None, path="Analysis Output"):
		rotation_dir = os.path.join(path, "Rotation")
		rotation_file = os.path.join(rotation_dir, f"{self.folder}.txt")

		if load:
			coordinates = np.loadtxt(rotation_file, delimiter=',')
		else:
			coordinates = coordinates

		if coordinates is None:
			coordinates = self.extract_coordinates(i=2, pixel=False)
			formatted_coordinates = np.array2string(
				coordinates,
				precision=8,
				separator=',',
				suppress_small=True,
				max_line_width=np.inf
			)
			print(f"The extracted coordinates to calculate the rotation are: {formatted_coordinates}")

		if save:
			# Ensure directory exists
			os.makedirs(rotation_dir, exist_ok=True)
			np.savetxt(rotation_file, coordinates, delimiter=',', fmt='%.8f')

		# Angle extraction
		dx = coordinates[1, 0] - coordinates[0, 0]
		dy = coordinates[1, 1] - coordinates[0, 1]
		angle = np.arctan2(dy, dx) * 180 / np.pi  # safer than plain division

		# Scan rotation
		real_part = rotate(np.real(self.map), angle, reshape=False)
		imag_part = rotate(np.imag(self.map), angle, reshape=False)

		# Update the class instance
		self.map = real_part + 1j * imag_part

		return self 
	
	def rectangle_cut(self, Lx, Ly, load: bool, save: bool, coordinates=None, path="Analysis Output"):
		dx = int(Lx / (self.x[1] - self.x[0]))
		dy = int(Ly / (self.y[1] - self.y[0]))

		center_dir = os.path.join(path, "Center")
		center_file = os.path.join(center_dir, f"{self.folder}.txt")

		if load:
			coordinates = np.loadtxt(center_file, delimiter=',', dtype=int).reshape(1, 2)
		else:
			coordinates = coordinates

		if coordinates is None:
			coordinates = self.extract_coordinates(i=1, pixel=True)
			formatted_coordinates = np.array2string(
				coordinates,
				precision=8,
				separator=',',
				suppress_small=True,
				max_line_width=np.inf
			)
			print(f"The extracted coordinates of the unit cell center are: {formatted_coordinates}")

		if save:
			os.makedirs(center_dir, exist_ok=True)
			np.savetxt(center_file, coordinates, delimiter=',', fmt='%d')

		self.cut(
			x_range=[coordinates[0, 0] - dx // 2, coordinates[0, 0] + dx // 2],
			y_range=[coordinates[0, 1] - dy // 2, coordinates[0, 1] + dy // 2],
			y_reset=True
		)

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

	# FFT analysis functions

	def fft_2D(self):
		"""
		Perform 2D FFT on self.map and adapt coordinates.
		- Computes FFT resolution from x,y spacing
		- Applies 2D FFT and shift
		- Updates reciprocal-space coordinates
		- Optional log scaling of FFT intensity
		"""

		# FFT resolutions
		dx = self.x[1] - self.x[0]
		dy = self.y[1] - self.y[0]
		qx_max = 1/(2*dx)
		qy_max = 1/(2*dy)

		# 2D FFT + shift
		fft_map = np.fft.fftshift(np.fft.fft2(np.abs(self.map)))

		# Update map
		self.map = fft_map

		# Reciprocal coordinates
		self.x = np.linspace(-qx_max, qx_max, self.map.shape[1])
		self.y = np.linspace(-qy_max, qy_max, self.map.shape[0])
		self.X, self.Y = np.meshgrid(self.x, self.y)

		self.fft2D_flag = True

		return self

	def inverse_fft_2D(self):
		dx = 1/(2*self.x[-1])
		dy = 1/(2*self.y[-1])
		x_max = dx*self.map.shape[1]
		y_max = dy*self.map.shape[0]

		self.x = np.linspace(0, x_max, self.map.shape[1])
		self.y = np.linspace(0, y_max, self.map.shape[0])
		self.X, self.Y = np.meshgrid(self.x, self.y)

		ifft_map =  ifft2(ifftshift(self.map))
		self.map = ifft_map

		self.fft2D_flag = False

		return self

	def plot_kx_modes_circles_on_2Dfft(self, thickness, freq_cm1, modes=(0, 1),
		epsilon_minus=2.3, hbn_type=10, n_theta=500, plot_2k_mode=False,
		vmax=None, vmin=None, cres=1000, xlim=None, ylim=None, aspect="equal",
		colors=None, labels=None, show=True, logscale = True, cmap = "viridis", use_imshow = False):

		# Plot FFT map
		self.plot(
			vmax=vmax, vmin=vmin, cres=cres,
			xlim=xlim, ylim=ylim,
			pixel=False, show=False, save=False,
			data_type="fft", aspect=aspect, logscale= logscale,
			cmap=cmap, use_imshow = use_imshow
		)

		fig = plt.gcf()
		ax = plt.gca()

		# Overlay circles
		_, labels = plot_kx_circles(ax, thickness, freq_cm1, modes,
										epsilon_minus, hbn_type, n_theta,
										colors, labels, plot_2k_mode)
		ax.set_xlim([self.x.min(), self.x.max()])
		ax.set_ylim([self.y.min(), self.y.max()])
		if xlim is not None: ax.set_xlim(xlim)
		if ylim is not None: ax.set_ylim(ylim)
		ax.legend(frameon=False, labelcolor='white', loc = "upper right")

		if show:
			plt.tight_layout()
			plt.show()
		return self    

	def plot_kx_on_1Dfft(self, thickness, modes, hbn_type, epsilon_minus,
				x_lim = None, y_lim = None, vmax = None, cres = 200, savepath = None):
		"""
		Plot SNOM FFT data and overlay theoretical dispersion curves for multiple mode orders.

		Parameters
		----------
		snom_data : object
			SNOM dataset object with a .plot(...) method.
		thickness : float
			Layer thickness in meters (e.g. 38e-9).
		n_list : list of int
			Mode indices for theoretical calculation (e.g. [0, 1, 2]).
		hbn_type : int
			hBN type parameter for sp.k_x_theoretical.
		epsilon_minus : float
			Substrate dielectric constant (e.g. 1.5 for SiO2).
		freq_min, freq_max : float
			Frequency range in cm^-1.
		df : float
			Frequency step size.
		x_lim, y_lim : tuple
			Axis limits for kx (µm^-1) and frequency (cm^-1).
		vmax, cres : float
			Plotting parameters for snom_data.plot.
		savepath : str or None
			If given, path to save the figure (e.g. "Figures/snom_data/fft_with_theory.png").
		"""

		# Plot FFT data
		self.plot(
			vmax=vmax, cres=cres, xlim=x_lim, ylim=y_lim,
			pixel=False, show=False, save=False,
			data_type="fft"
		)
		fig = plt.gcf()
		ax = plt.gca()

		# Overlay theoretical curves for each mode order
		for n in modes:
			k_x_theoretical_ = k_x_theoretical(new_freqs= self.y, thickness=thickness, n=n, hbn_type=hbn_type,
												epsilon_minus=epsilon_minus, plot=False)
			k_x_theoretical_plot = np.real(k_x_theoretical_) * 1e-6  # µm^-1

			ax.plot(k_x_theoretical_plot, self.y, lw=2.5, label=f"Re($k_x$), n={n}, $\epsilon_-$={epsilon_minus:.2f}")
			ax.plot(2*k_x_theoretical_plot, self.y, lw=2.5, linestyle="--", label=f"2 Re($k_x$), n={n}" )

		ax.legend(fontsize=12,loc = "upper right")
		ax.set_xlim(x_lim)
		ax.set_ylim(y_lim)

		fig.tight_layout()
		if savepath:
			fig.savefig(savepath, dpi=300)
			plt.close(fig)
		else:
			plt.show()    
		return self

	def filtering_real_space(self, mask_func, mask_args=(), mask_kwargs=None,
		thickness=None, modes=(0,1), epsilon_minus=2.3, hbn_type=10, plot = False, plot_circles=False, plot_2k_mode=False,
			vmin = None, vmax = None, logscale = True, use_imshow = False):
		"""Apply FFT, filter with a mask, and plot results with *snompy*-style formatting."""

		
		if mask_kwargs is None:
			mask_kwargs = {}

		### --- filtering functions ----- 
		dx, dy = self.x[1] - self.x[0], self.y[1] - self.y[0]

		self.fft_2D()
		
		if plot: 
			if plot_circles: 
				self.plot_kx_modes_circles_on_2Dfft(thickness=thickness, freq_cm1= self.wavelength, epsilon_minus=epsilon_minus,
													hbn_type=hbn_type, plot_2k_mode=plot_2k_mode, vmin=vmin, vmax=vmax, logscale=logscale, use_imshow = use_imshow)
			else:
				self.plot(logscale=True, vmin = vmin , vmax = vmax)

		mask = mask_func(self.map.shape, dx=dx, dy=dy, *mask_args, **mask_kwargs)

		if plot: 
			fig, ax = plt.subplots(figsize=(6, 5))
			im = ax.imshow(mask, cmap="gray", extent=[self.x.min(), self.x.max(), self.y.min(), self.y.max()], 
							origin="lower")
			ax.set_title("Mask")
			ax.set_xlabel(r"$q_x$ ($\mu$m$^{-1}$)", fontsize=18)
			ax.set_ylabel(r"$q_y$ ($\mu$m$^{-1}$)", fontsize=18)
			ax.tick_params(axis="both", which="major", labelsize=16)

			if plot_circles:
				_, labels = plot_kx_circles(ax, thickness, self.wavelength, modes, epsilon_minus, hbn_type,
											plot_2k_mode=plot_2k_mode)
				ax.legend(frameon=True, bbox_to_anchor=(1.2, 1), loc='upper left')

			cbar = fig.colorbar(im, ax=ax, shrink=0.8)
			cbar.ax.set_ylabel("Intensity")

			ax.set_xlim(self.x.min(), self.x.max())
			ax.set_ylim(self.y.min(), self.y.max())
			
			plt.tight_layout()
			plt.show()

		masked_fft = self.map * mask
		self.map = masked_fft
			
		if plot: 
			if plot_circles: 
				self.plot_kx_modes_circles_on_2Dfft(thickness=thickness, freq_cm1= self.wavelength, epsilon_minus=epsilon_minus,
													hbn_type=hbn_type, plot_2k_mode=plot_2k_mode, logscale=logscale,
													cmap = "inferno", use_imshow=use_imshow) #vmin=vmin, vmax=vmax,
			else:
				self.plot(logscale=True, vmin = vmin , vmax = vmax)

		self.inverse_fft_2D()

		if plot: 
			self.plot(cmap = "hot", use_imshow=use_imshow)

		return self        

	def findpeaks_2DFFT(self, mindistance_peaks, threshold, denoising = True, plot = True):
		data = np.log1p(np.abs(self.map))   # carte FFT déjà calculée

		if denoising:
			data = denoise_wavelet(data, method='BayesShrink', mode='hard')

		coordinates = peak_local_max(
				data,
				min_distance=mindistance_peaks,
				threshold_rel=threshold,
				exclude_border=False
			)    

		peak_qx = self.x[coordinates[:, 1]] # column → x 
		peak_qy = self.y[coordinates[:, 0]]
		peaks = np.column_stack((peak_qx, peak_qy))

		# --- Plot ---
		if plot:
			# extent_q = [self.x.min(), self.x.max(), self.y.min(), self.y.max()]

			# fig, ax = plt.subplots(figsize=(6, 5))
			# im = ax.imshow(data, cmap='viridis', origin='lower', aspect='auto', extent = extent_q)
			self.plot( aspect='auto', show = False, logscale=True)
			fig = plt.gcf()
			ax = plt.gca()
			ax.plot(peak_qx, peak_qy, 'r+', markersize=10, label="Peaks détectés")
			ax.legend()
			plt.show()  

		return peaks

# --------------------------------------------- FUNCTIONS --------------------------------------------- #

# theoretical computation of modes of hPhPs in hBN 10 or 11
def k_x_theoretical(thickness, n, new_freqs, epsilon_minus = -10000 , hbn_type = 11, plot=False):

	if hbn_type == 11:
		# fit part in h11BN for thickness 
		epsilon_xx = 5.32 * (1.+ (((1608.8)**2 -1359.8**2)/(1359.8**2 - new_freqs**2. - 1j * new_freqs* 2.1))) 
		epsilon_zz = 3.15 * (1.+ (((814)**2 -755.**2)/(755.**2 - new_freqs**2. - 1j * new_freqs* 1.)))

	if hbn_type == 10:
		# dielectric functions (example: h10BN)
		epsilon_xx = 5.1 * (1.+ (((1650)**2 -1394.5**2)/(1394.5**2 - new_freqs**2 - 1j * new_freqs*1.8)))
		epsilon_zz = 2.5 * (1.+ (((845)**2 -785.**2)/(785.**2 - new_freqs**2 - 1j * new_freqs* 1.)))

	# surrounding media
	#epsilon_minus = -10000 # gold
	#epsilon_minus = 1.17**2   # SiO2
	epsilon_plus = 1.0        # air

	# anisotropy factor
	phi = np.sqrt(-epsilon_xx/epsilon_zz + 0j)

	# reflection coefficients
	r_plus  = (epsilon_xx - 1j * epsilon_plus * phi)/(epsilon_xx + 1j * epsilon_plus * phi)
	r_minus = (epsilon_xx - 1j * epsilon_minus * phi)/(epsilon_xx + 1j * epsilon_minus * phi)

	# phases
	rho_plus  = np.angle(r_plus) * (1/np.pi)
	rho_minus = np.angle(r_minus) * (1/np.pi)

	# full complex kz (include imaginary term!)
	k_z_real = (np.pi / (2*thickness)) * (2*n + rho_plus + rho_minus)
	k_z_imag = (1j / (2*thickness)) * np.log(np.abs(r_plus) * np.abs(r_minus))
	k_z = k_z_real + k_z_imag

	# --- PROPER COMPLEX DIVISION ---
	# Instead of k_z/phi, do the full complex division:
	numerator   = (np.real(k_z)*np.real(phi) + np.imag(k_z)*np.imag(phi)) \
				+ 1j*(-np.real(k_z)*np.imag(phi) + np.imag(k_z)*np.real(phi))

	denominator = np.abs(phi)**2
	wavenumber_k_x = numerator / denominator

	# Convert from wavevector (rad/m) to spatial frequency (cycles/m)
	k_x_theoretical = wavenumber_k_x /(2*np.pi)

	if plot:
		plt.figure(figsize=(8,6))
		plt.plot(np.real(k_x_theoretical), new_freqs, label=f"t={(thickness*1e9):.1f} nm")
		plt.xlabel("Fringes Wavevector (m$^{-1}$)", fontsize=16)
		plt.ylabel("Wavenumber (cm$^{-1}$)", fontsize=16)
		plt.xticks(fontsize=18)
		plt.yticks(fontsize=18)
		plt.legend()
		plt.grid(True, linestyle="--", alpha=0.6)
		plt.tight_layout()
		plt.show()

	return k_x_theoretical

# Spacing between frignes peaks analysis
def peak_spacing_in_SNOM(
	snom_data,
	measure_position,
	pixel_range=[1],
	plot=False,
	saveindex=None,
	savedir="Figures",
	window_length=11,
	polyorder=3,
	highest_side="right",
	height_factor=0.05,
	return_peaks=False,
):
	"""
	Compute mean peak spacing (in µm) for SNOM line profiles.

	Parameters
	----------
	snom_data : object
		Must contain snom_data.x (µm), snom_data.y (cm⁻¹), snom_data.map[pixel, x].
	measure_position : tuple
		(xmin, xmax) in µm defining the window where peaks are considered.
	pixel_range : list
		List of pixel indices to analyze.
	highest_side : str
		"left" or "right". Indicates where the highest-amplitude peak is expected.
	height_factor : float
		Fraction of peak prominence relative to signal amplitude.
	return_peaks : bool
		If True, also return detected peak positions for debugging.

	Returns
	-------
	peak_distances : list of floats
		Mean peak spacing for each pixel (in µm).
	peak_positions (optional) : list of arrays
		x positions of detected peaks.
	"""

	if saveindex is None:
		saveindex = []

	peak_distances = []
	all_peak_positions = []

	x_vals = snom_data.x

	for i in pixel_range:
		y_vals = np.abs(snom_data.map[i, :])

		# Smooth signal
		smoothed = savgol_filter(y_vals, window_length=window_length, polyorder=polyorder)
		amplitude = np.max(smoothed) - np.min(smoothed)

		# Peak detection
		peaks, _ = find_peaks(smoothed, prominence=amplitude * height_factor)

		# Keep only peaks inside measurement window
		peaks_in_range = [p for p in peaks if measure_position[0] <= x_vals[p] <= measure_position[1]]
		all_peak_positions.append(x_vals[peaks_in_range])

		if len(peaks_in_range) < 3:
			peak_distances.append(np.nan)
			continue

		# Compute consecutive spacings
		diffs = np.diff(x_vals[peaks_in_range])  # spacing between consecutive peaks

		# Direction-aware averaging
		if highest_side == "left":
			# Highest peak on left → decay to the right
			if len(peaks_in_range) >= 4 and y_vals[peaks_in_range[-1]] < y_vals[peaks_in_range[-2]]:
				mean_spacing = np.mean(diffs[-3:-1])
			else:
				mean_spacing = np.mean(diffs[-2:])
		else:
			# Highest peak on right → decay to the left
			if len(peaks_in_range) >= 4 and y_vals[peaks_in_range[0]] < y_vals[peaks_in_range[1]]:
				mean_spacing = np.mean(diffs[:2])
			else:
				mean_spacing = np.mean(diffs[-2:])

		peak_distances.append(mean_spacing)

		# Optional plotting
		if plot:
			plt.figure(figsize=(7,4))
			plt.plot(x_vals, y_vals, label="Raw", alpha=0.6)
			plt.plot(x_vals, smoothed, label=f"Smoothed (window={window_length})", color="grey")
			plt.plot(x_vals[peaks], smoothed[peaks], "o", label="Detected peaks")
			plt.xlim(measure_position)
			plt.xlabel("x (µm)")
			plt.ylabel("I (a.u.)")
			plt.title(f"Pixel {i} — {snom_data.y[i]} cm-1")
			plt.grid(True, linestyle="--", alpha=0.5)
			plt.legend()

			if i in saveindex:
				os.makedirs(savedir, exist_ok=True)
				fname = f"{i}_{snom_data.y[i]}_highest_{highest_side}".replace(" ", "_")
				plt.savefig(os.path.join(savedir, f"{fname}.png"), dpi=300, transparent=True)

			plt.show()

	if return_peaks:
		return peak_distances, all_peak_positions
	return peak_distances

def plot_peak_spacing(
	peak_distances,
	snom_data,
	good_index_list,
	thickness,
	n,
	hbn_type,
	epsilon_minus,
	plot=True,
	save=False,
	polariton_type="reflected",
):
	"""
	Convert peak spacing (µm) into spatial frequency q (µm⁻¹),
	compare with theoretical kx dispersion, and optionally plot.

	Returns
	-------
	kx_theory_plot : array
		Theoretical wavevector (µm⁻¹) for plotting.
	freqs : array
		Frequencies (cm⁻¹) corresponding to the selected pixels.
	"""

	freqs = np.array(snom_data.y[good_index_list])
	peak_distances = np.array(peak_distances)

	# Convert spacing → spatial frequency q = 1/Δx
	q_exp = 1 / peak_distances  # µm⁻¹

	# Compute theoretical kx
	kx = k_x_theoretical(thickness, n, freqs, hbn_type=hbn_type,
						epsilon_minus=epsilon_minus, plot=False)
	kx = np.real(kx) * 1e-6  # convert from m⁻¹ to µm⁻¹

	if polariton_type == "reflected":
		kx_plot = 2 * kx  # q_fringes = 2 Re(kx)
	else:
		kx_plot = kx

	if plot:
		# Experimental spacing vs frequency
		plt.figure(figsize=(7,5))
		plt.plot(peak_distances, freqs, "o")
		plt.xlabel("Mean peak spacing delta x (um)")
		plt.ylabel("Wavenumber (cm-1)")
		plt.grid(True, linestyle="--", alpha=0.5)
		plt.tight_layout()
		plt.show()

		# Dispersion plot
		plt.figure(figsize=(7,5))
		plt.plot(q_exp, freqs, "o", label="Experimental q")
		plt.plot(kx_plot, freqs, "-", label=f"Theory (t={thickness*1e9:.1f} nm)")
		plt.xlabel("Wavevector q (um-1)")
		plt.ylabel("Wavenumber (cm-1)")
		plt.legend()
		plt.grid(True, linestyle="--", alpha=0.5)
		plt.tight_layout()
		plt.show()

	return kx_plot, freqs

# Fit of a single fringe
def savitzky_golay_smooth(y, window_length=11, polyorder=3):
	"""Apply Savitzky-Golay smoothing filter to data."""
	# Ensure window_length is odd and valid
	if window_length % 2 == 0:
		window_length += 1
	if window_length > len(y):
		window_length = len(y) if len(y) % 2 == 1 else len(y) - 1
	if polyorder >= window_length:
		polyorder = window_length - 1

	return savgol_filter(y, window_length=window_length, polyorder=polyorder)

def analyze_fringes(i, data, k_x_theoretical, measure_position, highest_side="right", plot=True, smooth=False, window_length=11, polyorder=3):

	def fit_fringes_data_prep(i, data, measure_position, highest_side="right", smooth=False, window_length=11, polyorder=3):
		"""Extract, slice, flip, and optionally smooth data for index i. Returns (x, y, y_original) if smooth=True, else (x, y, None)."""
		print(f"Raw y[{i}]: {data.y[i]}")
		x, y = data.x, (data.map[i, :])
		print(f"x: {x}", np.isnan(x).sum())

		# Select measurement window
		mask = (x >= measure_position[0]) & (x <= measure_position[1])
		x, y = x[mask], y[mask]

		# Flip and normalize
		if highest_side == "left":
			y = np.flip(y)
		else:
			y = y        
		x = x - x[0] + 1e-4
		
		# Apply Savitzky-Golay smoothing
		y_original = None
		if smooth:
			y_original = y.copy()  # Keep original for plotting
			y = savitzky_golay_smooth(y, window_length=window_length, polyorder=polyorder)
			print(f"Applied Savitzky-Golay smoothing: window_length={window_length}, polyorder={polyorder}")
		
		return x, y, y_original

	def fit_fringes(x, y, qr, qi, plot=True, y_original=None):
		"""Fit the fringes model and optionally plot results. Fits to y (smoothed), but plots both y_original and y if provided."""

		def fringes(x, A, B, C, loss, phi):
			return (A * np.sin(2*qr*x + phi) * np.exp(-2*qi*loss*x) / np.sqrt(x)
				+ B * np.sin(qr*x + phi) * np.exp(-qi*loss*x) / np.sqrt(x**3)
				+ C)

		# Estimate adaptive initial parameters from data
		y_real = np.real(y)
		print(f"y_real: {x}", np.isnan(x).sum())
		C_init = np.mean(y_real)  # Offset = mean
		amplitude = (np.max(y_real) - np.min(y_real)) / 2  # Peak-to-peak amplitude
		A_init = amplitude * np.sqrt(np.mean(x))  # Scale by sqrt(x) since model has 1/sqrt(x)
		B_init = amplitude * np.mean(x)**1.5  # Scale by x^(3/2) since model has 1/sqrt(x^3)
		loss_init = 1.0  # Default loss
		phi_init = 0.0  # Default phase
		
		# Adaptive bounds based on data range
		y_range = np.max(y_real) - np.min(y_real)
		A_bound = max(100., abs(A_init) * 10)
		B_bound = max(10000., abs(B_init) * 10)
		C_bound = max(10., abs(C_init) + y_range)
		
		print(f"Initial params: A={A_init:.2f}, B={B_init:.2f}, C={C_init:.2f}, loss={loss_init:.2f}, phi={phi_init:.2f}")

		try:
			popt, _ = curve_fit(
				fringes, x, y_real,
				p0=[A_init, B_init, C_init, loss_init, phi_init],
				bounds=([-A_bound, -B_bound, -C_bound, 0.01, -2*np.pi],
						[A_bound, B_bound, C_bound, 10., 2*np.pi]),
				maxfev=1_000_000
			)
		except RuntimeError as e:
			print(f"Fit failed: {e}. Using initial parameters as result.")
			popt = np.array([A_init, B_init, C_init, loss_init, phi_init])

		print(f"A:    {popt[0]:.1f}")
		print(f"B:    {popt[1]:.1f}")
		print(f"C:    {popt[2]:.1f}")
		print(f"loss: {popt[3]:.1f}")
		print(f"phi:  {popt[4]*180/np.pi:.1f}")

		print("ghello")
		if plot:
			y_fit = fringes(x, *popt)
			plt.figure(figsize=(10, 5))
			
			# Plot original noisy data if available
			if y_original is not None:
				plt.plot(x, np.real(y_original), '.', label='Original Data', alpha=0.3, color='gray')
				plt.plot(x, np.real(y), '.', label='Smoothed Data', alpha=0.7)
			else:
				plt.plot(x, np.real(y), '.', label='Data (Real)', alpha=0.5)
			
			plt.plot(x, np.real(y_fit), '-', label='Fit (Real)', linewidth=2, color='red')
			plt.xlabel("x")
			plt.ylabel("Re(y)")
			plt.legend()
			plt.tight_layout()
			plt.show()

		return popt

	def fit_snom_1D_fringes(x, y, plot=True, y_original=None, window_length=11, polyorder=3):
		"""Detect peaks, fit sqrt(x^3) model, and optionally plot results. Fits to y (smoothed), but plots both y_original and y if provided."""

		# Find peaks on smoothed data (use same smoothing as for fringes fitting)
		y_for_peaks = savitzky_golay_smooth(y, window_length=window_length, polyorder=polyorder)
		peaks, _ = find_peaks(y_for_peaks, height=0)

		def model(x, a, b):
			return a / np.sqrt(x) + b

		params, _ = curve_fit(model, x[peaks], np.real(y[peaks]), p0=[1, 0])
		a_fit, b_fit = params
		print(f"Fitted parameters: a = {a_fit:.3f}, b = {b_fit:.3f}")

		if plot:
			plt.figure(figsize=(8, 6))
			
			# Plot original noisy data if available
			if y_original is not None:
				plt.plot(x, np.real(y_original), '.', label="Original Data", alpha=0.3, color='gray')
				plt.plot(x, np.real(y), '.', label="Smoothed Data", alpha=0.7)
			else:
				plt.plot(x, np.real(y), '.', label="Data")
			
			plt.plot(x[peaks], np.real(y[peaks]), '.', label="Peaks", markersize=10)
			plt.plot(x, model(x, a_fit, b_fit), color="red", label=r"$1/\sqrt{x^3}$ Fit", linewidth=2)
			plt.xlabel("x")
			#plt.xlim([0.1,1])
			plt.ylim([min(np.real(y))-0.1*min(np.real(y)),max(np.real(y))+0.1*max(np.real(y))])
			plt.ylabel("y")
			plt.legend()
			plt.tight_layout()
			plt.show()

		return a_fit, b_fit

	"""Main wrapper: prepares data, fits fringes, and fits peaks model. Set smooth=True to reduce noise and compare original vs smoothed."""
	x, y, y_original = fit_fringes_data_prep(i, data, measure_position, highest_side, smooth=smooth, window_length=window_length, polyorder=polyorder)
	qr, qi = np.real(k_x_theoretical[i]), np.imag(k_x_theoretical[i])

	print("/n qr :",qr, " qi :", qi, "/n")

	popt_fringes = fit_fringes(x, y, qr, qi, plot=plot, y_original=y_original)
	popt_model = fit_snom_1D_fringes(x, y, plot=plot, y_original=y_original, window_length=window_length, polyorder=polyorder)

	return popt_fringes, popt_model

## 2D FFT

def compute_kx_radii(thickness, freq_cm1, modes=(0,1), epsilon_minus=2.3, hbn_type=10):
	"""
	Compute theoretical k_x radii (in µm⁻¹) for given modes.

	Parameters
	----------
	thickness : float
		hBN thickness in meters.
	freq_cm1 : float
		IR wavenumber (cm⁻¹).
	modes : tuple[int] or list[int]
		Mode indices to compute.
	epsilon_minus : float, optional
		Substrate dielectric constant.
	hbn_type : int, optional
		hBN type (10 or 11).

	Returns
	-------
	radii : list[float]
		Radii in µm⁻¹ for each mode.
	"""
	new_freqs = np.array([freq_cm1])
	radii = []
	for m in modes:
		kx = k_x_theoretical(
			thickness, m, new_freqs,
			epsilon_minus=epsilon_minus,
			hbn_type=hbn_type,
			plot=False,
		)
		kx_um = np.real(kx) * 1e-6  # cycles/µm
		radii.append(float(kx_um[0]))
	return radii

def plot_kx_circles(ax, thickness, freq_cm1, modes=(0,1), epsilon_minus=2.3,
					hbn_type=10, n_theta=500, colors=None, labels=None,
					plot_2k_mode=False):
	"""
	Overlay theoretical k_x mode circles on an existing axis.
	Returns radii and labels for legend handling.
	"""
	if colors is None:
		default_colors = ["red", "orange", "cyan", "magenta", "lime"]
		colors = [default_colors[i % len(default_colors)] for i in range(len(modes))]
	if labels is None:
		labels = [f"{m} mode" for m in modes]

	# --- Compute radii separately ---
	radii = compute_kx_radii(thickness, freq_cm1, modes, epsilon_minus, hbn_type)

	theta = np.linspace(0, 2*np.pi, n_theta)

	for r, color, label in zip(radii, colors, labels):
		x_circle = r * np.cos(theta)
		y_circle = r * np.sin(theta)
		ax.plot(x_circle, y_circle, "-", color=color, linewidth=2, label=label)
		if plot_2k_mode:
			ax.plot(2*x_circle, 2*y_circle, "--", color=color, linewidth=2, label=f"2k {label}")

	return radii, labels

def lattice_cste(peaks, plot=False):
	# 1) Unique pairwise distances
	def unique_axis_distances(points):
		X = points[:, 0]
		Y = points[:, 1]
		dx = X[None, :] - X[:, None]
		dy = Y[None, :] - Y[:, None]
		i, j = np.triu_indices(len(points), k=1)
		return dx[i, j], dy[i, j]

	# 2) Histogram + raw peaks
	def histogram_peaks(distances, bins_nb=100, prominence=20, plot=False, label=""):
		fig, ax = plt.subplots(figsize=(6,4))
		n, bins, patches = ax.hist(distances, bins=bins_nb, color='steelblue', edgecolor='black')

		peak_idx, _ = find_peaks(n, prominence=prominence)
		peak_pos = 0.5 * (bins[peak_idx] + bins[peak_idx+1])

		for idx in peak_idx:
			patches[idx].set_edgecolor('red')

		ax.set_title(f"Histogram peaks ({label})")
		ax.set_xlabel("Distance")
		ax.set_ylabel("Count")

		if plot:
			plt.tight_layout()
			plt.show()
		else:
			plt.close(fig)

		return np.sort(peak_pos)

	# 3) Merge double peaks
	def merge_close_peaks(positions, tol_factor=0.4):
		if len(positions) < 2:
			return positions

		spacings = np.diff(positions)
		median_spacing = np.median(spacings) if len(spacings) > 0 else 0
		tol = tol_factor * median_spacing if median_spacing > 0 else 0

		merged = [positions[0]]
		for p in positions[1:]:
			if abs(p - merged[-1]) < tol:
				merged[-1] = 0.5 * (merged[-1] + p)
			else:
				merged.append(p)

		return np.array(merged)
	
	# 4) Detect missing harmonics
	def correct_missing_harmonics(positions):
		"""
		Detect large gaps and shift indices above the gap by +1.
		"""
		if len(positions) < 3:
			return np.arange(len(positions))

		spacings = np.diff(positions)
		median_spacing = np.median(spacings)

		# A missing harmonic creates a spacing ≈ 2× the median
		missing = np.where(spacings > 1.5 * median_spacing)[0]

		idx = np.arange(len(positions))

		if len(missing) == 0:
			return idx

		# Only handle the first missing harmonic (simple, robust)
		gap = missing[0]

		# Shift all indices above the gap by +1
		idx[gap+1:] += 1

		return idx

	# 5) Fit harmonic index → position
	def estimate_period(dist, label=""):
		raw_peaks = histogram_peaks(dist, plot=plot, label=label)
		merged_peaks = merge_close_peaks(raw_peaks)

		if len(merged_peaks) < 2:
			return np.nan, np.nan

		# Assign indices with missing-peak correction
		indices = correct_missing_harmonics(merged_peaks)

		x = indices
		y = merged_peaks

		slope, intercept, r, p, stderr = linregress(x, y)

		# Diagnostic plot
		if plot:
			fig, ax = plt.subplots(figsize=(6,4))
			ax.scatter(x, y, color='blue', label='Peaks')
			xx = np.linspace(x.min(), x.max(), 200)
			ax.plot(xx, intercept + slope*xx, color='red', label=f'Fit, slope = {slope:.3f}')
			ax.set_title(f"Linear fit ({label})")
			ax.set_xlabel("Harmonic index (corrected)")
			ax.set_ylabel("Peak position")
			ax.legend()
			plt.tight_layout()
			plt.show()

		return slope, stderr

	# 6) Apply to x and y axes

	dist_x, dist_y = unique_axis_distances(peaks)

	lattice_x, err_x = estimate_period(dist_x, label="x")
	lattice_y, err_y = estimate_period(dist_y, label="y")

	return {
		"xaxis": [lattice_x, err_x],
		"yaxis": [lattice_y, err_y]
	}

## Masks for FFT filtering

def single_gaussian(shape, sigma, center=(0.0, 0.0), dx=1.0, dy=1.0):
	"""
	Gaussian spot in reciprocal space, defined in physical units.
	"""
	ny, nx = shape
	qx_max = 1 / (2 * dx)
	qy_max = 1 / (2 * dy)
	qx = np.linspace(-qx_max, qx_max, nx)
	qy = np.linspace(-qy_max, qy_max, ny)
	Qy, Qx = np.meshgrid(qy, qx, indexing="ij")

	r2 = (Qx - center[0])**2 + (Qy - center[1])**2
	mask = np.exp(-r2 / (2 * sigma**2))
	return mask

def gaussian_spots_filter(shape, coords, sigma, dx=1.0, dy=1.0):
	"""
	Sum of Gaussian spots at given reciprocal-space coordinates.
	"""
	mask = np.zeros(shape)
	for alpha in coords:
		mask += single_gaussian(shape, sigma, center=alpha, dx=dx, dy=dy)
	return mask

def gaussian_ring(shape, radius, sigma, dx=1.0, dy=1.0):
	"""
	Gaussian ring in reciprocal space, defined in physical units.
	"""
	ny, nx = shape
	qx_max = 1 / (2 * dx)
	qy_max = 1 / (2 * dy)
	qx = np.linspace(-qx_max, qx_max, nx)
	qy = np.linspace(-qy_max, qy_max, ny)
	Qy, Qx = np.meshgrid(qy, qx, indexing="ij")

	dist = np.sqrt(Qx**2 + Qy**2)
	mask = np.exp(-((dist - radius)**2) / (2 * sigma**2))
	return mask

def gaussian_ringss_filter(shape, radiuss, sigma, dx=1.0, dy=1.0):
	"""
	Sum of Gaussian spots at given reciprocal-space coordinates.
	"""
	mask = np.zeros(shape)
	for alpha in radiuss:
		mask += gaussian_ring(shape, alpha, sigma, dx=dx, dy=dy)
	return mask

def lattice_spot_grid_filter(shape, peaks, sigma, interval_x, interval_y,  plot=False, dx = 1.0, dy = 1.0):
	# def min_pairwise_distance(points):
	#     """
	#     Compute the smallest Euclidean distance between all pairs of points.
	#     """
	#     # Compute pairwise differences using broadcasting
	#     diff = points[:, None, :] - points[None, :, :]
		
	#     # Compute squared distances
	#     dist_sq = np.sum(diff**2, axis=-1)
		
	#     # Remove zero distances on the diagonal
	#     np.fill_diagonal(dist_sq, np.inf)
		
	#     # Return the minimum distance
	#     return np.sqrt(np.min(dist_sq))

	def closest_to_origin(coords):
		coords = np.asarray(coords)
		d2 = coords[:,0]**2 + coords[:,1]**2
		return coords[np.argmin(d2)]

	def symmetric_grid(center, dx, dy, interval_x, interval_y):
		"""
		center     : (cx, cy)
		dx, dy     : spacing in x and y
		interval_x : half-width to cover in x direction
		interval_y : half-height to cover in y direction
		"""
		cx, cy = center

		# Number of steps needed to cover the interval
		Nx = int(np.ceil(interval_x / dx))
		Ny = int(np.ceil(interval_y / dy))

		# Generate symmetric coordinate vectors
		xs = cx + dx * np.arange(-Nx, Nx + 1)
		ys = cy + dy * np.arange(-Ny, Ny + 1)

		# Build full grid
		grid_x, grid_y = np.meshgrid(xs, ys)
		points = np.column_stack([grid_x.ravel(), grid_y.ravel()])

		return points

	lattice_constants = lattice_cste(peaks, plot)
	# minpd = min_pairwise_distance(peaks)
	# print(minpd)
	
	closest = closest_to_origin(peaks)

	pts = symmetric_grid(closest, lattice_constants["xaxis"][0], lattice_constants["yaxis"][0], interval_x, interval_y)
	# pts = symmetric_grid(closest, minpd, minpd, interval_x, interval_y)

	mask = gaussian_spots_filter(shape, pts, sigma, dx=dx, dy = dy)

	return mask

# --------------------------------------------- TEST CODE --------------------------------------------- #

if __name__ == '__main__':

	Data_path = "T:\\dataSNOM\\Cavity QED\\Patterned hBN\\PBN06\\High_resolution_scans_DoS"

	i = 0

	example = snom(load_folder(Data_path,i))	# select scan
	example.print_details()						# select channel

	example.plot()
