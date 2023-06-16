# entry validation: https://stackoverflow.com/questions/4140437/interactively-validating-entry-widget-content-in-tkinter
# https://www.geeksforgeeks.org/python-string-replace/

# import all components
# from the tkinter library
from tkinter import ttk, Tk, IntVar, END, W, E

# import filedialog module
from tkinter import filedialog
from time import strftime

# Function for opening the
# file explorer window
def browseFolders(label):
	parent = filedialog.askdirectory(initialdir = "./",
										title = "Select a directory",
										)
	
	# Change label contents
	label.configure(text=parent)

def time():
	string = strftime("%y-%m-%d_%H-%M")
	if custom_time.get() == 0:
		entry_time.configure(state='normal')
		entry_time.delete(0, END)
		entry_time.insert(0, string)
		entry_time.configure(state='disabled')
		entry_time.after(100, time)
	else:
		entry_time.configure(state='normal')
		entry_time.after(100, time)

def temp():
	string = str(15.5)
	if custom_temp.get() == 0:
		entry_temp.configure(state='normal')
		entry_temp.delete(0, END)
		entry_temp.insert(0, string)
		entry_temp.configure(state='disabled')
		entry_temp.after(100, temp)
	else:
		entry_temp.configure(state='normal')
		entry_temp.after(100, temp)

def relh():
	string = str(55.5)
	if custom_relh.get() == 0:
		entry_relh.configure(state='normal')
		entry_relh.delete(0, END)
		entry_relh.insert(0, string)
		entry_relh.configure(state='disabled')
		entry_relh.after(100, relh)
	else:
		entry_relh.configure(state='normal')
		entry_relh.after(100, relh)	

def atmp():
	string = str(1013)
	if calc_imp.get() == 0:
		entry_atmp.configure(state='normal')
		entry_atmp.delete(0, END)
		entry_atmp.insert(0, string)
		entry_atmp.configure(state='disabled')
		entry_atmp.after(100, atmp)
	else:
		entry_atmp.configure(state='normal')
		entry_atmp.after(100, atmp)																								
# Create the root window
window = Tk()

# Set window title
window.title('Impedance Tube measurement - Measurement')

# Set window size
# window.geometry("500x500")

#Set window background color
# window.config(background = "white")

s = ttk.Style()
s.configure('my.TButton', font=("Ubuntu", 14, "bold"))
# Create a File Explorer label
label_project_folder = ttk.Label(window,
					text = "Project Folder:",
					padding=5,
					foreground = "black")
label_folder_explorer = ttk.Label(window,
							text = "--- No parent directory chosen ---",
							width = 50, padding=5,
							foreground = "blue")
	
button_explore = ttk.Button(window,
						text = "Browse Folders",
						command = lambda: browseFolders(label_folder_explorer)
						)

label_variant = ttk.Label(window,
					text = "Variant:",
					padding=5,
					foreground = "black")
entry_variant = ttk.Entry(window,
					width=50,)

label_time = ttk.Label(window,
					text='Timestamp:')
entry_time = ttk.Entry(window, 
					width=50,
					state="disabled")

custom_time = IntVar()
check_custom_time = ttk.Checkbutton(window,
					text='Custom',
					variable=custom_time)

label_temp = ttk.Label(window,
					text='Temperature [degC]:',
					anchor='w')
entry_temp = ttk.Entry(window, 
					width=50,
					state="disabled")

custom_temp = IntVar()
check_custom_temp = ttk.Checkbutton(window,
					text='Custom',
					variable=custom_temp)

label_relh = ttk.Label(window,
					text='Relative Humidity [%]:')
entry_relh = ttk.Entry(window, 
					width=50,
					state="disabled")

custom_relh = IntVar()
check_custom_relh = ttk.Checkbutton(window,
					text='Custom',
					variable=custom_relh)

calc_imp = IntVar()
check_calc_imp = ttk.Checkbutton(window,
					text='Calculate impedance',
					variable=calc_imp)
label_atmp = ttk.Label(window,
					text='Atmospheric pressure [hPa]:')
entry_atmp = ttk.Entry(window, 
					width=50,
					state="disabled")

label_calibration_folder = ttk.Label(window,
					text = "Calibration Folder:",
					padding=5,
					foreground = "black")
label_folder_explorer_cal = ttk.Label(window,
							text = "--- No parent directory chosen ---",
							width = 50, padding=5,
							foreground = "blue")
	
button_explore_cal = ttk.Button(window,
						text = "Browse Folders",
						command = lambda: browseFolders(label_folder_explorer_cal))

action_frame = ttk.Frame(window)
button_measure = ttk.Button(action_frame,
						text = "Measure",
						padding=[25, 10],
						style='my.TButton'
						# command = lambda: browseFolders(label_folder_explorer_cal)
						)
button_export = ttk.Button(action_frame,
						text = "Export",
						padding=[25, 10],
						style='my.TButton'
						# command = lambda: browseFolders(label_folder_explorer_cal)
						)
button_discard = ttk.Button(action_frame,
						text = "Discard",
						padding=[25, 10],
						style='my.TButton'
						# command = lambda: browseFolders(label_folder_explorer_cal)
						)
					
time()
temp()
relh()
atmp()
# Grid method is chosen for placing
# the widgets at respective positions
# in a table like structure by
# specifying rows and columns
label_project_folder.grid(column = 1, row = 1, sticky= E)
label_folder_explorer.grid(column = 2, row = 1)

button_explore.grid(column = 3, row = 1)

label_variant.grid(column=1, row=3, sticky= E)
entry_variant.grid(column=2, row=3 )
label_time.grid(column=1, row=4, sticky= E)
entry_time.grid(column=2, row=4)
check_custom_time.grid(column=3, row=4, sticky= W)

label_temp.grid(column=1, row=5, sticky= E)
entry_temp.grid(column=2, row=5)
check_custom_temp.grid(column=3, row=5, sticky= W)

label_relh.grid(column=1, row=6, sticky= E)
entry_relh.grid(column=2, row=6)
check_custom_relh.grid(column=3, row=6, sticky= W)

check_calc_imp.grid(column=2, row=7, sticky= W)
label_atmp.grid(column=1, row=8, sticky= E)
entry_atmp.grid(column=2, row=8)

label_calibration_folder.grid(column = 1, row = 9, sticky= E)
label_folder_explorer_cal.grid(column = 2, row = 9)
button_explore_cal.grid(column = 3, row = 9)

action_frame.grid(column = 1, row = 10, columnspan=3)
button_measure.grid(column = 1, row = 1)
button_export.grid(column = 2, row = 1)
button_discard.grid(column = 3, row = 1)
# Let the window wait for any events
window.mainloop()
