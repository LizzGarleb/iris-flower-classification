import pandas as pd
from functions import knn_predict
from tkinter import filedialog

while True:
	# Get user input
	initial_msg = input("Would you like to import a file? (y/n): ")
	if initial_msg.lower() == 'y':
		file_path = filedialog.askopenfilename()
		data = pd.read_csv(file_path)
		x_new = data.values.tolist()

		# Call the knn_predict function from functions.py
		knn_predict(x_new)

	else:
		user_input = input(
			"\nEnter the sepal length, sepal width, petal length, and petal width \nfor each flower, separated by spaces. Enter 'q' to quit: ")

		if user_input.lower() == 'q' or user_input.lower() == 'quit':
			print("\nGoodbye!\n")
			break

		# Split the input string into a list of strings, then convert each string to a float
		x_new = [float(x) for x in user_input.split()]

		if len(x_new) % 4 == 0:

			# Split the list into sublists of 4 numbers each
			x_new = [x_new[i:i+4] for i in range(0, len(x_new), 4)]

			# Call the knn_predict function from functions.py
			knn_predict(x_new)
