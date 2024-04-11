from model import knn, iris

while True:
    # Get user input
    user_input = input(
        "Enter the sepal length, sepal width, petal length, and petal width \nfor each flower, separated by spaces. Enter 'q' to quit: ")

    if user_input.lower() == 'q' or user_input.lower() == 'quit':
        print("Goodbye!")
        break

    # Split the input string into a list of strings, then convert each string to a float
    x_new = [float(x) for x in user_input.split()]

    if len(x_new) % 4 == 0:

        # Split the list into sublists of 4 numbers each
        x_new = [x_new[i:i+4] for i in range(0, len(x_new), 4)]

        flower_names = []
        for x in x_new:
            prediction = knn.predict([x])
            flower_name = iris['target_names'][prediction]
            flower_names.append(flower_name[0])
        print('lol')
        print(f"\nBased on your input, these are the types of the iris flowers: {
              ', '.join(flower_names)}\n")