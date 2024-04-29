import tensorflow as tf
from modules.textModel.textModel import TextModel
from modules.imageModel.imageModel import ImageModel

tm = TextModel()
im = ImageModel()

while True:
    initial_msg = input(
        f"\nWould you like to import a file? If yes, type 'csv' for CSV file or 'img' for image file. Enter 'n' to manually input data or 'q' to quit: \n")
    
    if initial_msg.lower() == 'csv':
        list = None
        list = tm.csv_input()
        flower_names = tm.predict(list)
        print(flower_names)
    elif initial_msg.lower() == 'img':
        img = im.img_input()
        prediction = im.load_and_predict(img)
        print(prediction)
    elif initial_msg.lower() == 'n':
        user_input = input(
            "\nEnter the sepal length (cm), sepal width (cm), petal length (cm), and petal width (cm) for each flower, separated by spaces.\n")
        x_new = [float(x) for x in user_input.split()]
        
        if len(x_new) % 4 == 0:
            # Split the list into sublists of 4 numbers each
            x_new = [x_new[i:i+4] for i in range(0, len(x_new), 4)]

            prediction = tm.predict(x_new)
            print(prediction[0])
    else:
        print("\nGoodbye!\n")
        break