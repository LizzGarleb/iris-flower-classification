from model import knn, iris

def knn_predict(x_new):
	flower_names = []
	for x in x_new:
		prediction = knn.predict([x])
		flower_name = iris['target_names'][prediction]
		flower_names.append(flower_name[0])
	print(f"\nBased on your input, these are the types of the iris flowers: {', '.join(flower_names)}\n")
