import numpy as np
import DIAToolbox as dia
import matplotlib.pyplot as plt
import sys
import scipy

def displayImage(image, name):
	
	plt.title('subject0' + str(i) + '.' + suffix[j])
	plt.imshow(img,cmap = cm.Greys_r, aspect='auto')
	plt.show()

def PCATrans(img):
	
	mean = np.sum(img, axis=0)/(img.shape[0])
	
	resImage = 1.0*img - 1.0*mean
	
	imgCov = dia.varCorrMatrix(img)[0]
	
	#Create a list containing the number of faces
	bands = []
	
	for i in range(img.shape[0]):
		
		bands.append(i + 1)
	
	#Compute the eigenvalues and eigenvectors of the covariance matrix
	covEigVal, covEigVec = np.linalg.eig(imgCov)
	
	#Output the eigenvalues, eigenvectors and %variance values to the output file
	#outfile.write("EigenValues:\n" + str(covEigVal))
	#outfile.write("\n\n%Variance:\n" + str(covEigVal/np.sum(covEigVal)))
	#outfile.write("\n\nEigenVectors:\n" + str(covEigVec))

	#Create an array of eigenvalue, eigenvector and band number pairs
	pairs = [(covEigVal[i],covEigVec[i],bands[i]) for i in range(len(bands))]

	#Sort the eigenvalue pairs in descending order
	pairs.sort()
	pairs.reverse()
 
	#Generate the eigenvector transformation matrix (G)
	G = []

	for i in range(len(pairs)):
 
		G.append(pairs[i][1])

	#Convert the images into a linear image
	linearImage = resImage.reshape(img.shape[0],img.shape[1]*img.shape[2])

	G = np.array(G)
	#Compute the principle components of the image
	linearPC = np.dot(G.T,linearImage)
	PC = []
 
	#Transform from a linear image to a square image
	for i in range(len(linearPC)):
		
		temp = linearPC[i]/np.sqrt(np.sum(linearPC[i]*linearPC[i]))
		PC.append(temp.reshape(img.shape[1],img.shape[2]))

	return np.array(PC)

def computeTrainingData(args):
	
	#A list of the different file suffixes corresponding to the different face variants for each subject
	name = ['9326871','9336923','ahodki','anpage','doraj','ekavaz','jmedin','klclar','mbutle','moors','obeidn','pspliu']
	
	#For each person
	for i in name:
		
		imageStorage = []
		#We use the first 8 of 11 images as training data, and the final 3 as tests
		for j in range(1,7):
			
			img = dia.openImage('TrainingData/' + i + '.' + str(j) + '.jpg')[0]
			
			imageStorage.append(img)
		
		imageCube = np.array(imageStorage)

		PCImages = PCATrans(imageCube)
		
		#Save the component images to form the database of training data
		for l in range(len(PCImages)):
				
				
				dia.saveImage(np.uint8(255.0*(PCImages[l]-np.amin(PCImages[l]))/(np.amax(PCImages[l])-np.amin(PCImages[l]))), 'TrainingData/PrincipalComponents/' + i + '.' + str(l) + '.jpg')
				
				np.savetxt('TrainingData/PrincipalComponents/' + i + '.component' + str(l) + '.txt',PCImages[l])
		
		mean = np.sum(imageCube, axis=0)/(imageCube.shape[0])
	
		dia.saveImage(np.uint8(255.0*(mean - np.amin(mean))/(np.amax(mean) - np.amin(mean))), 'TrainingData/PrincipalComponents/' + str(i) + 'mean.jpg')
		
		mean = mean/np.sqrt(np.sum(mean*mean))
		np.savetxt('TrainingData/PrincipalComponents/' + i + 'mean.txt',mean)
		
def computeWeights():
	
	#A list of the different file suffixes corresponding to the different face variants for each subject
	name = ['9326871','9336923','ahodki','anpage','doraj','ekavaz','jmedin','klclar','mbutle','moors','obeidn','pspliu']
	
	weightDB = {}
	
	#For each person
	for i in name:
		
		imageStorage = []
		
		weights = []
		
		for j in range(1,5):
			
			img = np.loadtxt('TrainingData/PrincipalComponents/' + i + '.component' + str(j) + '.txt')

			imageStorage.append(img)
		
		imageStorage = np.array(imageStorage)
		
		mean = np.loadtxt('TrainingData/PrincipalComponents/' + i + 'mean.txt')
		
		for j in range(1,4):
			
			img = dia.openImage('TrainingData/' + i + '.' + str(j) + '.jpg')[0]
			
			img = img/np.sqrt(np.sum(img*img))
			
			temp = []
			
			for k in range(imageStorage.shape[0]):
				
				temp.append(imageStorage[k]*(img-mean))

			weights.append(np.array(temp))
			
		weights = np.array(weights)
		weightDB[i] = np.sum(weights, axis=0)/weights.shape[0]
		
		##np.savetxt('RegularPrincipalComponents/subject' + str(i) + 'weights.txt',avgWeight)		
	return weightDB
	
def computeFace(image, weightDB, keys):
	
	
	minDifference = 999999999999
	face = 9999
	image = image/np.sqrt(np.sum(image*image))
	#For each person
	for i in keys:
		
		imageStorage = []
		
		for j in range(1,5):
			
			img = np.loadtxt('TrainingData/PrincipalComponents/' + i + '.component' + str(j) + '.txt')
			
			imageStorage.append(img)
		
		imageStorage = np.array(imageStorage)
		
		mean = np.loadtxt('TrainingData/PrincipalComponents/' + i + 'mean.txt')
		
		weight = []
		
		for k in range(imageStorage.shape[0]):
			
			weight.append(imageStorage[k]*(image-mean))
			
		weight = np.array(weight)
		
		faceWeight = weightDB[i]
		
		difference = np.sqrt(np.sum(np.power(weight-faceWeight,2)))

		if difference < minDifference:
			
			minDifference = difference
			face = i
			
	return face

if __name__=='__main__':
	
	#A list of the different file suffixes corresponding to the different face variants for each subject
	name = ['9326871','9336923','ahodki','anpage','doraj','ekavaz','jmedin','klclar','mbutle','moors','obeidn','pspliu']
	
	computeTrainingData(sys.argv)
	weightDB = computeWeights()
	#For each person
	
	summ = 0
	total = 0
	
	for i in name:
		
		for j in range(7,9):
			
			image = dia.openImage('TrainingData/' + i + '.' + str(j) + '.jpg')[0]
				
			face = computeFace(image, weightDB, name)
			print 'Image Name: subject ' + i + '.' + str(j) + '\nIdentified as: subject0' + str(face) + '\n\n\n'
			#input()
			
			total += 1
			
			if face == i:
				
				summ += 1
				
	
	print 1.0*summ/total
