#Clase que representa un clasificador bayesiano definido por un conjunto de entrenamiento
#definido como un vector de identificadores y una matriz de datos correspondientes (preprocesados).

import numpy as np
from collections import Counter

class Euclideo:

	def getCentroides(self):
		return self.medias

	def getClases(self):
		return self.clases

	def distance(self,point,centroid):
		result=0
		for i in range(0,len(point)):
			result=result+(point[i]-centroid[i])**2
		return result

	def __init__(self,entren,ident):
		matriz=[[ident[i]]+entren[i] for i in range(len(entren))]
		matriz.sort()
		resultado=Counter([i[0] for i in matriz]).items()
		cuenta=dict(resultado)
		self.clases=cuenta.keys()
		num=cuenta.values()
		#Creo una lista con una lista vacia por cada clase
		separated=[0 for i in range(len(self.clases))]
		for i in range(len(separated)):
			separated[i]=[]

		#Relleno las listas en base al indice del identificador
		for i in range(len(entren)):
			index=self.clases.index(ident[i])
			separated[index].append(entren[i])

		#Calculo los parametros
		self.medias=[np.mean(i,axis=0) for i in separated]

		#print self.medias

	def clasificar(self,data,ident):

		#Creo la matriz de densidades y otras variables
		distancias=np.zeros((len(data),len(self.clases)))
		aciertos=0.0
		confusion=[np.zeros(len(self.clases)) for i in range(len(self.clases))]

		#Clasifico los datos
		for i in range(len(data)):
			for j in range(len(self.clases)):
				distancias[i][j]=self.distance(data[i],self.medias[j])
			indexdist=np.nonzero(distancias[i]==min(distancias[i]))[0][0]
			indexcor=self.clases.index(ident[i])
			if(indexdist==indexcor):
				aciertos=aciertos+1
			confusion[indexcor][indexdist]=confusion[indexcor][indexdist]+1

		#Print de los resultados
		print "Matriz de confusion: "
		print self.clases
		np.set_printoptions(suppress=True)
		for i in range(len(confusion)):
			print confusion[i]
		porcentaje=(aciertos/len(data))*100
		print "Porcentaje de aciertos: ", porcentaje
		return porcentaje