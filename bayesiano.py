#Clase que representa un clasificador bayesiano definido por un conjunto de entrenamiento
#definido como un vector de identificadores y una matriz de datos correspondientes (preprocesados).

import numpy as np
from collections import Counter

class Bayesiano:

	def bayes(self,m,cov,datos):
		dat=np.array(datos)
		resta=np.subtract(dat,m)

		resultado=-0.5*np.dot((np.dot(resta.T,np.linalg.inv(cov))),resta)-(0.5*np.log(np.linalg.det(cov)))+(np.log(1.0/(len(self.clases))))
		return resultado

	def __init__(self,entren,ident):
		#Creo una matriz que corresponde cada identificador con su vector de datos y la ordeno
		matriz=[[ident[i]]+entren[i] for i in range(len(entren))]
		matriz.sort()
		#Cuento la cantidad de elementos de cada clase (con un mismo identificador)
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
		self.covarianzas=[np.cov(i,rowvar=0) for i in separated]

		np.set_printoptions(suppress=True)

		#print self.medias

	def clasificar(self,data,ident):

		#Creo la matriz de densidades y otras variables
		densidades=np.zeros((len(data),len(self.clases)))
		aciertos=0.0
		confusion=[np.zeros(len(self.clases)) for i in range(len(self.clases))]

		#Clasifico los datos
		for i in range(len(data)):
			for j in range(len(self.clases)):
				densidades[i][j]=self.bayes(self.medias[j],self.covarianzas[j],data[i])
			indexdist=np.nonzero(densidades[i]==max(densidades[i]))[0][0]
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

	def getCentroides(self):
		return self.medias
