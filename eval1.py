from bayesiano import Bayesiano
from euclideo import Euclideo
import numpy as np
import sys

data=0
ident=0
dataset=0
evaluacion=0
tipo=0

def distance(c1,c2):
	result=0
	for i in range(0,len(c1)):
		result=result+(c1[i]-c2[i])**2
	return result

if(len(sys.argv)>1):
	tipo=sys.argv[1]
if(len(sys.argv)>2):
	dataset=sys.argv[2]
if(len(sys.argv)>3):
	evaluacion=sys.argv[3]

if dataset=='ocr':
	raw=np.load('datos_procesados.npy')
	ident=[i[9] for i in raw]
	data=[i[0:9] for i in raw]
elif(dataset=='wine'):
	data=np.genfromtxt('wine.data',delimiter=',',usecols=(1,2,3,4,5,6,7,8,9,10,11,12,13))
	ident=np.genfromtxt('wine.data',delimiter=',',usecols=(0))
elif dataset=='cancer':
	raw=np.genfromtxt('wdbc.data',delimiter=',')
	ident=np.genfromtxt('wdbc.data',dtype=None,delimiter=',',usecols=(1))
	data=[i[2:32] for i in raw]
elif dataset=='iris':
	data=np.genfromtxt("iris.data",delimiter=',',usecols=(0,1,2,3))
	ident=np.genfromtxt("iris.data",dtype=None,delimiter=',',usecols=(4))
else:
	print "Dataset no valido"
entren=data
idtr=ident

data=[i.tolist() for i in data]
entren=[i.tolist() for i in entren]

if(evaluacion=='-d1'):
	entren=data[0:500]
	idtr=ident[0:500]
	ident=ident[0:500]
	data=data[0:500]
if(evaluacion=='-d2'):
	entren=data[0:500]
	idtr=ident[0:500]
	ident=ident[500:]
	data=data[500:]

clasif=0
if tipo=='b':
	clasif=Bayesiano(entren,idtr)
else:
	clasif=Euclideo(entren,idtr)

"""medias=clasif.getCentroides()
distancias=np.zeros((10,10))
for i in range(len(medias)):
	for j in range(len(medias)):
		distancias[i][j]=distance(medias[i],medias[j])

print clasif.getClases()
print np.around(distancias,decimals=1)"""

clasif.clasificar(data,ident)

