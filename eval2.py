from bayesiano import Bayesiano
from euclideo import Euclideo
import numpy as np
import sys

data=0
ident=0
tipo=0

if(len(sys.argv)>1):
	tipo=sys.argv[1]

raw=np.load('datos_procesados.npy')
ident=[i[9] for i in raw]
data=[i[0:9] for i in raw]

entren=data[0:500]
idtr=ident[0:500]

data=[i.tolist() for i in data]
entren=[i.tolist() for i in entren]

chunks=[]
idchunk=[]
for i in range(10):
	chunks.append([])
	idchunk.append([])
print chunks

for i in range(len(entren)):
	chunks[i/50].append(entren[i])
	idchunk[i/50].append(idtr[i])

media=0
for i in range(len(chunks)):
	datos=[]
	ids=[]
	clasif=0
	if tipo=='b':
		clasif=Bayesiano(entren,idtr)
	else:
		clasif=Euclideo(entren,idtr)
	for j in range(len(chunks)):
		if i!=j:
			datos=datos+chunks[j]
			ids=ids+idchunk[j]
	media+=clasif.clasificar(datos,ids)

print 'Porcentaje de acierto medio', media/10
	#print len(datos)
	#print len(ids)