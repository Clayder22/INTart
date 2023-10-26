import pandas as pd
import seaborn as sns
import numpy as np
import sys
import matplotlib.pyplot as plt

!wget https://raw.githubusercontent.com/ezgisubasi/breast-cancer-gene-expression/main/data/METABRIC_RNA_Mutation.csv
df = pd.read_csv("METABRIC_RNA_Mutation.csv",low_memory=False)
df.head(10)
df.info()

#El Dominio es todos los tipos de datos por columna y la variedad de datos sin repitencia
for columna in df.columns:
  dominio = df[columna].unique()
  print(f'El dominio es {columna}: {dominio}')

print (df['age_at_diagnosis'].dtypes) # Muestra de que tipo es cada variable, resumido

for column in df.columns: # Muestra de que tipo es cada variable, total
    print(column, "|es de tipo:", df[column].dtype)


conteos = df['cellularity'].value_counts()
etiqueta = conteos.index
valores = conteos
plt.pie(valores, labels=etiqueta, autopct='%1.2f%%')             #Una manera de graficar
plt.title('cellularity graph')
plt.axis('equal')
plt.show()

conteos = df['cancer_type'].value_counts()
etiqueta = conteos.index
valores = conteos                                            #Otra manera
plt.bar(etiqueta, valores)
plt.title('cancer_type graph')
plt.xlabel('cancer_type')
plt.ylabel('Frecuencia')
plt.show()

#   ♥1.
df._get_numeric_data().head(10)

#   ♥2.
df_sinCasosAusentes = df.dropna() #En caso de que se borren todos los casos es porque hay una columna con todo nan, hay que borrar esa columna.
# de.dropna('columna')
num_filas, num_columnas = df_sinCasosAusentes.shape
print("Cantidad de datos que  quedan:", num_filas, "filas(casos que quedan) y", num_columnas, "columnas")
print("Total de campos que quedan:",num_filas*num_columnas)

#   ♥3.
media=np.mean(df_sinCasosAusentes)
desviacion=np.std(df_sinCasosAusentes)
print("____Media de las campos____")
print( media)
print("_____desviacion de las campos____")
print(desviacion)

#   ♥5.
prop=df_sinCasosAusentes.apply(lambda x: x.value_counts(normalize=True).get(0,0))
prop.plot()
plt.show()         #Claro que existe un devalance de las clases, cada clase refleja una caracteristica distinta.


#   ♥6.
# primero debo escoger los valores numericos y realizarlos con ellos la correlacion
#df_correlacion = df_sinCasosAusentes.select_dtypes(include = 'number') #lamentablemente, esto demora mucho

new_clinic_data = df[df.columns[:15]] # por eso, creo una mini base para hacerlo solo con estos valores
df_correlacion = new_clinic_data.select_dtypes(include = 'number')

corr_matrix = df_correlacion.corr()
f, ax = plt.subplots(figsize=(16, 12))
sns.heatmap(corr_matrix, annot=True)
plt.show()