# Se calcula el promedio de la temperatura
# promedio_temperatura = tmax+tmin/2


def prom_temp(tmax, tmin):
  '''
  prom_temp: Calcular la Temperatura Media Anual 
  Variables de entrada
  tmax:  Temperatura máxima anual 
  tmin: Temperatura mínima anual'''

  import pandas as pd
  promedio = (tmax+tmin)/2
  promedio = pd.DataFrame(promedio)
  promedio = promedio.reset_index()
  return promedio 


# Para graficar el ráster

'''
  DEM: Graficar el Modelo Digital de Elevación (DEM)
  Variables de entrada
  raster: Modelo Digital de Elevación en formato ráster'''

def DEM(raster):
  import numpy as np
  Nonvalue = raster.GetRasterBand(1).GetNoDataValue()
  Array = raster.GetRasterBand(1).ReadAsArray().flatten().astype(np.float64)
  NanValues = np.where(Array == Nonvalue)[0]
  cP      = np.arange(0, raster.RasterXSize*raster.RasterYSize)
  cPP       = np.delete(cP, NanValues, axis=0)
  XX = np.delete(Array, NanValues, axis=0)
  temp = Array.copy()
  temp[temp==Nonvalue] = None
  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(10,10))
  plt.imshow(temp.reshape((raster.RasterYSize,raster.RasterXSize)), aspect='auto', cmap='jet')
  plt.colorbar()
  
  def regresion(altitud, temperatura):

  '''
  regresion: Calcular la regresión lineal entre los valores 
  de altitud y temperatura media anual
  Variables de entrada
  altitud:  Altura de las estaciones (msnm)
  temperatura: Temperatura media anual'''

  from sklearn import linear_model
  regresion = linear_model.LinearRegression()
  altitud = altitud.values.reshape(-1,1)
  modelo = regresion.fit(altitud, temperatura)
  # y = mx+b
  m = regresion.coef_
  b = regresion.intercept_
  label = r' Y = %0.4f*X %+ 0.4f '%(m,b)
  prediccion = modelo.predict(altitud)

  
  import matplotlib.pyplot as plt
  figura = plt.scatter(altitud, temperatura)
  plt.scatter(altitud, modelo.predict(altitud), color='red')
  plt.plot(altitud, modelo.predict(altitud), color='black', label=label)
  plt.xlabel ('Altitud (msnm)', fontsize= 15)
  plt.ylabel('Temperatura (°C)', fontsize= 15)
  plt.grid()
  plt.legend()

  return figura, print('La pendiente es de '+ str(m)+ 'y la intersección de '+ str(b))


def b(altitud, temperatura):
  '''
  b: Calcular parámetro b (intersección con el eje y) de la regresión lineal
  Variables de entrada
  altitud:  Altura de las estaciones (msnm)
  temperatura: Temperatura media anual'''

  from sklearn import linear_model
  regresion = linear_model.LinearRegression()
  altitud = altitud.values.reshape(-1,1)
  modelo = regresion.fit(altitud, temperatura)
  # y = mx+b
  b = regresion.intercept_
  return b


def Isotermas(m, XX, b): # calcula los valores de las isotermas 

  '''
  Isotermas: Calcular los valores de isotermas
  Variables de entrada
  m: pendiente de la regresión lineal
  XX : valores del DEM (Recuerde eliminar los valores nulos)
  b : intersección de la regresión lineal'''

  Isotermas = m*XX + b
  return Isotermas


def MapaIsotermas(raster, Isotermas, fn, nombre):

  '''
  Isotermas: Calcular el mapa de isotermas y exportarlo para ser visualizado en un software GIS
  Variables de entrada
  raster: DEM
  Isotermas : valores de las isotermas
  fn : Ruta del mapa de isotermas
  nombre: Nombre del mapa de isotermas
  Hecho por: Paul Goyes (goyes.yesid@gmail.com),
             Ana Mantilla (anagmd2019@gmail.com)
             Manuel Daza (manedaza12@gmail.com)'''

  import gdal
  import numpy as np

  Nonvalue = raster.GetRasterBand(1).GetNoDataValue()
  Array = raster.GetRasterBand(1).ReadAsArray().flatten().astype(np.float64)
  NanValues = np.where(Array == Nonvalue)[0]
  cP      = np.arange(0, raster.RasterXSize*raster.RasterYSize)
  cPP       = np.delete(cP, NanValues, axis=0)
  XX = np.delete(Array, NanValues, axis=0)
  
  driver = raster.GetDriver
  col = raster.RasterXSize
  rows  = raster.RasterYSize
  nelem = col*rows

  Rasterdataarray = np.zeros((rows,col)).flatten()
  for i in range(cPP.shape[0]):
    Rasterdataarray[cPP[i]]=Isotermas[i]
  for i in range(NanValues.shape[0]):
    Rasterdataarray[NanValues[i]]=Nonvalue

  driver = raster.GetDriver()
  Rasterout = driver.Create(fn + nombre + '.tif', col, rows, 1, gdal.GDT_Float32)
  Rasterout.SetGeoTransform(raster.GetGeoTransform())
  Rasterout.SetProjection(raster.GetProjection())
  Rasterout.GetRasterBand(1).WriteArray(Rasterdataarray.reshape(rows,col))
  Rasterout.GetRasterBand(1).SetNoDataValue(Nonvalue)
  Rasterout = None
  del Rasterout
  temp = Rasterdataarray.copy()
  temp[temp==Nonvalue] = None
  import matplotlib.pyplot as plt
  fig = plt.figure(figsize=(20,9))
  plt.subplot(131)
  plt.imshow(temp.reshape((raster.RasterYSize,raster.RasterXSize)), aspect='auto',cmap='jet')
  plt.colorbar()
  plt.title(nombre, fontsize=15)


