#Se utiliza el algoritmo Random Forest, creado por Leo Breiman en el año 2001
#Una libreria que implementa Rando Forest se llama  ranger
#La libreria esta implementada en lenguaje C y corre en paralelo, utiliza TODOS los nucleos del procesador
#Leo Breiman provenia de la estadistica y tenia "horror a los nulos", con lo cual el algoritmo necesita imputar nulos antes


#limpio la memoria
rm( list=ls() )  #Borro todos los objetos
gc()   #Garbage Collection

require("data.table")
require("ranger")
require("randomForest")  #solo se usa para imputar nulos

#Aqui se debe poner la carpeta de la computadora local
#setwd("~/buckets/b1/")  #Establezco el Working Directory. ojo respetar este folder para correr en cloud
setwd("C:\\DataMining_Economia_Finanzas\\datasets\\")  #Establezco el Working Directory


#cargo los datos donde entreno
#dataset  <- fread("./datasets/competencia2_2022.csv.gz", stringsAsFactors= TRUE)
dataset  <- fread("C:\\DataMining_Economia_Finanzas\\datasets\\competencia2_2022.csv.gz", stringsAsFactors= TRUE)

#imputo los nulos, ya que ranger no acepta nulos
#Leo Breiman, ¿por que le temias a los nulos?

# quito las columnas que tienen alta correlacion con otras o muchos nulos del dataset
#==================================================================================================================
xx1 <- c("mcomisiones_mantenimiento","Visa_mpagado", "ctarjeta_visa_transacciones","ctarjeta_visa_debitos_automaticos","mttarjeta_visa_debitos_automaticos","mtarjeta_visa_consumo","Master_Finiciomora","mtarjeta_master_consumo","Visa_msaldototal","Visa_msaldopesos","Master_mpagospesos","Master_mconsumospesos","ctarjeta_visa_transacciones","ctarjeta_visa_transacciones","ctarjeta_master_transacciones","Master_msaldopesos","Master_msaldopesos","cextraccion_autoservicio","Visa_msaldopesos","ccajas_transacciones","mtarjeta_master_consumo","ctarjeta_master","mtarjeta_master_consumo","Visa_msaldototal","Master_msaldototal","Master_msaldototal","mpayroll2","Visa_msaldopesos","Visa_msaldopesos","Visa_mconsumospesos","Visa_mconsumototal","mtarjeta_visa_consumo","ctarjeta_visa_transacciones","mtarjeta_master_consumo","mtarjeta_visa_consumo","tcallcenter","cprestamos_hipotecarios","mcaja_ahorro_dolares","Visa_msaldototal","Visa_msaldototal","Master_msaldopesos","ctarjeta_visa_transacciones","Visa_mpagospesos","mpasivos_margen","Visa_madelantopesos","ctarjeta_master_debitos_automaticos","mtarjeta_visa_consumo","Visa_mconsumospesos","Visa_mpagospesos","cextraccion_autoservicio","ctarjeta_debito_transacciones","mrentabilidad","cextraccion_autoservicio","ccomisiones_mantenimiento","ctarjeta_visa_debitos_automaticos","Master_msaldototal","mextraccion_autoservicio","mtarjeta_visa_consumo","catm_trx","cforex","cliente_antiguedad","cforex","ctarjeta_master_transacciones","catm_trx_other","Master_fechaalta","ctarjeta_visa_transacciones","mextraccion_autoservicio","Master_msaldodolares","mtarjeta_master_consumo","mtarjeta_master_consumo","cliente_antiguedad","mtarjeta_visa_consumo","mtarjeta_visa_consumo","Visa_msaldodolares","Master_madelantopesos","cforex_sell","Master_msaldototal","Master_mlimitecompra","Visa_msaldototal","mcomisiones","Visa_mconsumospesos","Master_mconsumospesos") # vector of columns you DON'T want
#xx2 <- c("Visa_mpagado","ctarjeta_visa_debitos_automaticos","Visa_cconsumos","Visa_cconsumos","mttarjeta_visa_debitos_automaticos","Visa_Finiciomora","Master_cconsumos","Visa_mpagospesos","Visa_mpagospesos","Master_mconsumototal","Master_mpagospesos","Visa_mconsumospesos","Visa_mconsumototal","mtarjeta_master_consumo","Master_mconsumototal","Master_mconsumospesos","matm","Visa_mpagominimo","ccajas_extracciones","Master_mpagospesos","Master_status","Master_msaldopesos","Visa_mpagominimo","Master_mconsumototal","Master_mconsumospesos","cpayroll2_trx","Visa_mconsumototal","Visa_mconsumospesos","Visa_cconsumos","Visa_cconsumos","Visa_msaldopesos","Visa_mpagospesos","Master_msaldototal","Visa_msaldototal","ccallcenter_transacciones","mprestamos_hipotecarios","mcuentas_saldo","Visa_mconsumototal","Visa_mconsumospesos","Master_mpagominimo","mtarjeta_visa_consumo","Visa_cconsumos","mcaja_ahorro","Visa_cadelantosefectivo","mttarjeta_master_debitos_automaticos","Visa_cconsumos","Visa_mpagospesos","Visa_mconsumototal","mextraccion_autoservicio","mautoservicio","mrentabilidad_annual","catm_trx","mcomisiones_mantenimiento","mttarjeta_visa_debitos_automaticos","Master_mpagominimo","catm_trx","Visa_mpagospesos","matm","mforex_sell","Master_fechaalta","cforex_sell","Master_cconsumos","matm_other","Visa_fechaalta","Visa_cconsumos","matm","Master_mconsumosdolares","Master_mconsumototal","Master_mconsumospesos","Visa_fechaalta","Visa_mconsumototal","Visa_mconsumospesos","Visa_mconsumosdolares","Master_cadelantosefectivo","mforex_sell","Master_msaldopesos","Visa_mlimitecompra","Visa_msaldopesos","mcomisiones_otras","Visa_mconsumototal","Master_mconsumototal")

# subset
dataset <- dataset[, !xx1, with = FALSE]
#==================================================================================================================


dataset  <- na.roughfix( dataset[ foto_mes %in% c( 202103, 202105 ) ] )

dtrain  <- dataset[ foto_mes == 202103 ]
dapply  <- dataset[ foto_mes == 202105 ]


#genero el modelo de Random Forest con la libreria ranger
#notar como la suma de muchos arboles contrarresta el efecto de min.node.size=1
param  <- list( "num.trees"=       300,  #cantidad de arboles
                "mtry"=             30,  #cantidad de variables que evalua para hacer un split  sqrt(ncol(dtrain))
                "min.node.size"=  1500,  #tamaño minimo de las hojas
                "max.depth"=        12   # 0 significa profundidad infinita
              )

set.seed(100069) #Establezco la semilla aleatoria

setorder( dtrain, clase_ternaria )  #primero quedan los BAJA+1, BAJA+2, CONTINUA


#genero el modelo de Random Forest llamando a ranger()
modelo  <- ranger( formula= "clase_ternaria ~ .",
                   data=  dtrain, 
                   probability=   TRUE,  #para que devuelva las probabilidades
                   num.trees=     param$num.trees,
                   mtry=          param$mtry,
                   min.node.size= param$min.node.size,
                   max.depth=     param$max.depth
                   #,class.weights= c( 1,40, 1)  #siguiendo con la idea de Maite San Martin
                 )

#aplico el modelo recien creado a los datos del futuro
prediccion  <- predict( modelo, dapply )

#Genero la entrega para Kaggle
entrega  <- as.data.table( list( "numero_de_cliente"= dapply[  , numero_de_cliente],
                                 "Predicted"= as.numeric(prediccion$predictions[ ,"BAJA+2" ] > 1/40) ) ) #genero la salida

#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
#dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( "C:\\DataMining_Economia_Finanzas\\exp\\",  showWarnings = FALSE ) 

#dir.create( "./exp/KA6310/", showWarnings = FALSE )
dir.create( "C:\\DataMining_Economia_Finanzas\\exp\\RandomForest0\\", showWarnings = FALSE )

#archivo_salida  <- "./exp/KA6310/KA6310_001.csv"
archivo_salida  <- "C:\\DataMining_Economia_Finanzas\\exp\\RandomForest0\\KA6310_001.csv"

#genero el archivo para Kaggle
fwrite( entrega, 
        file= archivo_salida, 
        sep="," )
