# para correr el Google Cloud
#   8 vCPU
#  64 GB memoria RAM
# 256 GB espacio en disco

# son varios archivos, subirlos INTELIGENTEMENTE a Kaggle

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("lightgbm")


#defino los parametros de la corrida, en una lista, la variable global  PARAM
#  muy pronto esto se leera desde un archivo formato .yaml
PARAM <- list()
PARAM$experimento  <- "KA7240"

PARAM$input$dataset       <- "./datasets/competencia2_2022.csv.gz"
PARAM$input$training      <- c( 202103 )
PARAM$input$future        <- c( 202105 )

PARAM$finalmodel$max_bin           <-     31
PARAM$finalmodel$learning_rate     <-      0.0280015981   #0.0142501265
PARAM$finalmodel$num_iterations    <-    328  #615
PARAM$finalmodel$num_leaves        <-   1015  #784
PARAM$finalmodel$min_data_in_leaf  <-   5542  #5628
PARAM$finalmodel$feature_fraction  <-      0.7832319551  #0.8382482539
PARAM$finalmodel$semilla           <- 100069

#------------------------------------------------------------------------------
#------------------------------------------------------------------------------
#Aqui empieza el programa
setwd( "~/buckets/b1" )

#cargo el dataset donde voy a entrenar
dataset  <- fread(PARAM$input$dataset, stringsAsFactors= TRUE)


# quito las columnas que tienen alta correlacion con otras o muchos nulos del dataset
#==================================================================================================================
#xx1 <- c("mcomisiones_mantenimiento","Visa_mpagado", "ctarjeta_visa_transacciones","ctarjeta_visa_debitos_automaticos","mttarjeta_visa_debitos_automaticos","mtarjeta_visa_consumo","Master_Finiciomora","mtarjeta_master_consumo","Visa_msaldototal","Visa_msaldopesos","Master_mpagospesos","Master_mconsumospesos","ctarjeta_visa_transacciones","ctarjeta_visa_transacciones","ctarjeta_master_transacciones","Master_msaldopesos","Master_msaldopesos","cextraccion_autoservicio","Visa_msaldopesos","ccajas_transacciones","mtarjeta_master_consumo","ctarjeta_master","mtarjeta_master_consumo","Visa_msaldototal","Master_msaldototal","Master_msaldototal","mpayroll2","Visa_msaldopesos","Visa_msaldopesos","Visa_mconsumospesos","Visa_mconsumototal","mtarjeta_visa_consumo","ctarjeta_visa_transacciones","mtarjeta_master_consumo","mtarjeta_visa_consumo","tcallcenter","cprestamos_hipotecarios","mcaja_ahorro_dolares","Visa_msaldototal","Visa_msaldototal","Master_msaldopesos","ctarjeta_visa_transacciones","Visa_mpagospesos","mpasivos_margen","Visa_madelantopesos","ctarjeta_master_debitos_automaticos","mtarjeta_visa_consumo","Visa_mconsumospesos","Visa_mpagospesos","cextraccion_autoservicio","ctarjeta_debito_transacciones","mrentabilidad","cextraccion_autoservicio","ccomisiones_mantenimiento","ctarjeta_visa_debitos_automaticos","Master_msaldototal","mextraccion_autoservicio","mtarjeta_visa_consumo","catm_trx","cforex","cliente_antiguedad","cforex","ctarjeta_master_transacciones","catm_trx_other","Master_fechaalta","ctarjeta_visa_transacciones","mextraccion_autoservicio","Master_msaldodolares","mtarjeta_master_consumo","mtarjeta_master_consumo","cliente_antiguedad","mtarjeta_visa_consumo","mtarjeta_visa_consumo","Visa_msaldodolares","Master_madelantopesos","cforex_sell","Master_msaldototal","Master_mlimitecompra","Visa_msaldototal","mcomisiones","Visa_mconsumospesos","Master_mconsumospesos") # vector of columns you DON'T want
xx2 <- c("Visa_mpagado","ctarjeta_visa_debitos_automaticos","Visa_cconsumos","Visa_cconsumos","mttarjeta_visa_debitos_automaticos","Visa_Finiciomora","Master_cconsumos","Visa_mpagospesos","Visa_mpagospesos","Master_mconsumototal","Master_mpagospesos","Visa_mconsumospesos","Visa_mconsumototal","mtarjeta_master_consumo","Master_mconsumototal","Master_mconsumospesos","matm","Visa_mpagominimo","ccajas_extracciones","Master_mpagospesos","Master_status","Master_msaldopesos","Visa_mpagominimo","Master_mconsumototal","Master_mconsumospesos","cpayroll2_trx","Visa_mconsumototal","Visa_mconsumospesos","Visa_cconsumos","Visa_cconsumos","Visa_msaldopesos","Visa_mpagospesos","Master_msaldototal","Visa_msaldototal","ccallcenter_transacciones","mprestamos_hipotecarios","mcuentas_saldo","Visa_mconsumototal","Visa_mconsumospesos","Master_mpagominimo","mtarjeta_visa_consumo","Visa_cconsumos","mcaja_ahorro","Visa_cadelantosefectivo","mttarjeta_master_debitos_automaticos","Visa_cconsumos","Visa_mpagospesos","Visa_mconsumototal","mextraccion_autoservicio","mautoservicio","mrentabilidad_annual","catm_trx","mcomisiones_mantenimiento","mttarjeta_visa_debitos_automaticos","Master_mpagominimo","catm_trx","Visa_mpagospesos","matm","mforex_sell","Master_fechaalta","cforex_sell","Master_cconsumos","matm_other","Visa_fechaalta","Visa_cconsumos","matm","Master_mconsumosdolares","Master_mconsumototal","Master_mconsumospesos","Visa_fechaalta","Visa_mconsumototal","Visa_mconsumospesos","Visa_mconsumosdolares","Master_cadelantosefectivo","mforex_sell","Master_msaldopesos","Visa_mlimitecompra","Visa_msaldopesos","mcomisiones_otras","Visa_mconsumototal","Master_mconsumototal")

# subset
dataset <- dataset[, !xx2, with = FALSE]
#==================================================================================================================

#--------------------------------------

#paso la clase a binaria que tome valores {0,1}  enteros
#set trabaja con la clase  POS = { BAJA+1, BAJA+2 } 
#esta estrategia es MUY importante
dataset[ , clase01 := ifelse( clase_ternaria %in%  c("BAJA+2","BAJA+1"), 1L, 0L) ]

#--------------------------------------

#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )

#--------------------------------------


#establezco donde entreno
dataset[ , train  := 0L ]
dataset[ foto_mes %in% PARAM$input$training, train  := 1L ]

#--------------------------------------
#creo las carpetas donde van los resultados
#creo la carpeta donde va el experimento
# HT  representa  Hiperparameter Tuning
dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( paste0("./exp/", PARAM$experimento, "/" ), showWarnings = FALSE )
setwd( paste0("./exp/", PARAM$experimento, "/" ) )   #Establezco el Working Directory DEL EXPERIMENTO



#dejo los datos en el formato que necesita LightGBM
dtrain  <- lgb.Dataset( data= data.matrix(  dataset[ train==1L, campos_buenos, with=FALSE]),
                        label= dataset[ train==1L, clase01] )

#genero el modelo
#estos hiperparametros  salieron de una laaarga Optmizacion Bayesiana
modelo  <- lgb.train( data= dtrain,
                      param= list( objective=          "binary",
                                   max_bin=            PARAM$finalmodel$max_bin,
                                   learning_rate=      PARAM$finalmodel$learning_rate,
                                   num_iterations=     PARAM$finalmodel$num_iterations,
                                   num_leaves=         PARAM$finalmodel$num_leaves,
                                   min_data_in_leaf=   PARAM$finalmodel$min_data_in_leaf,
                                   feature_fraction=   PARAM$finalmodel$feature_fraction,
                                   seed=               PARAM$finalmodel$semilla
                                  )
                    )

#--------------------------------------
#ahora imprimo la importancia de variables
tb_importancia  <-  as.data.table( lgb.importance(modelo) ) 
archivo_importancia  <- "impo.txt"

fwrite( tb_importancia, 
        file= archivo_importancia, 
        sep= "\t" )

#--------------------------------------


#aplico el modelo a los datos sin clase
dapply  <- dataset[ foto_mes== PARAM$input$future ]

#aplico el modelo a los datos nuevos
prediccion  <- predict( modelo, 
                        data.matrix( dapply[, campos_buenos, with=FALSE ])                                 )

#genero la tabla de entrega
tb_entrega  <-  dapply[ , list( numero_de_cliente, foto_mes ) ]
tb_entrega[  , prob := prediccion ]

#grabo las probabilidad del modelo
fwrite( tb_entrega,
        file= "prediccion.txt",
        sep= "\t" )

#ordeno por probabilidad descendente
setorder( tb_entrega, -prob )


#genero archivos con los  "envios" mejores
#deben subirse "inteligentemente" a Kaggle para no malgastar submits
cortes <- seq( 5000, 12000, by=500 )
for( envios  in  cortes )
{
  tb_entrega[  , Predicted := 0L ]
  tb_entrega[ 1:envios, Predicted := 1L ]

  fwrite( tb_entrega[ , list(numero_de_cliente, Predicted)], 
          file= paste0(  PARAM$experimento, "_", envios, ".csv" ),
          sep= "," )
}

#--------------------------------------

quit( save= "no" )
