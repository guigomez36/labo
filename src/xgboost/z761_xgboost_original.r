# XGBoost  sabor original ,  cambiando algunos de los parametros

#limpio la memoria
rm( list=ls() )  #remove all objects
gc()             #garbage collection

require("data.table")
require("xgboost")

#Aqui se debe poner la carpeta de la computadora local
setwd("C:\\DataMining_Economia_Finanzas\\")   #Establezco el Working Directory

#cargo el dataset donde voy a entrenar
dataset  <- fread("C:\\DataMining_Economia_Finanzas\\datasets\\competencia2_2022.csv.gz", stringsAsFactors= TRUE)

# quito las columnas que tienen alta correlacion con otras o muchos nulos del dataset
# quito tambien las variables que tienen poco peso en el arbol original al quitar las xx1. 1 orden de magnitud por debajo de las principales
#==================================================================================================================
xx1 <- c("mcomisiones_mantenimiento","Visa_mpagado", "ctarjeta_visa_transacciones","ctarjeta_visa_debitos_automaticos","mttarjeta_visa_debitos_automaticos","mtarjeta_visa_consumo","Master_Finiciomora","mtarjeta_master_consumo","Visa_msaldototal","Visa_msaldopesos","Master_mpagospesos","Master_mconsumospesos","ctarjeta_visa_transacciones","ctarjeta_visa_transacciones","ctarjeta_master_transacciones","Master_msaldopesos","Master_msaldopesos","cextraccion_autoservicio","Visa_msaldopesos","ccajas_transacciones","mtarjeta_master_consumo","ctarjeta_master","mtarjeta_master_consumo","Visa_msaldototal","Master_msaldototal","Master_msaldototal","mpayroll2","Visa_msaldopesos","Visa_msaldopesos","Visa_mconsumospesos","Visa_mconsumototal","mtarjeta_visa_consumo","ctarjeta_visa_transacciones","mtarjeta_master_consumo","mtarjeta_visa_consumo","tcallcenter","cprestamos_hipotecarios","mcaja_ahorro_dolares","Visa_msaldototal","Visa_msaldototal","Master_msaldopesos","ctarjeta_visa_transacciones","Visa_mpagospesos","mpasivos_margen","Visa_madelantopesos","ctarjeta_master_debitos_automaticos","mtarjeta_visa_consumo","Visa_mconsumospesos","Visa_mpagospesos","cextraccion_autoservicio","ctarjeta_debito_transacciones","mrentabilidad","cextraccion_autoservicio","ccomisiones_mantenimiento","ctarjeta_visa_debitos_automaticos","Master_msaldototal","mextraccion_autoservicio","mtarjeta_visa_consumo","catm_trx","cforex","cliente_antiguedad","cforex","ctarjeta_master_transacciones","catm_trx_other","Master_fechaalta","ctarjeta_visa_transacciones","mextraccion_autoservicio","Master_msaldodolares","mtarjeta_master_consumo","mtarjeta_master_consumo","cliente_antiguedad","mtarjeta_visa_consumo","mtarjeta_visa_consumo","Visa_msaldodolares","Master_madelantopesos","cforex_sell","Master_msaldototal","Master_mlimitecompra","Visa_msaldototal","mcomisiones","Visa_mconsumospesos","Master_mconsumospesos","Master_mpagado","cplazo_fijo","Master_madelantodolares","Master_mpagosdolares","cseguro_accidentes_personales","Visa_delinquency","Master_cadelantosefectivo","ccaja_ahorro","cseguro_vida","Visa_cadelantosefectivo","matm_other","cpagomiscuentas","ccuenta_debitos_automaticos","Visa_mpagosdolares","Master_fultimo_cierre","Master_mconsumosdolares","Visa_fultimo_cierre","internet","mplazo_fijo_dolares","Visa_mlimitecompra","matm","ctarjeta_debito","mpagomiscuentas","ctransferencias_emitidas","ctransferencias_recibidas","Master_mconsumototal","Master_cconsumos","Visa_mconsumosdolares","Visa_mfinanciacion_limite","mtransferencias_emitidas","Master_mpagominimo","thomebanking","mtransferencias_recibidas","cmobile_app_trx","mcuenta_debitos_automaticos","ccaja_seguridad","ccallcenter_transacciones","mautoservicio","Visa_Fvencimiento","Master_mfinanciacion_limite","chomebanking_transacciones","Visa_status","cprestamos_personales","ccomisiones_otras","ccajas_consultas","Master_Fvencimiento") 
# vector of columns you DON'T want
#xx2 <- c("Visa_mpagado","ctarjeta_visa_debitos_automaticos","Visa_cconsumos","Visa_cconsumos","mttarjeta_visa_debitos_automaticos","Visa_Finiciomora","Master_cconsumos","Visa_mpagospesos","Visa_mpagospesos","Master_mconsumototal","Master_mpagospesos","Visa_mconsumospesos","Visa_mconsumototal","mtarjeta_master_consumo","Master_mconsumototal","Master_mconsumospesos","matm","Visa_mpagominimo","ccajas_extracciones","Master_mpagospesos","Master_status","Master_msaldopesos","Visa_mpagominimo","Master_mconsumototal","Master_mconsumospesos","cpayroll2_trx","Visa_mconsumototal","Visa_mconsumospesos","Visa_cconsumos","Visa_cconsumos","Visa_msaldopesos","Visa_mpagospesos","Master_msaldototal","Visa_msaldototal","ccallcenter_transacciones","mprestamos_hipotecarios","mcuentas_saldo","Visa_mconsumototal","Visa_mconsumospesos","Master_mpagominimo","mtarjeta_visa_consumo","Visa_cconsumos","mcaja_ahorro","Visa_cadelantosefectivo","mttarjeta_master_debitos_automaticos","Visa_cconsumos","Visa_mpagospesos","Visa_mconsumototal","mextraccion_autoservicio","mautoservicio","mrentabilidad_annual","catm_trx","mcomisiones_mantenimiento","mttarjeta_visa_debitos_automaticos","Master_mpagominimo","catm_trx","Visa_mpagospesos","matm","mforex_sell","Master_fechaalta","cforex_sell","Master_cconsumos","matm_other","Visa_fechaalta","Visa_cconsumos","matm","Master_mconsumosdolares","Master_mconsumototal","Master_mconsumospesos","Visa_fechaalta","Visa_mconsumototal","Visa_mconsumospesos","Visa_mconsumosdolares","Master_cadelantosefectivo","mforex_sell","Master_msaldopesos","Visa_mlimitecompra","Visa_msaldopesos","mcomisiones_otras","Visa_mconsumototal","Master_mconsumototal")

# subset
dataset <- dataset[, !xx1, with = FALSE]
#==================================================================================================================


#paso la clase a binaria que tome valores {0,1}  enteros
dataset[ foto_mes==202103, clase01 := ifelse( clase_ternaria=="BAJA+2", 1L, 0L) ]

#los campos que se van a utilizar
campos_buenos  <- setdiff( colnames(dataset), c("clase_ternaria","clase01") )


#dejo los datos en el formato que necesita XGBoost
dtrain  <- xgb.DMatrix( data= data.matrix(  dataset[ foto_mes==202103 , campos_buenos, with=FALSE]),
                        label= dataset[ foto_mes==202103, clase01 ] )

#genero el modelo con los parametros por default
modelo  <- xgb.train( data= dtrain,
                      param= list( objective=       "binary:logistic",
                                   max_depth=           13,
                                   min_child_weight=    6,
                                   eta=                 0.0102,
                                   colsample_bytree=    0.6376,
                                   gamma=               0.0,
                                   alpha=               0.0,
                                   lambda=              0.0,
                                   subsample=           1.0,
                                   scale_pos_weight=    1.0
                                   ),
                      #base_score= mean( getinfo(dtrain, "label")),
                      nrounds= 490
                    )


#aplico el modelo a los datos nuevos
prediccion  <- predict( modelo, 
                        data.matrix( dataset[ foto_mes==202105, campos_buenos, with=FALSE ]) )


#Genero la entrega para Kaggle
entrega  <- as.data.table( list( "numero_de_cliente"= dataset[ foto_mes==202105, numero_de_cliente],
                                 "Predicted"= as.integer( prediccion > 0.0231066132524011) )  ) #genero la salida

dir.create( "./exp/",  showWarnings = FALSE ) 
dir.create( "./exp/KA7610_xgb/", showWarnings = FALSE )
archivo_salida  <- "./exp/KA7610_xgb/KA7610_001.csv"

#genero el archivo para Kaggle
fwrite( entrega, 
        file= archivo_salida, 
        sep= "," )
