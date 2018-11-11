import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from keras.datasets import mnist
from ann_visualizer.visualize import ann_viz
from keras.preprocessing.image import ImageDataGenerator

def treinaCNN ( alturaImagem, larguraImagem ) :
    """
    Função que treina a rede neural
    convolucional com a base de dados
    de treinamento

    """

    print("------ Início do treinamento -------\n")
    i = 0
    resultado = []

    cNN = Sequential (  )

    # camada de convolução
    cNN.add ( Conv2D  (
            filters = 64,
            kernel_size = ( 3, 3 ),
            input_shape = ( larguraImagem, alturaImagem, 1 ),
            activation = "relu"
    ) )

    # normalização dos dados após a 1 camada de convolução
    cNN.add ( BatchNormalization (  ) )

    # camada de pooling
    cNN.add ( MaxPooling2D (
            pool_size = ( 2, 2 )
    ) )

    # 2 camada da rede neural
    cNN.add ( Conv2D (
             filters = 64,
             kernel_size = ( 3, 3 ),
             input_shape = ( larguraImagem, alturaImagem, 1 ),
             activation = "relu"
    ) )

    cNN.add ( BatchNormalization (  ) )

    cNN.add ( MaxPooling2D (
            pool_size = ( 2, 2 )
    ) )

    # camada de Flatten
    cNN.add ( Flatten ( ) )

    # montando a rede neural de camada densa de duas camadas escondidas
    cNN.add ( Dense (
            units = 128,
            activation = "relu"
    ) )

    cNN.add ( Dropout ( 0.2 ) )

    cNN.add ( Dense (
            units = 128,
            activation = "relu"
    ) )

    cNN.add ( Dropout ( 0.2 ) )

    # camada saída
    cNN.add ( Dense (
             units = 10,
             activation = "softmax"
    ) )

    cNN.compile (
             loss = "categorical_crossentropy",
             optimizer = "adam",
             metrics = ["accuracy"]
    )

    return cNN

def preImageProcessing ( dadoEntrada, dadoEntradaTeste ) :
    """
    Função que faz o pré - processamento de todas
    as imagens do dataset de treino e teste

    """

    # remodelando o formato da imagem para o modelo
    # de tensorflow para o dado {value for value in variable}de entrada treinameto
    previsaoTreinamento = dadoEntrada.reshape (
        dadoEntrada.shape[0],
        len ( dadoEntrada[:][0] ),
        len ( dadoEntrada[0][:] ),
        1
    )

    previsaoTeste = dadoEntradaTeste.reshape (
        dadoEntradaTeste.shape[0],
        len ( dadoEntradaTeste[:][0] ),
        len ( dadoEntradaTeste[0][:] ),
        1
    )

    previsaoTreinamento = previsaoTreinamento.astype ( "float64" )
    previsaoTeste = previsaoTeste.astype ( "float64" )

    previsaoTreinamento /= 255
    previsaoTeste /= 255

    return previsaoTreinamento, previsaoTeste

def main (  ) :

    seed = 10

    np.random.seed ( seed )

    ( dadoEntrada, dadoSaida ), ( dadoEntradaTeste, dadoSaidaTeste ) = mnist.load_data (  )

    previsaoTreinamento, previsaoTeste = preImageProcessing (
        dadoEntrada = dadoEntrada,
        dadoEntradaTeste = dadoEntradaTeste
    )

    dummyRespTreinamento = np_utils.to_categorical ( dadoSaida, 10 ) # conversão dos dados categóricos de treinamento
    dummyRespTeste = np_utils.to_categorical ( dadoSaidaTeste, 10 )  # conversão dos dados categóricos de teste

    tamTreinamento = len ( dummyRespTreinamento )
    tamTeste = len ( dummyRespTeste )

    #tamTreinamento = int ( tamTreinamento/128 )

    geradorImagemTreinamento = ImageDataGenerator (
        rotation_range = 7,
        horizontal_flip = True,
        shear_range = 0.2,
        height_shift_range = 0.07,
        zoom_range = 0.2
    )

    geradorImageTeste = ImageDataGenerator (  )

    baseTreinamento = geradorImageTeste.flow(
        previsaoTreinamento,
        dummyRespTreinamento,
        batch_size = 128
    )

    baseTeste = geradorImagemTreinamento.flow (
        previsaoTeste,
        dummyRespTeste,
        batch_size = 128
    )

    cNN = treinaCNN (
            alturaImagem = len ( dadoEntrada[:][0] ),
            larguraImagem = len ( dadoEntrada[0][:] )
    )

    cNN.fit_generator (
        baseTreinamento,
        steps_per_epoch = tamTreinamento/128,
        epochs = 10,
        validation_data = baseTreinamento,
        validation_steps = tamTeste/128
    )

if __name__ == '__main__':

    main (  )
