import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from ann_visualizer.visualize import ann_viz

def criaCNN ( larguraImagem, alturaImagem ) :
    """
    Função que cria a CNN para classificação
    de imagem

    """

    cNN = Sequential ( )

    # o número de kernels vem da forma n = 32*i
    # onde i você decide a quantidade de bytes

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

    #ann_viz ( cNN, title = "Rede Convolucional" )

    return cNN

def preImageProcessing ( xDadoTreinamento, xDadoTeste, yDadoTreinamento, yDadoTeste ) :
    """
    Função que faz o pré - processamento de todas
    as imagens do dataset de treino e teste

    """

    # remodelando o formato da imagem para o modelo
    # de tensorflow para o dado {value for value in variable}de entrada treinameto
    previsaoTreinamento = xDadoTreinamento.reshape (
        xDadoTreinamento.shape[0],
        len ( xDadoTreinamento[:][0] ),
        len ( xDadoTreinamento[0][:] ),
        1
    )

    previsaoTreinamento = previsaoTreinamento.astype ( "float64" )

    # remodelando o formato da imagem para o modelo
    # de tensorflow para o dado de entra teste
    previsaoTeste = xDadoTeste.reshape (
        xDadoTeste.shape[0],
        len ( xDadoTeste[:][0] ),
        len ( xDadoTeste[0][:] ),
        1
    )

    previsaoTeste = previsaoTeste.astype ( "float64" )

    # fazendo a normalização dos pixels cinzas
    previsaoTreinamento /= 255
    previsaoTeste /= 255

    return previsaoTreinamento, previsaoTeste

def main (  ) :
    """
    Função principal que faz o processo
    de rede convolucional

    """
    ( xDadoTreinamento, yDadoTreinamento ), ( xDadoTeste, yDadoTeste ) = mnist.load_data()

    larguraImagem = len ( xDadoTeste[:][0] )
    alturaImagem = len ( xDadoTeste[0][:] )

    previsaoTreinamento, previsaoTeste = preImageProcessing (
        xDadoTreinamento = xDadoTreinamento,
        xDadoTeste = xDadoTreinamento,
        yDadoTreinamento = yDadoTreinamento,
        yDadoTeste = yDadoTeste )

    dummyRespTreinamento = np_utils.to_categorical ( yDadoTreinamento, 10 ) # conversão dos dados categóricos de treinamento
    dummyRespTeste = np_utils.to_categorical ( yDadoTreinamento, 10 ) # conversão dos dados categóricos de teste

    cNN = criaCNN (
        larguraImagem = larguraImagem,
        alturaImagem = alturaImagem
    )

    cNN.fit ( previsaoTreinamento, dummyRespTreinamento,
              batch_size = 128,
              epochs = 20,
              validation_data = ( previsaoTeste, dummyRespTeste )
    )

    resultado = cNN.evaluate (
        previsaoTeste,
        dummyRespTreinamento
    )

if __name__ == '__main__':
    main()
