import numpy as np

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten,Conv2D,MaxPooling2D,Dropout
from keras.utils import np_utils
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import StratifiedKFold
from ann_visualizer.visualize import ann_viz

def salvaRede ( cNN ) :
    """

    Função que salva os pesos e a estrutura neural da rede
    em um arquivo json

    """

    cNNJson = cNN.to_json()

    with open("Cnn.json", "w") as jsonFile :

        jsonFile.write ( cNNJson )

    cNN.save_weights("CnnWeights.h5")

    return

def treinaCNN ( kFold, dadoSaida, classe , dadoEntradaTeste, dadoSaidaTeste ,alturaImagem, larguraImagem ) :
    """
    Função que treina a rede neural
    convolucional com a base de dados
    de treinamento

    """

    print("------ Início do treinamento -------\n")
    i = 0
    score = []
    scoreTotal = []

    for indiceTreinamento, indiceTeste in kFold.split ( dadoSaida,
                                            np.zeros ( shape = ( classe.shape[0], 1 ) ) ) :

        print ( "\n---------- epoca {} ----------\n".format ( i + 1 ) )

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

        cNN.fit ( dadoSaida [ indiceTreinamento ], classe [ indiceTreinamento ],
            batch_size = 128,
            epochs = 10
        )

        score = cNN.evaluate ( dadoEntradaTeste, dadoSaidaTeste, verbose = 10 )
        print("Score da época {} : {}\n".format ( ( i + 1 ), score[1] ) )
        scoreTotal.append ( score[1] )

        i += 1

    media = sum ( scoreTotal )/len ( scoreTotal )
    print ( "Media de acertos totais : {}".format ( media ) )

    # salvando os pesos da rede neural em um arquivo json

    with open ( "outputPesos", "w" ) as f :

        f.write ( "Pesos de cada época\n" )

        for i in range ( len ( scoreTotal ) ) :

            string = ""
            string += "Època "
            string += str ( i + 1 )
            string += " : "
            string += str ( scoreTotal[i] )

            f.write ( string )

        string = ""
        string += "Média de acertos : "
        string += str ( media )
        string = " %"

        f.write ( string )

    return


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
    previsaoTeste = previsaoTeste.astype("float64")

    previsaoTreinamento /= 255
    previsaoTeste /= 255

    return previsaoTreinamento, previsaoTeste

def main (  ) :

    seed = 10

    np.random.seed ( seed )

    ( dadoEntrada, dadoSaida ), ( dadoEntradaTeste, dadoSaidaTeste ) = mnist.load_data (  )

    print ( "Dado Entrada  {}".format ( dadoEntrada ) )
    print ( "Dado dado Saida  {}".format ( dadoSaida ) )


    previsaoTreinamento, previsaoTeste = preImageProcessing (
        dadoEntrada = dadoEntrada,
        dadoEntradaTeste = dadoEntradaTeste
    )

    dummyRespTreinamento = np_utils.to_categorical ( dadoSaida, 10 ) # conversão dos dados categóricos de treinamento
    dummyRespTeste = np_utils.to_categorical ( dadoSaidaTeste, 10 )

    #print ( len ( dummyRespTeste ) )
    print ( len ( dummyRespTreinamento ) )

    kFold = StratifiedKFold (
        n_splits = 10,
        shuffle = True,
        random_state = seed
    )

    treinaCNN (
        kFold = kFold,
        dadoSaida = previsaoTreinamento,
        classe = dummyRespTreinamento,
        dadoEntradaTeste = previsaoTeste,
        dadoSaidaTeste = dummyRespTeste,
        alturaImagem = len ( dadoEntrada[:][0] ),
        larguraImagem = len ( dadoEntrada[0][:] )
    )


if __name__ == '__main__':

    main (  )
