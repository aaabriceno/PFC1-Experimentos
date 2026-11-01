import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification

# Esta clase se usa para tareas de clasificar a nivel de oraciones, en este caso el modelo toma características del modelo 
# y las convierte en una predicción de clasificación binaria (devuelve dos puntajes)
# para que luego la decisión salga de la funcion softmax y del umbral establecido
class RobertaClassificationHead(nn.Module): #ese nn.Module es la clase base para modulos en la libreria pytorch
    """Head for sentence-level classification tasks."""
    def __init__(self, config): # ese config es un objeto que contiene configuraciones del modelo
        super().__init__() # llamamos al constructor de la clase nn.Module
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) #con esto creamos una especie de capa que toma vectores de entrada del tamaño de hidden_size y devuelve vectores de tamaño hidden_size
        # matriz de 768 x 768 + 768 sesgo
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob) #es un valor de dropout que permite apagar neuronas al azar
        # por ejemplo p = 0.1 (valor que se usa) y se apaga una cantidad de neuronas para que las demas aprendan y no memoricen, en este
        # caso se apagan valores del vector de 768 valores
        
        # el out_proj es una capa que toma el vector de 768 valores (el de <s> ya procesado por dropout, dense, tanh y dropout)
        # cabe aclarar que esos 768 salen del valor de hidden_size, y out_proj es como una especie de funcion que comprime esa informacion
        # para convertirla en dos valores (logits), uno por clase [no vulnerable, vulnerable]: le pasamos un tensor de entrada de 768 valores
        # y retorna un tensor de salida de 2 valores. Sus pesos nacen al azar (no vienen de CodeBERT) y se aprenden en el fine-tuning
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

# estamos heredando todo lo que tiene la clase RobertaForSequenceClassification (RoBERTa con una cabeza de clasificacion de HuggingFace) y vamos a agregarle mas cosas, se parece a ella, pero con mas funciones extra, como si fuese una caja de herramientas
# que junta lo que LineVul necesita        
class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        # este super().__init__ hace que el padre cree TAMBIEN su propio roberta (con pesos al azar) y su propio classifier,
        # pero en el forward nunca se usan: el que se usa es self.encoder.roberta (por eso el checkpoint pesa casi el doble)

        # self.encoder es el CodeBERT ya preentrenado (viene de linevul_main.py), el cerebro que llega a leer el texto de la funcion
        # aqui solo usamos su parte .roberta, que son los 12 pisos de atencion (se puede entender como un libro que tiene recetas para hacer acciones)
        self.encoder = encoder

        # el tokenizer llega a ser ese elemento que convierte el texto del codigo a numeros (IDs de tokens, con BPE), como si fuese un traductor
        self.tokenizer = tokenizer

        # el padre (RobertaForSequenceClassification) ya crea su propio self.classifier, aqui lo reemplazamos por la cabeza de LineVul
        # (la clase de arriba), porque con config.num_labels = 1 la del padre daria 1 sola salida y LineVul necesita 2
        self.classifier = RobertaClassificationHead(config)

        # esto llega a ser las instrucciones del experimento que pasamos por consola: tamaño de batch, epochs, etc
        self.args = args
    
        
    # forward = lo que pasa cuando le damos funciones al modelo. Parametros (ejemplo con un batch de 4 funciones de 512 tokens cada una):
    #  - input_ids: los tokens de las funciones ya convertidos a IDs, forma (4, 512). Ej de una fila: [0, ..., 2, 1, 1] (0 = <s>, 2 = </s>, 1 = <pad>)
    #  - input_embed: alternativa a input_ids, los embeddings ya hechos, forma (4, 512, 768). Solo se usa en la explicacion por atribucion (DeepLift, etc), no al entrenar
    #  - labels: la etiqueta real de cada funcion, 0 (no vulnerable) o 1 (vulnerable), forma (4,). Ej: tensor([0, 1, 0, 1]). Sin labels no hay loss
    #  - output_attentions: True o False. Si es True devuelve tambien las atenciones de las 12 capas (para explicar). En train, evaluate y test va en False
    # Devuelve: (loss, prob) si hay labels, solo prob si no los hay; con output_attentions=True se agregan las attentions al final
    # En main siempre se llama por nombre (input_ids=...) porque en la firma input_ids es el ULTIMO parametro, no el primero
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        # camino 1: igual que el de abajo pero ademas devuelve las atenciones (que tokens miran a cuales), sirve para explicar el resultado
        if output_attentions:
            if input_ids is not None:
                # attention_mask = input_ids.ne(1): ne = "not equal", da True donde el token es distinto de 1. El 1 es el <pad>, asi que la mascara vale True en los tokens reales y
                # False en el relleno, y el modelo ignora el relleno. Ej: ids [0, 8, 9, 2, 1, 1] -> mascara [True, True, True, True, False, False]
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                # si no nos dan ids, entramos directo con los embeddings
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            # attentions = tupla de 12 tensores, uno por capa, forma (4, 12 cabezas, 512, 512): cuanto mira cada token a cada otro token
            # last_hidden_state = la salida de los 12 pisos, forma (4, 512, 768): un vector de 768 por token
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            # la cabeza (RobertaClassificationHead) recibe last_hidden_state (4, 512, 768), se queda con el vector de <s> (4, 768) y devuelve los logits (4, 2): 2 puntajes por funcion
            logits = self.classifier(last_hidden_state)
            # softmax convierte los 2 puntajes de cada funcion en probabilidades que suman 1: [prob no vulnerable, prob vulnerable]. Ej: [2.0, -1.0] -> [0.95, 0.05]
            # dim=-1 significa "el ultimo eje". logits tiene forma (batch, 2): eje 0 = las funciones, eje 1 = las 2 clases. Como -1 es el ultimo eje, aqui -1 y 1 dan lo mismo,
            # pero -1 siempre apunta al eje de las clases sin importar cuantos ejes tenga el tensor. Con dim=0 se mezclarian funciones distintas del batch y no sumarian 1
            prob = torch.softmax(logits, dim=-1)
            # labels = la etiqueta real de cada funcion, forma (4,), enteros 0 o 1. Ej: tensor([0, 1, 0, 1])
            # CrossEntropyLoss compara los 2 logits de cada funcion contra su label y devuelve UN solo numero (el promedio del batch).
            # Se le pasan los logits y no prob porque ya hace el softmax por dentro. Ej: logits [2.0, -1.0] con label 0 -> loss ~0.049 (acierta); con label 1 -> loss ~3.05 (se equivoca)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                # devolvemos el loss, las probabilidades y las atenciones
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            # camino 2 (el normal, sin atenciones): es el que usan train, evaluate y test en linevul_main.py -> model(input_ids=..., labels=...) devuelve (loss, prob)
            if input_ids is not None:
                # el [0] se queda solo con el primer elemento de lo que devuelve roberta, que es last_hidden_state (4, 512, 768) = la salida de los 12 pisos
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            # la cabeza recibe (4, 512, 768) y devuelve los logits (4, 2): 2 puntajes por funcion
            logits = self.classifier(outputs)
            # softmax con dim=-1 (el ultimo eje = las 2 clases de cada funcion), igual que arriba
            prob = torch.softmax(logits, dim=-1)
            # igual que arriba: si hay labels sacamos el loss (un solo numero) y devolvemos (loss, prob)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                # sin labels solo devolvemos las probabilidades (en linevul_main.py la variable se llama logits pero ya son probabilidades)
                return prob