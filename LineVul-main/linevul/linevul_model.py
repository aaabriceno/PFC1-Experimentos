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
    
        
    # forward: lo que pasa cuando le damos una funcion al modelo. Recibe input_ids (los tokens de la funcion) o input_embed (embeddings ya hechos,
    # se usan para explicar la prediccion), labels (0 o 1 reales, solo cuando entrenamos o evaluamos) y output_attentions (si queremos que devuelva tambien las atenciones)
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        # camino 1: igual que el de abajo pero ademas devuelve las atenciones (que tokens miran a cuales), sirve para explicar el resultado
        if output_attentions:
            if input_ids is not None:
                # ne(1) = "distinto de 1", y el 1 es el <pad>: asi la mascara le dice al modelo que ignore el relleno y mire solo los tokens reales
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                # si no nos dan ids, entramos directo con los embeddings
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            # attentions = la atencion de cada una de las 12 capas; last_hidden_state = la salida de los 12 pisos (512 tokens x 768)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            # la cabeza (RobertaClassificationHead) toma esa salida, se queda con el vector de <s> y devuelve los 2 puntajes (logits)
            logits = self.classifier(last_hidden_state)
            # softmax convierte los 2 puntajes en probabilidades que suman 1: [prob no vulnerable, prob vulnerable]
            prob = torch.softmax(logits, dim=-1)
            # si nos dan labels calculamos el loss; CrossEntropyLoss se calcula con los logits y no con prob porque ya hace el softmax por dentro
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                # devolvemos el loss, las probabilidades y las atenciones
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            # camino 2 (el normal, sin atenciones): lo mismo pero sin devolver las atenciones
            if input_ids is not None:
                # el [0] se queda solo con el primer elemento de lo que devuelve roberta, que es last_hidden_state (la salida de los 12 pisos)
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)[0]
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)[0]
            logits = self.classifier(outputs)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob
            else:
                # sin labels solo devolvemos las probabilidades (en linevul_main.py la variable se llama logits pero ya son probabilidades)
                return prob