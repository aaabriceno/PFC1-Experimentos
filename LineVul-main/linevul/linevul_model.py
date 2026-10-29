import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import RobertaForSequenceClassification

#Esta clase es un módulo en Pytorch que se usa para tareas de clasificar a nivel de oraciones, en este caso el modelo toma características del modelo 
# y las convierte en una predicción de clasificación binaria (T o F)
class RobertaClassificationHead(nn.Module): #ese nn.Module es la clase base para modulos en la libreria pytorch
    """Head for sentence-level classification tasks."""
    def __init__(self, config): # ese config es un objeto que contiene configuraciones del modelo
        super().__init__() # llamamos al constructor de la clase nn.Module
        self.dense = nn.Linear(config.hidden_size, config.hidden_size) #con esto creamos una especie de capa que toma vectores de entrada del tamaño de hidden_size y devuelve vectores de tamaño hidden_size
        self.dropout = nn.Dropout(config.hidden_dropout_prob) #es un valor de dropout que permite apagar neuromas al azar
        # por ejemplo p = 0.1 (valor que se usa) y se apaga una cantidad de neuronas para que las demas aprendan y no memorizen
        
        # el out_proj es una capa que permite poder tomar los 768 valores de cada la ultima parte de los pasos osea antes de que termine todo se devuelve un vector de 768 valores
        # cabe aclarar que esos 768 salen del valor de hidden_size,ese vector es tomado por out_proj que es como una especie de funcion que comprime esa informacion para convertirlo
        # en dos valores digamoslo asi en este caso el nombre real es el tensor de salida que se obtuvo de poder psasarle un tensor de entrada de 768 valores a out_proj para que retorne
        # un tensor de salida de 2 valores 
        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x
        
class Model(RobertaForSequenceClassification):   
    def __init__(self, encoder, config, tokenizer, args):
        super(Model, self).__init__(config=config)
        self.encoder = encoder
        self.tokenizer = tokenizer
        self.classifier = RobertaClassificationHead(config)
        self.args = args
    
        
    def forward(self, input_embed=None, labels=None, output_attentions=False, input_ids=None):
        if output_attentions:
            if input_ids is not None:
                outputs = self.encoder.roberta(input_ids, attention_mask=input_ids.ne(1), output_attentions=output_attentions)
            else:
                outputs = self.encoder.roberta(inputs_embeds=input_embed, output_attentions=output_attentions)
            attentions = outputs.attentions
            last_hidden_state = outputs.last_hidden_state
            logits = self.classifier(last_hidden_state)
            prob = torch.softmax(logits, dim=-1)
            if labels is not None:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits, labels)
                return loss, prob, attentions
            else:
                return prob, attentions
        else:
            if input_ids is not None:
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
                return prob