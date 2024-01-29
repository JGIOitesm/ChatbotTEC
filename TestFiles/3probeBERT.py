from transformers import pipeline

nlpqa = pipeline("question-answering", model="nlp-en-es/roberta-base-bne-finetuned-sqac")

print(nlpqa("¿Cual es su nombre?","Como hijo de Maria Mariana Montoya Mayo quien vive en el pueblo de Santo Tomás de Aquino, me llamo Pedro Pablo Pérez Pereira,"))

nlpsa = pipeline("text-classification", model="finiteautomata/beto-sentiment-analysis",top_k=None)

print(nlpsa("no esta mal"))