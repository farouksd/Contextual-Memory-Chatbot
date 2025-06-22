import spacy
import networkx as nx
from pyvis.network import Network
from collections import defaultdict
#this code needs alot of extra work this isn t the final version 
nlp = spacy.load("en_core_web_sm")

class KnowledgeGraph:
    def __init__(self):
        self.graph = nx.Graph()
        self.entity_freq = defaultdict(int)
    
    def update_graph(self, text):
        doc = nlp(text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        
        for entity, label in entities:
            self.graph.add_node(entity, label=label, size=self.entity_freq[entity] + 10)
            self.entity_freq[entity] += 1
        
        for i in range(len(entities) - 1):
            self.graph.add_edge(entities[i][0], entities[i+1][0], weight=0.1)
    
    def visualize(self):
        net = Network(notebook=True, height="500px", width="100%")
        net.from_nx(self.graph)
        return net
