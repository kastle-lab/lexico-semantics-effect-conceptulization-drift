from rdflib import Graph
filename = "small_one"
g = Graph().parse(f"{filename}.xml", format="xml")
g.serialize(f"{filename}.ttl", format="turtle")
