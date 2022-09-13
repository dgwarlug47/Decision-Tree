from ast import In
from anytree import Node, RenderTree
from anytree import NodeMixin, RenderTree
udo = Node("Udo")
marc = Node("Marc", parent=udo)
lian = Node("Lian", parent=marc)
from anytree.exporter import DotExporter
DotExporter(udo).to_picture("udo.png")

class InnerTree(NodeMixin):
    def __init__(self, parentTree, id):
        self.parentTree = parentTree
        # this constraint variable is for the children, not for the current tree
        # this childrenTree0 is when the constraintVariable is equal to 0, thus the features is not present
        self.childrenTree0 = None
        # this childrenTree0 is when the constraintVariable is equal to 1, thus the features is present
        self.childrenTree1 = None
        self.prediction = None
        self.name = id
        self.parent = parentTree
        self.children = []
    
    def add_children(self, children0, children2):
        self.children = [children0, children2]

msft1 = InnerTree(None, 1)
msft2 = InnerTree(msft1, 2)
msft3 = InnerTree(msft1, 3)
msft1.add_children(msft2, msft3)
DotExporter(RenderTree(msft1)).to_picture("udo.png")
