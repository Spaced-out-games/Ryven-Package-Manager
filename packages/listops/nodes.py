
from ryven.NENV import Node,NodeInputBP, NodeOutputBP
from pprint import pprint

class slice_Node(Node):
    """
    Get a slice of a list, ie `l[:][0]`\n
    How to use:
        
        `l = [0,1,2,3,4,5,6,7,8,9]`\n
        `l[0]`\n
        `>>>0`\n
        `l[-1]`\n
        `>>>9`\n
        `l[5:]`\n
        `>>>[5, 6, 7, 8, 9]`\n
        If startindex is not given, 0 is assumed.\n
        If endindex is not given, -1 is assumed.
    """
    title = 'slice of list'
    init_inputs = [NodeInputBP(label = 'list-like'),
    NodeInputBP(label = 'startindex'),
    NodeInputBP(label = 'endindex')
    ]
    init_outputs = [NodeOutputBP()]
    color = '#aa2352'
    def update_event(self, inp = -1):
        si = self.input(1)
        ei = self.input(2)
        si = 0 if si == None else si
        ei = -1 if ei == None else ei
        self.set_output_val(self.input(0)[si:ei])
class typeNode(Node):
    init_inputs = [NodeInputBP(label = "x")]
    init_outputs = [NodeOutputBP()]
    color = "#23aa52"
    def update_event(self, inp=-1):
        self.set_output_val(0, str(type(self.input(0))))


slice_nodes = [slice_Node, typeNode]
export_nodes(*slice_nodes)