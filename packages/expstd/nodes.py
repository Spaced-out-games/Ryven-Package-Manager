
from ryven.NENV import Node,NodeInputBP, NodeOutputBP, export_nodes
from ryven.example_nodes.std.widgets import EvalNode_MainWidget
from pprint import pprint
from ast import literal_eval

'''
This file adds some missing key features into Ryven such as greater list support, support for list slices,safe evaluation node, and much more
'''
class Safe_Eval_Node(Node):
    """
    Useful for evaluating expressions that define dictionaries and lists.\nNot a security risk, as it cannot execute any potentially malicious code,\n\t and only parses expressions
    """
    title = 'Evaluate Expression'
    version = 'v0.1'
    init_inputs = [
        NodeInputBP(),
    ]
    init_outputs = [
        NodeOutputBP(),
    ]
    main_widget_class = EvalNode_MainWidget
    main_widget_pos = 'between ports'

    def __init__(self, params):
        super().__init__(params)

        self.actions['add input'] = {'method': self.add_param_input}

        self.number_param_inputs = 0
        self.expression_code = None

    def place_event(self):
        if self.number_param_inputs == 0:
            self.add_param_input()

    def add_param_input(self):
        self.create_input()

        index = self.number_param_inputs
        self.actions[f'remove input {index}'] = {
            'method': self.remove_param_input,
            'data': index
        }

        self.number_param_inputs += 1

    def remove_param_input(self, index):
        self.delete_input(index)
        self.number_param_inputs -= 1
        del self.actions[f'remove input {self.number_param_inputs}']

    def update_event(self, inp=-1):
        inp = [self.input(i) for i in range(self.number_param_inputs)]
        self.set_output_val(0, literal_eval(self.expression_code))

    def get_state(self) -> dict:
        return {
            'num param inputs': self.number_param_inputs,
            'expression code': self.expression_code,
        }

    def set_state(self, data: dict, version):
        self.number_param_inputs = data['num param inputs']
        self.expression_code = data['expression code'] 
nodes = [Safe_Eval_Node]
export_nodes(*nodes)
