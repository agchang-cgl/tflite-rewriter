from tflite.Model import Model
buf = open('/Users/andrew.chang/test/model.tflite', 'rb').read()
buf = bytearray(buf)
m = Model.GetRootAsModel(buf, 0)
s = m.Subgraphs(0)
# [70, 34, 39, 3, 0, 40, 36, 41, 77]
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/schema/schema.fbs#L224
OPS_TO_NAME = {
    70: "EXPAND_DIMS",
    34: "PAD",
    39: "TRANSPOSE",
    3: "CONV_2D",
    0: "ADD",
    40: "MEAN",
    36: "GATHER",
    41: "SUB",
    77: "SHAPE",
}

operator_codes = []
for i in range(m.OperatorCodesLength()):
    operator_codes.append(m.OperatorCodes(i).BuiltinCode())
print(operator_codes)

graph_ops = []
for i in range(s.OperatorsLength()):
    opcode_idx = s.Operators(i).OpcodeIndex()
    graph_ops.append(OPS_TO_NAME[operator_codes[opcode_idx]])

print(graph_ops)
