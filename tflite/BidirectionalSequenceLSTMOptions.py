# automatically generated by the FlatBuffers compiler, do not modify

# namespace: tflite

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class BidirectionalSequenceLSTMOptions(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsBidirectionalSequenceLSTMOptions(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = BidirectionalSequenceLSTMOptions()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def BidirectionalSequenceLSTMOptionsBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x54\x46\x4C\x33", size_prefixed=size_prefixed)

    # BidirectionalSequenceLSTMOptions
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # BidirectionalSequenceLSTMOptions
    def FusedActivationFunction(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int8Flags, o + self._tab.Pos)
        return 0

    # BidirectionalSequenceLSTMOptions
    def CellClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # BidirectionalSequenceLSTMOptions
    def ProjClip(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Float32Flags, o + self._tab.Pos)
        return 0.0

    # BidirectionalSequenceLSTMOptions
    def MergeOutputs(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return bool(self._tab.Get(flatbuffers.number_types.BoolFlags, o + self._tab.Pos))
        return False

def BidirectionalSequenceLSTMOptionsStart(builder): builder.StartObject(4)
def BidirectionalSequenceLSTMOptionsAddFusedActivationFunction(builder, fusedActivationFunction): builder.PrependInt8Slot(0, fusedActivationFunction, 0)
def BidirectionalSequenceLSTMOptionsAddCellClip(builder, cellClip): builder.PrependFloat32Slot(1, cellClip, 0.0)
def BidirectionalSequenceLSTMOptionsAddProjClip(builder, projClip): builder.PrependFloat32Slot(2, projClip, 0.0)
def BidirectionalSequenceLSTMOptionsAddMergeOutputs(builder, mergeOutputs): builder.PrependBoolSlot(3, mergeOutputs, 0)
def BidirectionalSequenceLSTMOptionsEnd(builder): return builder.EndObject()
