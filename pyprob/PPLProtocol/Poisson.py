# automatically generated by the FlatBuffers compiler, do not modify

# namespace: PPLProtocol

import flatbuffers

class Poisson(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAsPoisson(cls, buf, offset):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Poisson()
        x.Init(buf, n + offset)
        return x

    # Poisson
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Poisson
    def Rate(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            x = self._tab.Indirect(o + self._tab.Pos)
            from .ProtocolTensor import ProtocolTensor
            obj = ProtocolTensor()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

def PoissonStart(builder): builder.StartObject(1)
def PoissonAddRate(builder, rate): builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(rate), 0)
def PoissonEnd(builder): return builder.EndObject()
