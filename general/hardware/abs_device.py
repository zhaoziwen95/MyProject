import abc
import collections.abc
import logging
import uuid

logger = logging.getLogger(__name__)


class PAUDVDevice(object, metaclass=abc.ABCMeta):
    """
    abstract representation of any single device(Array, ControlHardware, Multiplexer, FPGA)
    """
    control = "control"
    fpga = "fpga"

    def __init__(self,
                 input_cons: collections.abc.Sequence,
                 output_cons: collections.abc.Sequence,
                 internal_map: collections.abc.Mapping=None):
        """
        :param input_cons: iterable containing indices of input connectors of the device
        :param output_cons: iterable containing indices of output connectors of the device
        :param internal_map: mapping from output to input connectors in the device
        """
        self.input_cons = input_cons
        self.output_cons = output_cons
        self.in_connections = {}
        self.out_connections = {}
        self.internal_map = internal_map
        self.uuid = uuid.uuid4()

    def connect(self, other: "PAUDVDevice",
                output_cons_other: collections.abc.Sequence=None,
                input_cons: collections.abc.Sequence=None):
        """
        connect the given inputs of the device with PAUDVDevice other

        :param other: device to connect the inputs to
        :param input_cons: indices of inputs to connect
        :param output_cons_other: indices of outputs of device other to connect to
        """
        # get default input or check whether the device has such inputs
        if input_cons is None:
            input_cons = self.input_cons
        else:
            for in_con in input_cons:
                if in_con not in self.input_cons:
                    raise ValueError("Device {} has no input connector {}".format(str(self), in_con))

        # get default output or check whether the device other has output with such indices
        if output_cons_other is None:
            output_cons_other = other.output_cons
        else:
            for out_con_other in output_cons_other:
                if out_con_other not in other.output_cons:
                    raise ValueError("Device {} has no output connector {}".format(str(other), out_con_other))

        # check count of connectors
        if len(input_cons) != len(output_cons_other):
            raise ValueError("Number of input connectors and output connectors has to match ({} != {})".format(
                len(input_cons),
                len(output_cons_other)
            ))

        # connect devices
        for in_id, out_id_other in zip(input_cons, output_cons_other):
            self.in_connections[in_id] = (other, out_id_other)

    # TODO: remove compatibility handler
    def __setstate__(self, state):
        """
        Compatibility handler for pickled data without uuid
        :param state:
        """
        self.__dict__ = state
        if "uuid" not in self.__dict__:
            self.__dict__["uuid"] = uuid.uuid4()

    def __hash__(self):
        return int(self.uuid)

    def __eq__(self, other):
        return hash(self) == hash(other)

    def get_channels(self, out_cons=None, mode=control):
        """
        runs through the hardware tree to get the channel numbers
        of the control device for the given inputs of the device

        :param out_cons: indices of inputs
        :type out_cons: tuple
        :param mode: defines whether to get ControlHardware channels or fpga channels
        :type mode: str ('control' or 'fpga')
        :returns array with corresponding control channels to address for the given device
        """
        # query all outputs if none are defined
        if out_cons is None:
            out_cons = self.output_cons
        else:
            # check if device contains all given output connectors
            for out_con in out_cons:
                if out_con not in self.output_cons:
                    raise ValueError("Device has no output connector {}.\n"
                                     "Valid connectors: {}".format(out_con, self.output_cons))

        # get corresponding input connectors of the device
        if self.internal_map is None:  # no internal mapping -> out_con_id == in_con_id
            in_cons = out_cons
        else:
            in_cons = [None] * len(out_cons)
            for index, out_con in enumerate(out_cons):
                in_cons[index] = self.internal_map[out_con]

        # construct query dict and result mapping
        query = {}
        results = [None] * len(in_cons)
        for index, in_con in enumerate(in_cons):
            if in_con in self.in_connections:
                connection = self.in_connections[in_con]
                if connection[0] in query:
                    qlist = query[connection[0]]
                    qlist.append(connection[1])
                    results[index] = (connection[0], len(qlist) - 1)
                else:
                    query[connection[0]] = [connection[1]]
                    results[index] = (connection[0], 0)
            else:
                logger.debug("pin {} is not connected.".format(in_con))

        # execute queries
        configs = []
        for device in query:
            res, config = device.get_channels(query[device], mode)
            query[device] = res
            configs.extend(config)

        # construct result
        for index, res in enumerate(results):
            if res is not None:
                results[index] = query[res[0]][res[1]]
        return results, configs

    def __str__(self):
        return "PAUDVDevice (n_in={}, n=out={})".format(len(self.input_cons), len(self.output_cons))


