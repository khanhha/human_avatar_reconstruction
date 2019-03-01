from enum import Enum

class AutoNumber(Enum):
    def __new__(cls):
        value = len(cls.__members__)  # note no + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

class SliceID(AutoNumber):
    Neck = ()
    Collar = ()
    Shoulder = ()
    Aux_Armscye_Shoulder_0 = ()
    Armscye = ()
    Aux_Bust_Armscye_0 = ()

    Bust = ()
    Aux_UnderBust_Bust_0 = ()
    UnderBust = ()

    Aux_Waist_UnderBust_2 = ()
    Aux_Waist_UnderBust_1 = ()
    Aux_Waist_UnderBust_0 = ()

    Waist = ()
    Aux_Hip_Waist_0 = ()
    Aux_Hip_Waist_1 = ()

    Hip = ()
    Aux_Crotch_Hip_2 = ()
    Aux_Crotch_Hip_1 = ()
    Aux_Crotch_Hip_0 = ()
    Crotch = ()

    UnderCrotch = ()
    Aux_Knee_UnderCrotch_3 = ()
    Aux_Knee_UnderCrotch_2 = ()
    Aux_Knee_UnderCrotch_1 = ()
    Aux_Knee_UnderCrotch_0 = ()

    Knee = ()
    Calf = ()
    Ankle = ()

    Aux_Shoulder_Elbow_0 = ()
    Aux_Shoulder_Elbow_1 = ()
    Aux_Shoulder_Elbow_2 = ()
    Elbow = ()
    Aux_Elbow_Wrist_0 = ()
    Aux_Elbow_Wrist_1 = ()
    Aux_Elbow_Wrist_2 = ()
    Wrist = ()

    @staticmethod
    def find_enum(name_id):
        for name, member in SliceID.__members__.items():
            if name == name_id:
                return member
        return None

class SliceModelInputDef:

    _input_dict = {}

    def __init__(self, mode):
        self.init_mode(mode)

    def get_input_def(self, name):
        id = SliceID[name]
        if id not in self._input_dict:
            return [name]
        else:
            return [en.name for en in self._input_dict[id]]

    def init_mode(self, mode):
        if mode == 'single':
            self._init_single_inputs_mode()
        elif mode == 'local':
            self._init_local_inputs_mode()
        elif mode == 'global':
            self._init_global_inputs_mode()
        elif mode == 'local_global':
            self._init_local_global_inputs_mode()
        elif mode == 'torso':
            self._init_torso_inputs_mode()
        else:
            assert 'unrecognized input mode'

    def _init_single_inputs_mode(self):
        # let the function get_input_def handle this case
        self._input_dict = {}

    def _init_local_inputs_mode(self):
        self._input_dict[SliceID.Hip] = [SliceID.Aux_Crotch_Hip_2, SliceID.Hip, SliceID.Aux_Hip_Waist_0]
        self._input_dict[SliceID.Aux_Crotch_Hip_2] = [SliceID.Aux_Crotch_Hip_1, SliceID.Aux_Crotch_Hip_2, SliceID.Hip]
        self._input_dict[SliceID.Aux_Crotch_Hip_1] = [SliceID.Aux_Crotch_Hip_0, SliceID.Aux_Crotch_Hip_1, SliceID.Aux_Crotch_Hip_2]
        self._input_dict[SliceID.Aux_Crotch_Hip_0] = [SliceID.Crotch, SliceID.Aux_Crotch_Hip_0, SliceID.Aux_Crotch_Hip_1]
        self._input_dict[SliceID.Crotch] = [SliceID.Crotch, SliceID.Aux_Crotch_Hip_0, SliceID.Aux_Crotch_Hip_1]

        self._input_dict[SliceID.Aux_Hip_Waist_0] = [SliceID.Hip, SliceID.Aux_Hip_Waist_0, SliceID.Aux_Hip_Waist_1]
        self._input_dict[SliceID.Aux_Hip_Waist_1] = [SliceID.Aux_Hip_Waist_0, SliceID.Aux_Hip_Waist_1, SliceID.Waist]
        self._input_dict[SliceID.Waist] = [SliceID.Aux_Hip_Waist_1, SliceID.Waist, SliceID.Aux_Waist_UnderBust_0]
        self._input_dict[SliceID.Aux_Waist_UnderBust_0] = [SliceID.Waist, SliceID.Aux_Waist_UnderBust_0, SliceID.Aux_Waist_UnderBust_1]
        self._input_dict[SliceID.Aux_Waist_UnderBust_1] = [SliceID.Aux_Waist_UnderBust_0, SliceID.Aux_Waist_UnderBust_1, SliceID.Aux_Waist_UnderBust_2]
        self._input_dict[SliceID.Aux_Waist_UnderBust_2] = [SliceID.Aux_Waist_UnderBust_1, SliceID.Aux_Waist_UnderBust_2, SliceID.UnderBust]
        self._input_dict[SliceID.UnderBust] = [SliceID.Aux_Waist_UnderBust_2, SliceID.UnderBust, SliceID.Aux_UnderBust_Bust_0]  # test
        self._input_dict[SliceID.Aux_UnderBust_Bust_0] = [SliceID.UnderBust, SliceID.Aux_UnderBust_Bust_0, SliceID.Bust]
        self._input_dict[SliceID.Bust] = [SliceID.Aux_UnderBust_Bust_0, SliceID.Bust, SliceID.Armscye]

        #TODO leg slices

    def _init_local_global_inputs_mode(self):
        global_ids = [SliceID.Hip, SliceID.Waist, SliceID.Bust]

        self._input_dict[SliceID.Hip] = [SliceID.Aux_Crotch_Hip_2, SliceID.Hip, SliceID.Aux_Hip_Waist_0] + global_ids
        self._input_dict[SliceID.Aux_Crotch_Hip_2] = [SliceID.Aux_Crotch_Hip_1, SliceID.Aux_Crotch_Hip_2, SliceID.Hip] + global_ids
        self._input_dict[SliceID.Aux_Crotch_Hip_1] = [SliceID.Aux_Crotch_Hip_0, SliceID.Aux_Crotch_Hip_1, SliceID.Aux_Crotch_Hip_2] + global_ids
        self._input_dict[SliceID.Aux_Crotch_Hip_0] = [SliceID.Crotch, SliceID.Aux_Crotch_Hip_0, SliceID.Aux_Crotch_Hip_1] + global_ids
        self._input_dict[SliceID.Crotch] = [SliceID.Crotch, SliceID.Aux_Crotch_Hip_0, SliceID.Aux_Crotch_Hip_1] + global_ids

        self._input_dict[SliceID.Aux_Hip_Waist_0] = [SliceID.Hip, SliceID.Aux_Hip_Waist_0, SliceID.Aux_Hip_Waist_1] + global_ids
        self._input_dict[SliceID.Aux_Hip_Waist_1] = [SliceID.Aux_Hip_Waist_0, SliceID.Aux_Hip_Waist_1, SliceID.Waist] + global_ids
        self._input_dict[SliceID.Waist] = [SliceID.Aux_Hip_Waist_1, SliceID.Waist, SliceID.Aux_Waist_UnderBust_0] + global_ids
        self._input_dict[SliceID.Aux_Waist_UnderBust_0] = [SliceID.Waist, SliceID.Aux_Waist_UnderBust_0, SliceID.Aux_Waist_UnderBust_1] + global_ids
        self._input_dict[SliceID.Aux_Waist_UnderBust_1] = [SliceID.Aux_Waist_UnderBust_0, SliceID.Aux_Waist_UnderBust_1, SliceID.Aux_Waist_UnderBust_2] + global_ids
        self._input_dict[SliceID.Aux_Waist_UnderBust_2] = [SliceID.Aux_Waist_UnderBust_1, SliceID.Aux_Waist_UnderBust_2, SliceID.UnderBust] + global_ids
        self._input_dict[SliceID.UnderBust] = [SliceID.Aux_Waist_UnderBust_2, SliceID.UnderBust, SliceID.Aux_UnderBust_Bust_0] + global_ids
        self._input_dict[SliceID.Aux_UnderBust_Bust_0] = [SliceID.UnderBust, SliceID.Aux_UnderBust_Bust_0, SliceID.Bust] + global_ids
        self._input_dict[SliceID.Bust] = [SliceID.Aux_UnderBust_Bust_0, SliceID.Bust, SliceID.Armscye] + global_ids

        #TODO leg slices

    def _init_global_inputs_mode(self):

        global_ids = [SliceID.Hip, SliceID.Waist, SliceID.Bust]

        self._input_dict[SliceID.Hip] = global_ids
        self._input_dict[SliceID.Aux_Crotch_Hip_2] = global_ids
        self._input_dict[SliceID.Aux_Crotch_Hip_1] = global_ids
        self._input_dict[SliceID.Aux_Crotch_Hip_0] = global_ids
        self._input_dict[SliceID.Crotch] = global_ids

        self._input_dict[SliceID.Aux_Hip_Waist_0] = global_ids
        self._input_dict[SliceID.Aux_Hip_Waist_1] = global_ids
        self._input_dict[SliceID.Waist] = global_ids
        self._input_dict[SliceID.Aux_Waist_UnderBust_0] = global_ids
        self._input_dict[SliceID.Aux_Waist_UnderBust_1] = global_ids
        self._input_dict[SliceID.Aux_Waist_UnderBust_2] = global_ids
        self._input_dict[SliceID.UnderBust] = global_ids
        self._input_dict[SliceID.Aux_UnderBust_Bust_0] = global_ids
        self._input_dict[SliceID.Bust] = global_ids

        #TODO leg slices


    def _init_torso_inputs_mode(self):

        torso_ids = [SliceID.Crotch, SliceID.Aux_Crotch_Hip_0,SliceID.Aux_Crotch_Hip_1, SliceID.Aux_Crotch_Hip_2, SliceID.Hip] + \
                    [SliceID.Aux_Hip_Waist_0, SliceID.Aux_Hip_Waist_1, SliceID.Waist] + \
                    [SliceID.Aux_Waist_UnderBust_0, SliceID.Aux_Waist_UnderBust_1, SliceID.Aux_Waist_UnderBust_2] + \
                    [SliceID.UnderBust, SliceID.Aux_UnderBust_Bust_0, SliceID.Bust] + \
                    [SliceID.Armscye, SliceID.Aux_Armscye_Shoulder_0, SliceID.Shoulder]

        self._input_dict[SliceID.Hip] = torso_ids
        self._input_dict[SliceID.Aux_Crotch_Hip_2] = torso_ids
        self._input_dict[SliceID.Aux_Crotch_Hip_1] = torso_ids
        self._input_dict[SliceID.Aux_Crotch_Hip_0] = torso_ids
        self._input_dict[SliceID.Crotch] = torso_ids

        self._input_dict[SliceID.Aux_Hip_Waist_0] = torso_ids
        self._input_dict[SliceID.Aux_Hip_Waist_1] = torso_ids
        self._input_dict[SliceID.Waist] = torso_ids
        self._input_dict[SliceID.Aux_Waist_UnderBust_0] = torso_ids
        self._input_dict[SliceID.Aux_Waist_UnderBust_1] = torso_ids
        self._input_dict[SliceID.Aux_Waist_UnderBust_2] = torso_ids
        self._input_dict[SliceID.UnderBust] = torso_ids
        self._input_dict[SliceID.Aux_UnderBust_Bust_0] = torso_ids
        self._input_dict[SliceID.Bust] = torso_ids


