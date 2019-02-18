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

    Elbow = ()
    Wrist = ()

class SliceModelInputDef:

    input_dict = {}

    tmp = [SliceID.Crotch.name, SliceID.Aux_Crotch_Hip_0.name, SliceID.Aux_Crotch_Hip_1.name, SliceID.Aux_Crotch_Hip_2.name, SliceID.Hip.name]
    input_dict[SliceID.Hip.name] = tmp
    input_dict[SliceID.Aux_Crotch_Hip_2.name] = tmp
    input_dict[SliceID.Aux_Crotch_Hip_1.name] = tmp
    input_dict[SliceID.Aux_Crotch_Hip_0.name] = tmp
    input_dict[SliceID.Crotch.name] = tmp

    input_dict[SliceID.Aux_Hip_Waist_0.name]         = [SliceID.Hip.name, SliceID.Waist.name, SliceID.Bust.name] + [SliceID.Aux_Hip_Waist_0.name]
    input_dict[SliceID.Aux_Hip_Waist_1.name]         = [SliceID.Hip.name, SliceID.Waist.name, SliceID.Bust.name] + [SliceID.Aux_Hip_Waist_1.name]
    input_dict[SliceID.Waist.name]                   = [SliceID.Hip.name, SliceID.Waist.name, SliceID.Bust.name]
    input_dict[SliceID.Aux_Waist_UnderBust_0.name]   = [SliceID.Hip.name, SliceID.Waist.name, SliceID.Bust.name] + [SliceID.Aux_Waist_UnderBust_0.name]
    input_dict[SliceID.Aux_Waist_UnderBust_1.name]   = [SliceID.Hip.name, SliceID.Waist.name, SliceID.Bust.name] + [SliceID.Aux_Waist_UnderBust_1.name]
    input_dict[SliceID.Aux_Waist_UnderBust_2.name]   = [SliceID.Hip.name, SliceID.Waist.name, SliceID.Bust.name] + [SliceID.Aux_Waist_UnderBust_2.name]
    input_dict[SliceID.UnderBust.name]               = [SliceID.Hip.name, SliceID.Waist.name, SliceID.Bust.name] #test
    input_dict[SliceID.Aux_UnderBust_Bust_0.name]    = [SliceID.Hip.name, SliceID.Waist.name, SliceID.Bust.name] + [SliceID.Aux_UnderBust_Bust_0.name]
    input_dict[SliceID.Bust.name]                    = [SliceID.Hip.name, SliceID.Waist.name, SliceID.Bust.name]

    @staticmethod
    def get_input_def(name):
        if name not in SliceModelInputDef.input_dict:
            return [name]
        else:
            return SliceModelInputDef.input_dict[name]
