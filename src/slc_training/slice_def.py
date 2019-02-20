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

    def find_enum(name_id):
        for name, member in SliceID.__members__.items():
            if name == name_id:
                return member
        return None

class SliceModelInputDef:

    input_dict = {}

    input_dict[SliceID.Hip] = [SliceID.Aux_Crotch_Hip_2, SliceID.Hip, SliceID.Aux_Hip_Waist_0]
    input_dict[SliceID.Aux_Crotch_Hip_2] = [SliceID.Aux_Crotch_Hip_1, SliceID.Aux_Crotch_Hip_2, SliceID.Hip]
    input_dict[SliceID.Aux_Crotch_Hip_1] = [SliceID.Aux_Crotch_Hip_0, SliceID.Aux_Crotch_Hip_1, SliceID.Aux_Crotch_Hip_2]
    input_dict[SliceID.Aux_Crotch_Hip_0] = [SliceID.Crotch, SliceID.Aux_Crotch_Hip_0, SliceID.Aux_Crotch_Hip_1]
    input_dict[SliceID.Crotch] = [SliceID.Crotch, SliceID.Aux_Crotch_Hip_0, SliceID.Aux_Crotch_Hip_1]

    input_dict[SliceID.Aux_Hip_Waist_0]         = [SliceID.Hip,SliceID.Aux_Hip_Waist_0, SliceID.Aux_Hip_Waist_1]
    input_dict[SliceID.Aux_Hip_Waist_1]         = [SliceID.Aux_Hip_Waist_0, SliceID.Aux_Hip_Waist_1, SliceID.Waist]
    input_dict[SliceID.Waist]                   = [SliceID.Aux_Hip_Waist_1, SliceID.Waist, SliceID.Aux_Waist_UnderBust_0]
    input_dict[SliceID.Aux_Waist_UnderBust_0]   = [SliceID.Waist, SliceID.Aux_Waist_UnderBust_0, SliceID.Aux_Waist_UnderBust_1]
    input_dict[SliceID.Aux_Waist_UnderBust_1]   = [SliceID.Aux_Waist_UnderBust_0, SliceID.Aux_Waist_UnderBust_1, SliceID.Aux_Waist_UnderBust_2]
    input_dict[SliceID.Aux_Waist_UnderBust_2]   = [SliceID.Aux_Waist_UnderBust_1, SliceID.Aux_Waist_UnderBust_2, SliceID.UnderBust]
    input_dict[SliceID.UnderBust]               = [SliceID.Aux_Waist_UnderBust_2, SliceID.UnderBust, SliceID.Aux_UnderBust_Bust_0] #test
    input_dict[SliceID.Aux_UnderBust_Bust_0]    = [SliceID.UnderBust, SliceID.Aux_UnderBust_Bust_0, SliceID.Bust]
    input_dict[SliceID.Bust]                    = [SliceID.Aux_UnderBust_Bust_0, SliceID.Bust, SliceID.Armscye]

    @staticmethod
    def get_input_def(name):
        id = SliceID[name]
        if id not in SliceModelInputDef.input_dict:
            return [name]
        else:
            return [en.name for en in SliceModelInputDef.input_dict[id]]
