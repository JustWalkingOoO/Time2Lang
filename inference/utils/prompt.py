GEN_SYS_PROMPT = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
PRE_SYS_PROMPT = "You are a helpful assistant that performs time series prediction. The user will provide a sequence and you will predict the sequence."
ANA_SYS_PROMPT = "You are a helpful assistant that performs time series analysis. The user will provide a sequence and you will respond to the questions based on this sequence."
#add
DIS_PRE_SYS_PROMPT = "You are a helpful assistant that performs time series prediction. The user will provide some sequences and each sequence is separated by a semicolon \";\". You will predict the sequence after \"Prediction Input\" based on all the sequences provided by the user."
T2T_SYS_PROMPT = "You are a helpful assistant that performs time series prediction. The user will provide a sequence and you will predict the sequence. \
                    The sequence provided by the user is a time series that has been discretized. \
                    We have previously agreed to use natural numbers: 0, 1, 2, 3, 4, etc., to represent individual discrete tokens. \
                    Therefore, the sequence provided by the user is a token sequence. You are required to predict and output a token sequence as well."

PRE_INST_PROMPT = "Please predict the following sequence carefully."
PRE_INST_PROMPT_TEXT = "Please predict the following sequence carefully. Context knowledge you may consider: {}"
ANA_INST_PROMPT_TEXT = "Please answer the following question carefully after analyzing the sequence: {}"
# add
DIS_INST_SYS_PROMPT = "Please predict the sequence after \"Prediction Input\" based on all input carefully."
PRE_INST_PROMPT_TEXT = "Please predict the following sequence carefully. Context knowledge you may consider: {}. They are the trend, seasonality, and residual components obtained from the STL decomposition of the sequences in \"Input\"."
T2T_INST_PROMPT_TEXT = "Please predict the following token sequence carefully, and output a token sequence as well. This sequence consists of several natural numbers, and the range of these natural numbers is between 0 and 255."

TEMPLATE = """{}

### Instruction:
{}

### Input:
{}

### Response:
{}"""

TEMPLATE_DIS = """{}

### Instruction:
{}

### Prediction Input:
{}

### Other sequences Input:
{}

### Response:
{}"""


def getPrompt(flag, instruction=None, input=None, input_other=None, response=None, context=None):
    if flag == "general":  # instruction, [input], [response]
        system = GEN_SYS_PROMPT
        if instruction is None:
            raise ValueError("Instruction must be provided for general tasks.")
        else:
            instruction = instruction
        input = "" if input is None else input
        response = "" if response is None else response

    elif flag == "prediction":  # [context], input, [response]
        system = PRE_SYS_PROMPT
        instruction = PRE_INST_PROMPT if context is None else PRE_INST_PROMPT_TEXT.format(context)
        if input is None:
            raise ValueError("Input must be provided for prediction tasks.")
        else:
            input = input
        response = "" if response is None else response

    elif flag == "analysis":  # instruction, input, [response]
        system = ANA_SYS_PROMPT
        if instruction is None:
            raise ValueError("Instruction must be provided for analysis tasks.")
        else:
            instruction = ANA_INST_PROMPT_TEXT.format(instruction)
        if input is None:
            raise ValueError("Input must be provided for analysis tasks.")
        else:
            input = input
        response = "" if response is None else response

    elif flag == "distortion_prediction":
        system = DIS_PRE_SYS_PROMPT
        instruction = DIS_INST_SYS_PROMPT
        if input is None:
            raise ValueError("Input must be provided for distortion prediction tasks.")
        else:
            input = input
        response = "" if response is None else response

        prompt = TEMPLATE_DIS.format(system, instruction, input, input_other, response)
        return prompt
    
    elif flag == "prediction_with_STL":  # [context], input, [response]
        system = PRE_SYS_PROMPT
        instruction = PRE_INST_PROMPT if context is None else PRE_INST_PROMPT_TEXT.format(context)
        if input is None:
            raise ValueError("Input must be provided for prediction tasks.")
        else:
            input = input
        response = "" if response is None else response

    elif flag == "prediction-token2token":  # [context], input, [response]
        system = T2T_SYS_PROMPT
        instruction = T2T_INST_PROMPT_TEXT
        if input is None:
            raise ValueError("Input must be provided for prediction tasks.")
        else:
            input = input
        response = "" if response is None else response

    else:
        raise ValueError("Flag must be one of 'general', 'prediction', or 'analysis'.")

    prompt = TEMPLATE.format(system, instruction, input, response)

    return prompt
