from model.model import ChatTime
import numpy as np
from tqdm import tqdm
import numpy as np
import json
from tqdm import tqdm

def load_remap_table(path="remap_table.json"):
    with open(path, "r") as f:
        return json.load(f)

def remap_sequences(sequences, remap_table):
    remap_vectorized = np.vectorize(lambda x: remap_table.get(str(x), x))
    return remap_vectorized(sequences)

def inverse_remap_sequences(sequences, remap_table):
    inverse_map = {int(v): int(k) for k, v in remap_table.items()}
    inverse_vectorized = np.vectorize(lambda x: inverse_map.get(x, x))
    return inverse_vectorized(sequences)


def sample_data_by_step(data, step):
    return data[::step]

if __name__ == "__main__":
    # dataset name
    dataset_name = "us_births"

    # change to your root path
    root_path = "/root/autodl-tmp/tokendata/"
    timex_file = root_path + "test_data/" + dataset_name + "/Tin96_Tout96/test_x_codes.npy"
    timex = np.load(timex_file)
    print(timex.shape)

    sample_ratio = 0.15
    support = 29

    test_sample_ratio = 0.2
    index = int(len(timex) * test_sample_ratio)
    timex = timex[index:]  

    step = 3
    timex = sample_data_by_step(timex, step)
    timex = timex.reshape(-1, timex.shape[1])
    print(timex.shape)

    # <----------------- load remap table ----------------->
    remap_table_file = "/root/myapp/ChatTime-main/token_pred_result/remap_table_few_shot/" \
                    f"{dataset_name}/remap_table_support{support}_sample{sample_ratio}.json"
    # remap_table_file = f"/root/myapp/ChatTime-main/token_pred_result/remap_table_3.json"
    # remap_table_file = f"/root/myapp/ChatTime-main/token_pred_result/remap_table_{dataset_name}_support35.json"
    remap_table = load_remap_table(remap_table_file)
    timex = remap_sequences(timex, remap_table)
    # <----------------- load remap table ----------------->

    model_path = "ChengsenWang/ChatTime-1-7B-Chat"
    # model_path = "meta-llama/Llama-2-7b-hf"

    pred_len = 24
    hist_len = 24
    hist_data = timex
    # load model
    print("current model path: ", model_path)
    model = ChatTime(hist_len=hist_len, pred_len=pred_len, model_path=model_path, max_pred_len=6)

    # predict
    batch_size = 7
    result = []
    for i in tqdm(range(0, len(hist_data), batch_size), desc="Predict time series"):
        batch = hist_data[i: i + batch_size]
        out = model.predict_token2token_batch(batch)
        print("out:", out)
        result.append(out)
    result = np.concatenate(result, axis=0)
    pred_data = np.array(result)

    # <----------------- inverse remapping ----------------->
    pred_data = inverse_remap_sequences(pred_data, remap_table)
    # <----------------- inverse remapping ----------------->

    # save_path = "/root/myapp/ChatTime-main/token_pred_result/few-shot/ChatTime/remap_itself/" \
    #             f"sample{sample_ratio}/test_pred_y_codes_{dataset_name}.npy"
    # save_path = "/root/myapp/ChatTime-main/token_pred_result/few-shot/ChatTime/remap_3/" \
    #             f"sample{sample_ratio}/test_pred_y_codes_{dataset_name}.npy"
    save_path = "/root/myapp/ChatTime-main/token_pred_result/few-shot/ChatTime/remap_itself_0.2test/" \
                f"sample{sample_ratio}/test_pred_y_codes_{dataset_name}.npy"
    np.save(save_path, pred_data)
    print("current model path: ", model_path)
