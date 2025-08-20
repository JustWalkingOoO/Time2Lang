import numpy as np
import torch
from model.revin import RevIN


def codes2time(code_ids, codebook, compression_factor, vqvae_decoder, revin_layer):
    bs = code_ids.shape[0]
    nvars = code_ids.shape[1]
    compressed_len = code_ids.shape[2]
    num_code_words = codebook.shape[0]
    code_dim = codebook.shape[1]
    device = code_ids.device
    input_shape = (bs * nvars, compressed_len, code_dim)

    with torch.no_grad():
        # scatter the label with the codebook
        one_hot_encodings = torch.zeros(int(bs * nvars * compressed_len), num_code_words, device=device)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        
        code_ids[code_ids > 255] = 255
        code_ids[code_ids < 0] = 0

        one_hot_encodings.scatter_(1, code_ids.reshape(-1, 1).to(device),1)  # one_hot_encodings: [bs x nvars x compressed_pred_len, num_codes]
        quantized = torch.matmul(one_hot_encodings, torch.tensor(codebook).to(device)).view(input_shape)  # quantized: [bs * nvars, compressed_pred_len, code_dim]
        quantized_swaped = torch.swapaxes(quantized, 1,2)  # quantized_swaped: [bs * nvars, code_dim, compressed_pred_len]
        prediction_recon = vqvae_decoder(quantized_swaped.to(device), compression_factor)  # prediction_recon: [bs * nvars, pred_len]
        prediction_recon_reshaped = prediction_recon.reshape(bs, nvars, prediction_recon.shape[-1])  # prediction_recon_reshaped: [bs x nvars x pred_len]
        predictions_revin_space = torch.swapaxes(prediction_recon_reshaped, 1,2)  # prediction_recon_nvars_last: [bs x pred_len x nvars]
        predictions_original_space = revin_layer(predictions_revin_space, 'denorm')  # predictions:[bs x pred_len x nvars]

    return predictions_revin_space, predictions_original_space

def calculate_overall_mae(array1, array2):
    assert array1.shape == array2.shape, "Both arrays must be the same shape"
    abs_error = np.abs(array1 - array2)
    overall_mae = np.mean(abs_error)
    mae_per_sensor = np.mean(abs_error, axis=(0, 1))
    
    return overall_mae, mae_per_sensor

def sample_data_by_step(data, step):
    return data[::step]

def calculate_overall_mse(array1, array2):
    assert array1.shape == array2.shape, "Both arrays must be the same shape"
    squared_error = np.square(array1 - array2)
    overall_mse = np.mean(squared_error)
    mse_per_sensor = np.mean(squared_error, axis=(0, 1))
    
    return overall_mse, mse_per_sensor

if __name__ == "__main__":
    # datatset name
    dataset_name = "us_births"

    # change to your root path
    root_path = "/root/autodl-tmp/tokendata/"
    trained_vqvae_model_path = root_path + "trained_vqvae/generatlist_vqvae/checkpoints/final_model.pth"
    device = 'cuda:' + str(0)
    vqvae_model = torch.load(trained_vqvae_model_path, weights_only=False, map_location=device)
    vqvae_model.eval()

    # original data
    timey_file = root_path + "test_data/" + dataset_name + "/Tin96_Tout96/test_y_original.npy"
    timey = np.load(timey_file)
    step =3
    timey = sample_data_by_step(timey, step)

    # predict data
    pred_file = f"./token_pred_result/zero-shot/ChatTime/test_pred_y_codes_ChatTime_{dataset_name}_remap.npy"
    # pred_file = f"./token_pred_result/zero-shot/ChatTime/test_pred_y_codes_ChatTime_{dataset_name}.npy"
    pred_code_ids = np.load(pred_file)
    print(pred_code_ids.shape)

    nvars = timey.shape[2]
    bs = pred_code_ids.shape[0] // nvars
    print(bs)
    pred_code_ids = pred_code_ids[:bs*nvars].reshape(bs, nvars, -1)

    # codebook
    codebook_file = root_path + "test_data/" + dataset_name + "/Tin96_Tout96/codebook.npy"
    codebook = np.load(codebook_file)
    compression_factor = 4

    pred_y = []
    for i in range(len(pred_code_ids)):
        batch_y = timey[i:i+1]

        revin_layer_y = RevIN(num_features=timey.shape[2], affine=False, subtract_last=False)
        batch_y_tensor = torch.from_numpy(batch_y).float().to(device)
        y_in_revin_space = revin_layer_y(batch_y_tensor, "norm")

        batch_pred_code_ids = torch.from_numpy(pred_code_ids[i:i+1]).to(device)
        y_predictions_revin_space, y_predictions_original_space = codes2time(batch_pred_code_ids, codebook, compression_factor, vqvae_model.decoder, revin_layer_y)

        pred_y.append(np.array(y_predictions_original_space.detach().cpu()))


    pred_y_arr = np.concatenate(pred_y, axis=0)
    np.save("./token_pred_result/test_pred_y_result.npy", pred_y_arr)

    # mae
    print(timey.shape)
    print(pred_y_arr.shape)
    y = timey[:, :, :]
    x = pred_y_arr[:, :, :]
    print(y.shape)
    print(x.shape)
    mae, mae_per_sensor = calculate_overall_mae(y, x)
    print(f"Overall Mean Absolute Error (MAE): {mae}")
    print(f"Mean Absolute Error (MAE) per sensor: {mae_per_sensor}")

    # mse
    mse, mse_per_sensor = calculate_overall_mse(y, x)
    print(f"Overall Mean Squared Error (MSE): {mse}")
    print(f"Mean Squared Error (MSE) per sensor: {mse_per_sensor}")