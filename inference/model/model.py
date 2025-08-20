import re
from statistics import mode

import numpy as np
import torch
from transformers import (
    LlamaForCausalLM,
    LlamaTokenizer,
    pipeline,
)

from utils.prompt import getPrompt
from utils.tools import Discretizer, Serializer

from statsmodels.tsa.seasonal import STL
from datasets import Dataset
from transformers.pipelines.pt_utils import KeyDataset


class ChatTime:
    def __init__(self, model_path, hist_len=None, pred_len=None,
                 max_pred_len=16, num_samples=8, top_k=100, top_p=1.0, temperature=1.0):
        self.model_path = model_path
        self.hist_len = hist_len
        self.pred_len = pred_len

        self.max_pred_len = max_pred_len
        self.num_samples = num_samples
        self.top_k = top_k
        self.top_p = top_p
        self.temperature = temperature

        self.discretizer = Discretizer()
        self.serializer = Serializer()

        self.model = LlamaForCausalLM.from_pretrained(
            self.model_path,
            low_cpu_mem_usage=True,
            return_dict=True,
            torch_dtype=torch.float16,
            device_map="auto",
            load_in_8bit=True
        )

        self.tokenizer = LlamaTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        self.eos_token_id = self.tokenizer.eos_token_id

    def predict(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        series = hist_data
        prediction_list = []
        remaining = self.pred_len

        while remaining > 0:
            dispersed_series = self.discretizer.discretize(series)
            serialized_series = self.serializer.serialize(dispersed_series)
            serialized_series = getPrompt(flag="prediction", context=context, input=serialized_series)

            pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                min_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                max_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                do_sample=True,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
            )
            samples = pipe(serialized_series)

            pred_list = []
            for sample in samples:
                serialized_prediction = sample["generated_text"].split("### Response:\n")[1]
                dispersed_prediction = self.serializer.inverse_serialize(serialized_prediction)
                pred = self.discretizer.inverse_discretize(dispersed_prediction)

                if len(pred) < min(remaining, self.max_pred_len):
                    pred = np.concatenate([pred, np.full(min(remaining, self.max_pred_len) - len(pred), np.nan)])

                pred_list.append(pred[:min(remaining, self.max_pred_len)])

            prediction = np.nanmedian(pred_list, axis=0)
            prediction_list.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_list, axis=-1)

        return prediction

    def analyze(self, question, series):
        dispersed_series = self.discretizer.discretize(series)
        serialized_series = self.serializer.serialize(dispersed_series)
        serialized_series = getPrompt(flag="analysis", instruction=question, input=serialized_series)

        pipe = pipeline(
            task="text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            max_new_tokens=self.max_pred_len,
            do_sample=True,
            num_return_sequences=self.num_samples,
            top_k=self.top_k,
            top_p=self.top_p,
            temperature=self.temperature,
            eos_token_id=self.eos_token_id,
        )
        samples = pipe(serialized_series)

        response_list = []
        for sample in samples:
            response = sample["generated_text"].split("### Response:\n")[1].split('.')[0] + "."
            response = re.findall(r"\([abc]\)", response)[0]
            response_list.append(response)

        response = mode(response_list)

        return response


    def predict_token2token(self, hist_data, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        series = hist_data
        prediction_list = []
        remaining = self.pred_len

        while remaining > 0:
            # 修改getPrompt
            separator = " "
            input_series = separator.join(str(x) for x in series)
            serialized_series = getPrompt(flag="prediction-token2token", context=context, input=input_series)

            pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                min_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                max_new_tokens=2 * min(remaining, self.max_pred_len) + 8,
                do_sample=True,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
            )
            samples = pipe(serialized_series)

            pred_list = []
            for sample in samples:
                serialized_prediction = sample["generated_text"].split("### Response:\n")[1]
                pred = serialized_prediction
                # print("serialized_prediction:", serialized_prediction)
                pred = np.fromstring(pred, sep=' ')
                pred = pred.astype(int)
                

                if len(pred) < min(remaining, self.max_pred_len):
                    pred = np.concatenate([pred, np.full(min(remaining, self.max_pred_len) - len(pred), 0)])

                pred_list.append(pred[:min(remaining, self.max_pred_len)])
                # print("pred:", pred)

            # prediction = np.nanmedian(pred_list, axis=0)
            # 检查pred_list中的元素类型
            if all(isinstance(p, np.ndarray) for p in pred_list):
                # 如果都是numpy数组，使用np.nanmedian
                prediction = np.nanmedian(pred_list, axis=0)
            else:
                # 如果是整数列表或其他类型，使用普通median
                prediction = np.median(pred_list, axis=0)

            prediction = prediction.astype(int)

            prediction_list.append(prediction)
            remaining -= prediction.shape[-1]

            if remaining <= 0:
                break

            series = np.concatenate([series, prediction], axis=-1)

        prediction = np.concatenate(prediction_list, axis=-1)

        print("hist_data:", hist_data)
        print("prediction:", prediction)

        return prediction
    
    def predict_token2token_batch(self, hist_data_batch, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        batch_size = hist_data_batch.shape[0]
        # eries_batch = hist_data_batch.copy()  # shape: (B, T)
        series_batch = [hist_data_batch[b].copy() for b in range(batch_size)]
        prediction_batch = [np.empty((0,), dtype=int) for _ in range(batch_size)]
        remaining = self.pred_len

        while remaining > 0:
            step = min(remaining, self.max_pred_len)

            pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                do_sample=True,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
                max_new_tokens=2 * step + 8,
                min_new_tokens=2 * step + 8,
                num_return_sequences=self.num_samples,
                batch_size=batch_size,
            )

            # 生成 batch 的 prompt
            prompts = []
            for b in range(batch_size):
                input_series = " ".join(str(x) for x in series_batch[b])
                prompt = getPrompt(flag="prediction-token2token", context=context, input=input_series)
                prompts.append(prompt)

            # ✅ 2. 转为 Hugging Face Dataset 加速
            prompt_ds = Dataset.from_dict({"text": prompts})

            # 将 Dataset 转成 KeyDataset，用于 pipeline 批量处理
            key_ds = KeyDataset(prompt_ds, "text")
            
            # 输出 shape: (batch_size, num_return_sequences)
            outputs = pipe(key_ds)

            pred_batch = [[] for _ in range(batch_size)]

            for b_idx, sample_list in enumerate(outputs):
                i = 0
                for sample in sample_list:
                    try:
                        text = sample["generated_text"]
                        idx = text.find("### Response:\n")
                        if idx == -1:
                            continue

                        serialized_prediction = text[idx + len("### Response:\n"):].strip()
                        pred = np.fromstring(serialized_prediction, sep=' ', dtype=int)

                        # print("serialized_prediction:", serialized_prediction, "pred:", pred)
                        if len(pred) == 0:
                            continue
                        if len(pred) < step:
                            pad_len = step - len(pred)
                            pred = np.pad(pred, (0, pad_len), constant_values=0)

                        pred_batch[b_idx].append(pred[:step])
                        
                    except Exception as e:
                        print(f"[Warning] Skipped one sample due to error: {e}")
                        continue

            # 聚合每个样本的多个预测结果
            for b in range(batch_size):
                if len(pred_batch[b]) == 0:
                    pred_batch[b].append(np.zeros((step,), dtype=int))

                pred_b = np.nanmedian(pred_batch[b], axis=0).astype(int)
                # print("pred_b:", pred_b)
                # print("prediction_batch[b]  :", prediction_batch[b])
                prediction_batch[b] = np.concatenate([prediction_batch[b], pred_b], axis=0)
                series_batch[b] = np.concatenate([series_batch[b], pred_b], axis=0)

            remaining -= step

        return np.stack(prediction_batch, axis=0)  # shape: (B, pred_len)
    
    def predict_batch(self, hist_data_batch, context=None):
        if self.hist_len is None or self.pred_len is None:
            raise ValueError("hist_len and pred_len must be specified before prediction")

        batch_size = hist_data_batch.shape[0]
        series_batch = [hist_data_batch[b].copy() for b in range(batch_size)]
        prediction_batch = [np.empty((0,), dtype=float) for _ in range(batch_size)]
        remaining = self.pred_len

        while remaining > 0:
            step = min(remaining, self.max_pred_len)

            # 生成每个样本的 prompt
            prompts = []
            for b in range(batch_size):
                dispersed_series = self.discretizer.discretize(series_batch[b])
                serialized_series = self.serializer.serialize(dispersed_series)
                prompt = getPrompt(flag="prediction", context=context, input=serialized_series)
                prompts.append(prompt)

            prompt_ds = Dataset.from_dict({"text": prompts})
            key_ds = KeyDataset(prompt_ds, "text")

            pipe = pipeline(
                task="text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                min_new_tokens=2 * step + 8,
                max_new_tokens=2 * step + 8,
                do_sample=True,
                num_return_sequences=self.num_samples,
                top_k=self.top_k,
                top_p=self.top_p,
                temperature=self.temperature,
                eos_token_id=self.eos_token_id,
                batch_size=batch_size,
            )

            outputs = pipe(key_ds)  # shape: (batch_size, num_return_sequences)
            pred_batch = [[] for _ in range(batch_size)]

            for b_idx, sample_list in enumerate(outputs):
                for sample in sample_list:
                    try:
                        serialized_prediction = sample["generated_text"].split("### Response:\n")[1]
                        dispersed_prediction = self.serializer.inverse_serialize(serialized_prediction)
                        pred = self.discretizer.inverse_discretize(dispersed_prediction)

                        if len(pred) < step:
                            pred = np.concatenate([pred, np.full(step - len(pred), np.nan)])

                        pred_batch[b_idx].append(pred[:step])
                    except Exception as e:
                        print(f"[Warning] Sample skipped for batch {b_idx}: {e}")
                        continue

            # 聚合预测并拼接
            for b in range(batch_size):
                if len(pred_batch[b]) == 0:
                    pred_batch[b].append(np.full(step, np.nan))

                pred_b = np.nanmedian(pred_batch[b], axis=0)
                prediction_batch[b] = np.concatenate([prediction_batch[b], pred_b], axis=0)
                series_batch[b] = np.concatenate([series_batch[b], pred_b], axis=0)

            remaining -= step

        return np.stack(prediction_batch, axis=0)  # shape: (batch_size, pred_len)


