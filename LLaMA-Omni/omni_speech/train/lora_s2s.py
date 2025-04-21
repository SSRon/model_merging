import argparse, os, json, math
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import whisper
from transformers import TrainingArguments, Trainer, AutoTokenizer
from omni_speech.conversation import conv_templates
from omni_speech.datasets.preprocess import tokenizer_speech_token
from omni_speech.model.builder import create_model_lora  

# Collate function
def collate_fn(batch):
    input_ids, labels, speech_tensors, speech_lengths = zip(*batch)
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=128009)
    labels = pad_sequence(labels, batch_first=True, padding_value=128009)
    speech_tensors = torch.stack(speech_tensors)
    speech_lengths = torch.stack(speech_lengths)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "speech": speech_tensors,
        "speech_lengths": speech_lengths
    }

# Dataset
class CustomDataset(Dataset):
    def __init__(self, data, tokenizer, model_config, input_type, mel_size, conv_mode):
        self.data = data
        self.tokenizer = tokenizer
        self.model_config = model_config
        self.input_type = input_type
        self.mel_size = mel_size
        self.conv_mode = conv_mode

    def __getitem__(self, idx):
        item = self.data[idx]
        speech_path = item["speech"]
        user_msg = item["conversations"][0]["value"]
        target_msg = item["conversations"][1]["value"]

        conv = conv_templates[self.conv_mode].copy()
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], target_msg)
        prompt = conv.get_prompt()

        speech = whisper.load_audio(speech_path)
        if self.input_type == "mel":
            speech = whisper.pad_or_trim(speech)
            speech = whisper.log_mel_spectrogram(speech, n_mels=self.mel_size).permute(1, 0)
        else:
            raise NotImplementedError("Only mel input is supported.")

        input_ids = tokenizer_speech_token(prompt, self.tokenizer, return_tensors="pt")
        return input_ids, input_ids.clone(), speech.to(torch.bfloat16), torch.tensor([speech.shape[0]])

    def __len__(self):
        return len(self.data)

def load_json(path):
    with open(path) as f:
        return json.load(f)

def train(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tokenizer, model, context_len = create_model_lora(
        model_path=args.model_path,
        model_base=args.model_base,
        is_lora=True,
        device=device,
    )

    # Dataset
    train_data = load_json(args.train_file)
    eval_data = load_json(args.eval_file)

    train_dataset = CustomDataset(train_data, tokenizer, model.config, args.input_type, args.mel_size, args.conv_mode)
    eval_dataset = CustomDataset(eval_data, tokenizer, model.config, args.input_type, args.mel_size, args.conv_mode)

    args.output_dir = os.path.join(args.output_dir, args.language)
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Trainer
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        num_train_epochs=args.epochs,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        logging_steps=1,
        learning_rate=1e-4,
        warmup_ratio=0.01,
        lr_scheduler_type='cosine',
        bf16=True,
        report_to="none"
    )

    tokenizer.pad_token = tokenizer.eos_token
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collate_fn,
    )

    trainer.train()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str)
    parser.add_argument("--model-base", type=str)
    parser.add_argument("--language", type=str, required=True, help="Language tag (e.g. de, fr, zh)")
    parser.add_argument("--train-file", type=str, required=True)
    parser.add_argument("--eval-file", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--input_type", type=str, default="mel")
    parser.add_argument("--mel_size", type=int, default=128)
    parser.add_argument("--conv_mode", type=str, default="llama_3")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--grad_accum_steps", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    args = parser.parse_args()
    train(args)
