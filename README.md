### Install
Please install the requirements first:
(You can find more information in https://github.com/ictnlp/LLaMA-Omni)
1. LLaMA-Omni
```bash
cd LLaMA-Omni
pip install pip==24.0
pip install -e .
```
2. fairseq
```bash
cd fairseq
pip install -e . --no-build-isolation
```
3. flash-attention
```bash
pip install packaging ninja (Please make sure you have these two)
MAX_JOBS=4 pip install flash-attn==2.7.3 --no-build-isolation
```
### Required Models:
1. Speech Encoder:
The model should be put under "LLaMA-Omni/omni_speech/model/speech_encoder"
```bash
import whisper
model = whisper.load_model("large-v3", download_root="models/speech_encoder/")
```
2. LLaMA-Omni(wherever you want):
```bash
git clone https://huggingface.co/ICTNLP/Llama-3.1-8B-Omni
```
3. Vocoder(wherever you want):
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/g_00500000 -P vocoder/
wget https://dl.fbaipublicfiles.com/fairseq/speech_to_speech/vocoder/code_hifigan/mhubert_vp_en_es_fr_it3_400k_layer11_km1000_lj/config.json -P vocoder/

### Data
Please go to (https://drive.google.com/file/d/1xBCV83E2zlOoAQ-tZSzLNuVhJEElN5x6/view?usp=sharing) and download the "data.tar.gz".
After extracting the data, please find the "hf_processed" for the "train.json", "dev.json", and "test.json" used for fine-tuning.
The audio files are put under the "origin/covost2_de/clips" folder

### Commands
1. Generate the outputs:

   **"--answer-file": where output json will be saved.**
   **--s2s: output json will include discrete units which will then be used for speech encoder**
```bash
cd /gscratch/tial/andysu/model_merging/

python "LLaMA-Omni/omni_speech/infer/infer.py" --model-path "Llama-3.1-8B-Omni" --question-file "data/hf_processed/de/test.json" --answer-file "experiment_results/s2s/test_answer_s2s.json" --s2s --conv-mode "llama_3" --input_type "mel" --max_new_tokens 256

python "LLaMA-Omni/omni_speech/infer/convert_jsonl_to_txt.py" "experiment_results/s2s/test_answer_s2s.json" "experiment_results/s2s/test_answer_s2s.unit"

python "LLaMA-Omni/fairseq/examples/speech_to_speech/generate_waveform_from_code.py" --in-code-file "experiment_results/s2s/test_answer_s2s.unit" --vocoder vocoder/g_00500000 --vocoder-cfg vocoder/config.json --results-path "experiment_results/s2s/answer_wav/" --dur-prediction
```
2. Fine-tuning:
```bash
python "LLaMA-Omni/omni_speech/train/lora_s2s.py" --model-base "Llama-3.1-8B-Omni" --train-file "data/hf_processed/de/train.json" --eval-file "data/hf_processed/de/dev.json" --output-dir "Llama-3.1-8B-Omni/lora_ft" --language de --epochs 1
```

