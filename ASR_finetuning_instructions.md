# ASR Finetuning Guide: Bangla & Banglish (Code-Switching)

This document provides a comprehensive analysis and step-by-step guide for fine-tuning ASR models for your specific use case (Bangla/English mixed "Banglish").

## 1. Model Selection & Verification

### **Hypothesis Verification**
> **Your Hypothesis:** "Finetuning can make CTC competitive... CTC can learn it if training data shows proper script separation."

**Verdict: CORRECT & RECOMMENDED**

*   **Why CTC (Connectionist Temporal Classification) is better for you:**
    1.  **Data Efficiency:** CTC models (like Wav2Vec2/MMS-1B) are strictly acoustic models. They learn to map sound to characters directly. They are less prone to "hallucination" than LLM-based decoders when training data is scarce (< 100 examples).
    2.  **Code-Switching:** CTC handles code-switching naturally as long as the vocabulary contains both scripts (Bengali and Latin). It doesn't rely on a complex language model that might bias it towards "pure" Bangla.
    3.  **Hardware Friendly:** You can fine-tune a 1B CTC model on your GPUs (see below) much more easily than the "LLM" variants.

*   **Why NOT the "LLM" model (Wav2Vec2 + LLaMA) for now:**
    *   **Architecture:** The `omniASR_LLM_1B` appears to be a composite of a Wav2Vec2 encoder + a LLaMA-style decoder (likely ~2B-3B params total).
    *   **Training Difficulty:** Training this requires backpropagation through the decoder. With only 20-100 examples, the decoder will likely **overfit** or become unstable (forgetting the pre-trained knowledge) or hallucinate.
    *   **VRAM Usage:** It is significantly heavier and might not fit on the 8GB GPU even with LoRA without aggressive optimization (4-bit quantization).

**Recommendation:** Stick to **`omniASR-CTC-1B`** (or its Hugging Face equivalent `facebook/mms-1b-all`) for your fine-tuning experiments.

---

## 2. Hardware Capability & GPU Selection

You have:
*   **PC 1:** RTX 2070? (8GB VRAM typically).
*   **PC 2:** RTX 2060 12GB.

### **Finetuning `CTC-1B` (1 Billion Parameters)**

| Method | VRAM Estimate | PC 1 (8GB) | PC 2 (12GB) | Speed |
| :--- | :--- | :--- | :--- | :--- |
| **Full Finetuning** | ~14-16 GB | ❌ No | ❌ No (Too tight) | Fast |
| **LoRA (PEFT)** | ~6-8 GB | ✅ **Yes** | ✅ **Yes** | Fast |
| **Frozen Encoder** | ~4-6 GB | ✅ **Yes** | ✅ **Yes** | Fastest |

*   **Conclusion:** You **CAN** fine-tune the CTC 1B model on **both** machines if you use **LoRA (Low-Rank Adaptation)** or **freeze the feature encoder**.
*   **Best Practice:** Use the **12GB GPU (RTX 2060)**. The extra VRAM allows for larger batch sizes (more stable gradients) and prevents OOM (Out of Memory) crashes.

---

## 3. Training / Finetuning Instructions

Since the `omnilingual-asr` repo you cloned (fairseq2 based) lacks a ready-to-use Training script (it is primarily configured for inference), the **industry standard best practice** is to use **Hugging Face Transformers**.

The `omniASR-CTC-1B` is essentially Meta's **MMS-1B** model (`facebook/mms-1b-all`).

### **Preparation**

1.  **Install Libraries** (in a new or existing environment):
    ```bash
    pip install transformers datasets accelerate peft bitsandbytes librosa torch evaluate jiwer
    ```

2.  **Data Preparation (Crucial for Banglish)**
    Create a dataset CSV/JSON with your 20-100 examples.
    
    *Structure:* `audio_path`, `sentence`
    
    **Text Normalization Rules (Best Practices):**
    *   **Bangla:** Keep standard spelling.
    *   **English/Banglish:** Decide on a convention. 
        *   Option A (Transliterated): "ami vat khai" (All Latin)
        *   Option B (Mixed Script): "আমি ভাত eat করি" (Mixed)
    *   **The model needs to learn your specific convention.** Ensure your training data matches how you want the output to look.
    *   **Vocab:** The MMS model usually supports both scripts.

### **Training Script (Template)**

Create a file named `finetune_mms.py` and accept the following template:

```python
import torch
from transformers import Wav2Vec2ForCTC, AutoProcessor, TrainingArguments, Trainer
from datasets import load_dataset, Audio
from peft import LoraConfig, get_peft_model
import numpy as np

# 1. Config
MODEL_ID = "facebook/mms-1b-all" # Equivalent to omniASR-CTC-1B
OUTPUT_DIR = "mms-1b-banglish-finetuned"
USE_LORA = True

# 2. Load Processor (Tokenizer + Feature Extractor)
processor = AutoProcessor.from_pretrained(MODEL_ID)
# Ensure the tokenizer has all characters you need. 
# If validation fails on new characters, you might need to add them to the tokenizer adapter:
# tokenizer = processor.tokenizer

# 3. Load Dataset
# Assuming you have a CSV with 'path' and 'sentence' columns
# dataset = load_dataset("csv", data_files="my_banglish_data.csv")
# For demo, using dummy data structure. REPLACE above line with your CSV load.
# dataset = dataset["train"].train_test_split(test_size=0.1) # Split for validation

# 4. Preprocessing
def prepare_dataset(batch):
    audio = batch["audio"]
    batch["input_values"] = processor(audio["array"], sampling_rate=16000).input_values[0]
    with processor.as_target_processor():
        batch["labels"] = processor(batch["sentence"]).input_ids
    return batch

# 5. Load Model
model = Wav2Vec2ForCTC.from_pretrained(
    MODEL_ID, 
    attention_dropout=0.1,
    hidden_dropout=0.1,
    feat_proj_dropout=0.1,
    target_lang="ben", # Initialize with Bengali adapter settings
    ignore_mismatched_sizes=True, # In case we resize vocab
    torch_dtype=torch.float16 # Half precision for GPU memory
)
model.init_adapter_layers() # Activate adapters if MMS uses them
model.freeze_base_model() # Optional: Freeze base to save memory, strictly finetune adapter/head

if USE_LORA:
    peft_config = LoraConfig(
        inference_mode=False, r=16, lora_alpha=32, lora_dropout=0.1, bias="none",
        target_modules=["q_proj", "v_proj"] # Target attention layers
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

# 6. Training Arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    group_by_length=True,
    per_device_train_batch_size=2, # Keep small for 8GB/12GB GPU
    gradient_accumulation_steps=4, # Effective batch size = 8
    evaluation_strategy="steps",
    num_train_epochs=10, # 10-30 epochs for small data
    fp16=True, # Essential for VRAM saving
    save_steps=50,
    eval_steps=50,
    logging_steps=10,
    learning_rate=3e-4, # Slightly higher for LoRA
    warmup_steps=50,
    save_total_limit=2,
    dataloader_num_workers=2,
)

# 7. Data Collator (Pads audio and labels dynamically)
from transformers import DataCollatorCTCWithPadding
data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

# 8. Train
# trainer = Trainer(
#     model=model,
#     data_collator=data_collator,
#     args=training_args,
#     train_dataset=dataset["train"],
#     eval_dataset=dataset["test"],
#     tokenizer=processor.feature_extractor,
# )

# trainer.train()
```

## 4. Best Practices for Your Use Case

1.  **Mixed Script Strategy:**
    *   Input: `Audio("আমি ভাত খাই")` -> Label: `"ami vat khai"` (if you want transliteration).
    *   Input: `Audio("I eat rice")` -> Label: `"I eat rice"` (if you want English).
    *   **Crucial:** Ensure your text labels strictly match what you said. If you used an English word in the audio but wrote the Bangla translation in the label, the model will get confused.

2.  **Audio Quality:**
    *   For 20-100 examples, clean audio is better.
    *   However, if you want robustness, add some background noise (office noise, street noise) to half of your examples (Data Augmentation).

3.  **Evaluation:**
    *   Don't trust the loss alone. Look at the generated transcriptions on your test set.
    *   Use **CER (Character Error Rate)** rather than WER (Word Error Rate) for Bangla/Banglish, as spelling variations (transliteration) can inflate WER.

## 5. Handling Numerics, Alphanumerics & Pauses (Advanced)

For your requirement of **English/Bangla digits (0-9)**, **Long IDs**, **Passport numbers**, and **Pauses**:

### **A. Tokenizer Configuration (Crucial)**
Standard ASR pipelines often normalize numbers to text (e.g., "10" -> "ten"). **You must disable this behavior** and ensuring your tokenizer recognizes digits.
1.  **Check Vocab:** The base MMS model might *not* have `0`, `1`...`9` in its vocabulary.
2.  **Add Tokens:** You must check and add them if missing.

**Updated Training Snippet (Add to `finetune_mms.py` before loading model):**
```python
# ... load processor ...
processor = AutoProcessor.from_pretrained(MODEL_ID)

# 1. Add Digits and Alphanumeric chars if missing
new_tokens = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
              "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M",
              "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Check if tokenizer has them, if not add them
missing_tokens = [t for t in new_tokens if t not in processor.tokenizer.get_vocab()]
if missing_tokens:
    print(f"Adding missing tokens: {missing_tokens}")
    processor.tokenizer.add_tokens(missing_tokens)
    # IMPORTANT: We will need to resize the model embeddings later
```
*And after loading the model:*
```python
model = Wav2Vec2ForCTC.from_pretrained(...)
if missing_tokens:
    model.resize_token_embeddings(len(processor.tokenizer))
```

### **B. Labeling Strategy for Pauses & Grouping**
CTC models are great at learning silence/pauses if you label them with **spaces** or **commas**.

*   **Scenario 1: Phone Numbers with pauses**
    *   *Speech:* "Zero one seven... [pause] ... two four six..."
    *   *Incorrect Label:* `017246` (Model learns to rush the output)
    *   *Correct Label:* `017 246` or `017, 246`
    *   **Recommendation:** Use **spaces** for human-like pauses. The model will learn to predict a space token when the speaker pauses.

*   **Scenario 2: Mixed Script Alphanumeric (Passport)**
    *   *Speech:* "Passport number... [pause] ... A zero two..."
    *   *Label:* `Passport number A02` (If spoken continuously)
    *   *Label:* `Passport number A 02` (If explicitly paused)

### **C. 500 Examples Optimization**
With 500 examples, you are entering the "safe zone" for more robust finetuning.
1.  **Unfreeze More Layers:** Instead of just the adapter/LoRA, you can try unfreezing the last 1-2 transformer layers of the encoder *after* initial convergence (though LoRA is still safer on 8GB VRAM).
2.  **Learning Rate:** You can increase the learning rate slightly (e.g., `1e-4` to `3e-4`) since you have more signal.
3.  **Epochs:** Increase to 20-40 epochs. With 500 items, one epoch is fast.

4.  **Local Training Workflow:**
    *   SSH into the **12GB GPU** machine.
    *   Run the script using `python finetune_mms.py`.
    *   Monitor VRAM with `watch -n 1 nvidia-smi`.
    *   If OOM (Out of Memory):
        *   Reduce `per_device_train_batch_size` to 1.
        *   Enable `gradient_checkpointing=True` in `TrainingArguments`.

## 6. Cloud GPU Strategy (24GB+ VRAM)

If you rent a **A10G, RTX 3090/4090, or A100 (24GB-40GB)**, you can skip the "low memory" optimizations and prioritize **Speed** and **Quality**.

### **A. Changes to Training Config**
You no longer need 4-bit loading or aggressive gradient checkpointing (which slows down training).

1.  **Disable Quantization:** Load model in full `float16` or `bfloat16` (if Ampere+).
    ```python
    # Remove load_in_4bit=True
    model = Wav2Vec2ForCTC.from_pretrained(..., torch_dtype=torch.float16)
    ```
2.  **Disable Gradient Checkpointing:**
    ```python
    training_args = TrainingArguments(..., gradient_checkpointing=False) # 20-30% faster
    ```
3.  **Increase Batch Size:**
    *   Set `per_device_train_batch_size=8` or `16`.
    *   This provides a much more stable gradient estimate than batch size 1 or 2.

### **B. Full Finetuning vs. LoRA**
With 24GB, you *can* technically fully finetune the 1B model (unfreeze all layers).
*   **Recommendation:** **Stick to LoRA for < 1000 examples.**
*   **Why?** A 1B parameter model will memorize 500 examples instantly (overfitting). LoRA acts as a regularizer.
*   **When to Full Finetune:** Once you have **10+ hours** of data (approx. 2000+ examples).

### **C. Optimal Data & Time Estimates**

**1. How much data is "Optimum"?**
For "Banglish" (a dialet/mix not in base training):
*   **Minimum (Functionality):** 100 examples (~15 mins). *Result: Works for specific phrases.*
*   **Robust (Variance):** 500-800 examples (~1-2 hours). *Result: Generalizes to new numbers/sentences.*
*   **Production (High Accuracy):** 2000+ examples (~5+ hours). *Result: Professional grade.*

**2. Training Time Estimate (on 24GB GPU)**
Assuming 500 examples (avg 5s each = ~42 mins of audio):

| Task | Configuration | Est. Time (10 Epochs) |
| :--- | :--- | :--- |
| **LoRA (Optimized)** | Batch 16, fp16 | **~5 - 8 Minutes** |
| **Full Finetune** | Batch 8, fp16 | **~15 - 20 Minutes** |

**Conclusion:** Training is extremely fast on this scale. You can iterate quickly.
*   **Workflow:** Train 500 examples -> Test -> Add 100 failed examples -> Retrain (Active Learning).

This approach gives you the best chance of success with limited data and hardware.
