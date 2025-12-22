
import os
import sys
import torch
import warnings
from transformers import (
    Wav2Vec2ForCTC,
    AutoProcessor,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Union, Any
from datasets import load_dataset, Audio, DatasetDict
from peft import LoraConfig, get_peft_model, TaskType

# Suppress warnings
warnings.filterwarnings("ignore")

@dataclass
class DataCollatorCTCWithPadding:
    processor: Any
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lenghts and need
        # different padding methods
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        labels_batch = self.processor.tokenizer.pad(
            label_features,
            padding=self.padding,
            return_tensors="pt",
        )

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        batch["labels"] = labels
        return batch

# --- Configuration ---
MODEL_ID = "facebook/mms-1b-all"
DATASET_DIR = "bangla_actual_dataset"
OUTPUT_DIR = "mms-1b-bangla-lora"
# CSV file name in the dataset directory
CSV_FILE = "generated_dataset.csv" 

# Hardware check
if not torch.cuda.is_available():
    print("WARNING: CUDA is not available. Training will be extremely slow on CPU.")
    DEVICE = "cpu"
else:
    DEVICE = "cuda"
    print(f"Using GPU: {torch.cuda.get_device_name(0)}")

def main():
    # 1. Load Dataset
    csv_path = os.path.join(DATASET_DIR, CSV_FILE)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Could not find dataset CSV at {csv_path}")

    print(f"Loading dataset from {csv_path}...")
    dataset = load_dataset("csv", data_files=csv_path, split="train")
    
    # Resolve relative audio paths
    def resolve_audio_path(batch):
        batch["audio"] = os.path.join(DATASET_DIR, batch["audio_path"])
        return batch

    dataset = dataset.map(resolve_audio_path)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))

    # Split into Train/Test (90/10)
    dataset_split = dataset.train_test_split(test_size=0.1)
    print(f"Train size: {len(dataset_split['train'])}, Test size: {len(dataset_split['test'])}")

    # 2. Processor & Tokenizer
    print(f"Loading processor for {MODEL_ID}...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, target_lang="ben")
    tokenizer = processor.tokenizer

    # 3. Data Preparation
    def prepare_dataset(batch):
        audio = batch["audio"]
        # Process audio to input values
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        
        # Process text to label ids
        # MMS processor handles normalization implicitly if configured, but for safety with Bangla:
        with processor.as_target_processor():
            batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    print("Preprocessing dataset...")
    encoded_dataset = dataset_split.map(
        prepare_dataset, 
        remove_columns=dataset_split["train"].column_names,
        num_proc=1 # Use 1 for stability with small data
    )

    # 4. Model Loading with LoRA
    print("Loading model...")
    model = Wav2Vec2ForCTC.from_pretrained(
        MODEL_ID,
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.1,
        target_lang="ben", # Initialize correct adapter for Bangla
        ignore_mismatched_sizes=True,
        torch_dtype=torch.float32 # Load in fp32 for stability with un-frozen head
    )
    
    # Freeze base model parameters
    model.freeze_base_model()
    
    # Init Adapter Layers (MMS specific)
    model.init_adapter_layers()

    # Apply LoRA
    peft_config = LoraConfig(
        inference_mode=False,
        r=32,            # Rank
        lora_alpha=64,   # Alpha
        lora_dropout=0.1, 
        target_modules=["q_proj", "v_proj", "k_proj", "out_proj"], # Target attention projection layers
        modules_to_save=["lm_head"] # TRAIN the output head (it was random!)
    )
    
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # PATCH: Fix for PEFT saving issue with Wav2Vec2
    # PEFT tries to access input embeddings to check if they need saving, but Wav2Vec2 doesn't implement it.
    if not hasattr(model, "get_input_embeddings") or True: # Force patch
        model.get_input_embeddings = lambda: None

    # 5. Data Collator
    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

    # 6. Training Arguments (Optimized for 12GB VRAM)
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        group_by_length=True,
        per_device_train_batch_size=4, # Increased for 12GB VRAM
        gradient_accumulation_steps=4, # Effective batch size = 16
        eval_strategy="steps",
        num_train_epochs=60,           # Increased for better accuracy
        fp16=True,                     # Essential for 12GB
        gradient_checkpointing=True,   # Saves VRAM at cost of speed
        save_steps=100,
        eval_steps=100,
        logging_steps=10,
        learning_rate=3e-4,
        warmup_steps=100,
        save_total_limit=2,
        push_to_hub=False,
        dataloader_num_workers=2,
        report_to=["tensorboard"],
        load_best_model_at_end=True,
    )

    # 7. Trainer
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        train_dataset=encoded_dataset["train"],
        eval_dataset=encoded_dataset["test"],
        tokenizer=processor.feature_extractor,
    )

    print("Starting training...")
    # Check for existing checkpoint
    checkpoint = None
    if os.path.exists(OUTPUT_DIR):
        checkpoints = [d for d in os.listdir(OUTPUT_DIR) if d.startswith("checkpoint-")]
        if checkpoints:
            # Sort by step number
            checkpoints.sort(key=lambda x: int(x.split("-")[1]))
            checkpoint = os.path.join(OUTPUT_DIR, checkpoints[-1])
            print(f"Resuming from checkpoint: {checkpoint}")

    trainer.train(resume_from_checkpoint=checkpoint)
    
    print("Training completed. Saving final model...")
    trainer.save_model(os.path.join(OUTPUT_DIR, "final_model"))
    processor.save_pretrained(os.path.join(OUTPUT_DIR, "final_model"))

if __name__ == "__main__":
    main()
