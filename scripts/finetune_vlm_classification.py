"""
This script finetunes a VLM (Vision-Language Model) using LoRA for reaction distribution prediction.
"""

import os
import sys
import argparse
import logging
import numpy as np
import json
import random
from typing import Dict, List, Any
from dataclasses import dataclass
from tqdm import tqdm
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, AutoModelForPreTraining, TrainingArguments, Trainer, ProcessorMixin
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import transformers
import wandb

# Local imports
from src.dataset import Video2Reaction, REACTION_CLASSES
from src.zoo import VLM_ZOO

# Emotion synonyms for data augmentation
# This helps the model learn robust representations by exposing it to different names for the same emotion
EMOTION_SYNONYMS = {
    'curiosity': ['interest', 'inquisitiveness', 'wonder', 'curiosity'],
    'admiration': ['respect', 'esteem', 'appreciation', 'admiration'],
    'sadness': ['sorrow', 'melancholy', 'unhappiness', 'sadness'],
    'embarrassment': ['shame', 'awkwardness', 'discomfort', 'embarrassment'],
    'grief': ['mourning', 'heartache', 'bereavement', 'grief'],
    'realization': ['awareness', 'understanding', 'recognition', 'realization'],
    'approval': ['agreement', 'endorsement', 'acceptance', 'approval'],
    'caring': ['compassion', 'concern', 'empathy', 'caring'],
    'disgust': ['revulsion', 'distaste', 'repulsion', 'disgust'],
    'relief': ['comfort', 'ease', 'reassurance', 'relief'],
    'confusion': ['bewilderment', 'perplexity', 'puzzlement', 'confusion'],
    'nervousness': ['anxiety', 'unease', 'tension', 'nervousness'],
    'annoyance': ['irritation', 'frustration', 'vexation', 'annoyance'],
    'joy': ['happiness', 'delight', 'pleasure', 'joy'],
    'amusement': ['entertainment', 'fun', 'humor', 'amusement'],
    'fear': ['dread', 'terror', 'fright', 'fear'],
    'anger': ['rage', 'fury', 'wrath', 'anger'],
    'disapproval': ['objection', 'criticism', 'condemnation', 'disapproval'],
    'excitement': ['enthusiasm', 'thrill', 'exhilaration', 'excitement'],
    'disappointment': ['letdown', 'dismay', 'disillusionment', 'disappointment'],
    'surprise': ['astonishment', 'shock', 'amazement', 'surprise']
}


class VLMReactionDataset(Dataset):
    """
    Dataset wrapper for VLM finetuning with reaction prediction.
    The model learns to predict audience reactions from the perspective of a viewer watching a movie clip.
    """
    def __init__(self, metadata_file_path: str, key_frame_dir: str, processor, output_type="reaction_distribution",
                 apply_augmentation=False, synonym_prob=0.3, reorder_prob=0.5, ablate_text=False, ablate_visual=False):
        """
        Args:
            metadata_file_path: Path to the metadata JSON file
            key_frame_dir: Directory containing key frames
            processor: VLM processor for tokenization
            output_type: Type of reaction outcome to predict
            apply_augmentation: Whether to apply taxonomy augmentation (only during training)
            synonym_prob: Probability of using a synonym for emotion names
            reorder_prob: Probability of reordering reaction options
            ablate_visual: If set, replace video frames with blank frames so the model
                receives no visual signal (text/clip_description only)
        """
        self.base_dataset = Video2Reaction(
            metadata_file_path, 
            input_type="vlm", 
            key_frame_dir=key_frame_dir, 
            lazy_load=True # TODO: change to False if memory allows and if speed is too slow
        )
        self.processor = processor
        self.output_type = output_type
        self.apply_augmentation = apply_augmentation
        self.synonym_prob = synonym_prob
        self.reorder_prob = reorder_prob
        self.ablate_text = ablate_text
        self.ablate_visual = ablate_visual

    def get_category_name(self, category: str) -> str:
        """Get category name, potentially using a synonym."""
        if self.apply_augmentation and random.random() < self.synonym_prob:
            synonyms = EMOTION_SYNONYMS.get(category, [category])
            return random.choice(synonyms)
        return category
    
    def get_reaction_options(self):
        """Get reaction options, potentially reordered for augmentation."""
        reactions = list(REACTION_CLASSES)
        
        # Apply reordering if augmentation is enabled
        if self.apply_augmentation and random.random() < self.reorder_prob:
            random.shuffle(reactions)
        
        # Apply synonym substitution
        if self.apply_augmentation:
            reactions = [self.get_category_name(r) for r in reactions]
        
        return reactions
        
    def __len__(self):
        return len(self.base_dataset)
    
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        
        # Get reaction options (potentially augmented)
        reaction_options = self.get_reaction_options()
        
        # Format the conversation for training
        conversation = self._create_conversation(sample, reaction_options)
        
        # Process the input
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=False)
        video_inputs = sample["video_input"]
        
        # Sample 16 frames evenly if more frames available
        num_frames = len(video_inputs)
        if num_frames > 16:
            selected_frames = np.linspace(0, num_frames - 1, 16, dtype=int)
            video_inputs = [video_inputs[i] for i in selected_frames]
        
        if num_frames < 16:
            # Pad right away with last frame if less than 16 frames
            last_frame = video_inputs[-1]
            while len(video_inputs) < 16:
                video_inputs.append(last_frame)

        if self.ablate_visual:
            # Replace frames with blank frames so no visual information reaches the model,
            # while keeping the video placeholder/frame count intact for the processor.
            video_inputs = [Image.new(frame.mode, frame.size) for frame in video_inputs]

        # Get target distribution
        if self.output_type == "reaction_distribution":
            target = np.zeros(len(REACTION_CLASSES))
            for reaction, distribution in sample["reaction_distribution"].items():
                if reaction in REACTION_CLASSES:
                    target[REACTION_CLASSES.index(reaction)] = distribution
            
            # If augmentation applied, reorder target to match augmented options
            if self.apply_augmentation:
                # Create mapping from original to augmented order
                reordered_target = np.zeros(len(reaction_options))
                for i, augmented_reaction in enumerate(reaction_options):
                    # Find the original reaction (handle synonyms)
                    original_reaction = None
                    for orig_reaction in REACTION_CLASSES:
                        if augmented_reaction in EMOTION_SYNONYMS.get(orig_reaction, [orig_reaction]):
                            original_reaction = orig_reaction
                            break
                    if original_reaction and original_reaction in REACTION_CLASSES:
                        orig_idx = REACTION_CLASSES.index(original_reaction)
                        reordered_target[i] = target[orig_idx]
                target = reordered_target
        else:
            # For dominant reaction, create one-hot encoding
            target = np.zeros(len(REACTION_CLASSES))
            target[REACTION_CLASSES.index(sample["reaction_dominant"])] = 1.0
            
            # If augmentation applied, reorder target to match augmented options
            if self.apply_augmentation:
                dominant_reaction = sample["reaction_dominant"]
                # Find new index in augmented options
                for i, augmented_reaction in enumerate(reaction_options):
                    if augmented_reaction in EMOTION_SYNONYMS.get(dominant_reaction, [dominant_reaction]):
                        reordered_target = np.zeros(len(reaction_options))
                        reordered_target[i] = 1.0
                        target = reordered_target
                        break
        
        return {
            "videos": video_inputs,
            "messages": conversation,
            "labels": torch.tensor(target, dtype=torch.float16),
            "video_id": sample["video_id"]
        }
    
    def _create_conversation(self, sample, reaction_options=None):
        """
        Create the conversation format for training.
        Emphasizes the perspective of an audience member watching the movie clip.
        
        Args:
            sample: Data sample containing video information
            reaction_options: List of reaction options (potentially augmented with synonyms/reordering)
        """
        if reaction_options is None:
            reaction_options = list(REACTION_CLASSES)
        
        # Create option labels A-U for reaction classes
        option_labels = [chr(65 + i) for i in range(len(reaction_options))]
        option_dict = {label: reaction for label, reaction in zip(option_labels, reaction_options)}
        
        options_text = ""
        for label, reaction in option_dict.items():
            options_text += f"{label}. {reaction}\n"
        
        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "You are an audience member watching a movie clip. You will be provided with the video, some context about the scene in the video, and a taxonomy of possible emotional reactions."},
                    {"type": "video"},
                    *([{"type": "text", "text": f"In this scene, {sample['clip_description']}."}] if not self.ablate_text else []),
                    {"type": "text", "text": f"Taxonomy: \n{options_text}"},
                    {"type": "text", "text": f"After watching the movie clip, how would you feel? Choose one letter corresponding to the emotion that best describes your reaction."},
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Answer: "}
                ]
            }
        ]
        
        return conversation


class ReactionDistributionTrainer(Trainer):
    """
    Custom trainer for reaction distribution prediction with VLMs.
    Supports multiple loss types: KL divergence, cross-entropy, MSE, MAE.
    """
    def __init__(self, *args, processor=None, debug=False, loss_type="kl_divergence", **kwargs):
        super().__init__(*args, **kwargs)
        self.processor = processor
        self.debug = debug
        self.debug_step_count = 0
        self.loss_type = loss_type
        
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override to stop after 1 batch in debug mode"""
        loss = super().training_step(model, inputs, num_items_in_batch=num_items_in_batch)
        
        if self.debug:
            self.debug_step_count += 1
            if self.debug_step_count >= 1:
                # Force training to stop by setting epoch to max
                self.state.epoch = self.args.num_train_epochs
        
        return loss
        
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        Custom loss computation for reaction distribution prediction.
        Supports multiple loss types: kl_divergence, cross_entropy, mse, mae.
        """
        labels = inputs.pop("labels")
        video_ids = inputs.pop("video_id", None)
        
        # Ensure pixel_values is present (should never be None)
        assert "pixel_values" in inputs and inputs["pixel_values"] is not None, \
            "pixel_values must be present in inputs and cannot be None"
        
        # Forward pass
        # for both llava and qwen2_vl, the video input key is "pixel_values_videos"
        inputs["pixel_values_videos"] = inputs.pop("pixel_values")

        outputs = model(**inputs)
        logits = outputs.logits
        
        # Get logits for the last token (where we expect the answer)
        last_token_logits = logits[:, -1, :]
        
        # Get option token IDs (A-U)
        option_labels = [chr(65 + i) for i in range(len(REACTION_CLASSES))]
        option_token_ids = self.processor.tokenizer(option_labels, add_special_tokens=False)["input_ids"]
        option_token_ids = torch.tensor([item[0] for item in option_token_ids]).to(logits.device)
        
        # Extract logits for reaction options only
        reaction_logits = last_token_logits[:, option_token_ids]
        
        # Compute loss based on specified loss type
        if self.loss_type == "kl_divergence":
            # Apply log_softmax for KL divergence
            log_probs = torch.nn.functional.log_softmax(reaction_logits, dim=-1)
            # KL divergence loss between predicted distribution and target distribution
            loss = torch.nn.functional.kl_div(log_probs, labels.to(log_probs.dtype), reduction='batchmean')
        
        elif self.loss_type == "cross_entropy":
            # Cross-entropy loss (treating it as classification)
            # Apply softmax to get probabilities
            probs = torch.nn.functional.softmax(reaction_logits, dim=-1)
            # Cross-entropy: -sum(target * log(pred))
            loss = -torch.sum(labels.to(probs.dtype) * torch.log(probs + 1e-10)) / labels.shape[0]
        
        elif self.loss_type == "mse":
            # Mean Squared Error between predicted and target distributions
            probs = torch.nn.functional.softmax(reaction_logits, dim=-1)
            loss = torch.nn.functional.mse_loss(probs, labels.to(probs.dtype))
        
        elif self.loss_type == "mae":
            # Mean Absolute Error (L1 loss) between predicted and target distributions
            probs = torch.nn.functional.softmax(reaction_logits, dim=-1)
            loss = torch.nn.functional.l1_loss(probs, labels.to(probs.dtype))
        
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}. Must be one of: kl_divergence, cross_entropy, mse, mae")
        
        return (loss, outputs) if return_outputs else loss


def get_model_and_processor(model_name: str, use_lora: bool = True):
    """
    Load VLM model with optional LoRA configuration
    """
    model_path = VLM_ZOO[model_name]["model_path"]
    
    # Load processor
    processor = AutoProcessor.from_pretrained(model_path)
    if model_name == "llava_next":
        processor.patch_size = 14
    
    # Configure for left padding (important for generation and causal LM)
    processor.tokenizer.padding_side = "left"
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token
    
    # Load model
    if model_name == "qwen2_vl":
        from transformers import Qwen2_5_VLForConditionalGeneration
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
    else:
        model = AutoModelForPreTraining.from_pretrained(
            model_path, 
            torch_dtype=torch.float16,
            device_map="auto"
        )
    
    if use_lora:
        # Prepare model for k-bit training (optional, for memory efficiency)
        # model = prepare_model_for_kbit_training(model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=4,  # LoRA rank
            lora_alpha=32,  # LoRA alpha
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # Target attention layers
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        # Apply LoRA
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, processor


@dataclass
class DataCollatorForVideoLLModeling:
    """
    Data collator for vision-language modeling tasks with both image and video support.
    Based on TRL's DataCollatorForVisionLanguageModeling but extended to handle videos.
    
    This collator handles batching of raw vision-language examples that contain:
    - `videos` or `images`: Raw image/video data (PIL Images or similar)
    - `messages`: Conversational inputs with role and content
    - `text`: Standard text inputs (alternative to messages)
    - `labels`: Target labels for training (optional)
    
    The collator performs:
    1. Processing videos/images through the processor to get pixel_values
    2. Tokenizing text using the processor's tokenizer
    3. Padding for input_ids and attention_mask
    4. Creating labels from input_ids or using provided labels
    5. Proper batching of grid tensors for positional embeddings
    
    Args:
        processor (`ProcessorMixin`):
            The processor used to tokenize text and process images/videos.
        padding_side (`str`, optional, defaults to `"left"`):
            Side on which to apply padding. Use "left" for causal language models.
        return_tensors (`str`, optional, defaults to `"pt"`):
            Type of tensors to return. Currently only "pt" (PyTorch) is supported.
    """
    processor: ProcessorMixin
    padding_side: str = "left"
    return_tensors: str = "pt"
    
    def __call__(self, examples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Collate a batch of raw examples.
        
        Args:
            examples: List of dictionaries containing raw unprocessed examples.
        
        Returns:
            Dictionary with batched tensors ready for model input.
        """
        # Extract videos or images
        has_videos = "videos" in examples[0]
        has_images = "images" in examples[0]
        
        if has_videos:
            videos = [example["videos"] for example in examples]
            # All videos should be None if no actual videos
            if all(v is None or v == [] for v in videos):
                videos = None
        else:
            videos = None
            
        if has_images:
            images = [example["images"] for example in examples]
            if all(img_list == [] or img_list is None for img_list in images):
                images = None
        else:
            images = None
        
        # Use videos if available, otherwise images
        visual_inputs = videos if videos is not None else images
        
        # Extract text - handle both messages and text formats
        if "messages" in examples[0]:
            # Apply chat template to messages
            texts = self.processor.apply_chat_template(
                [example["messages"] for example in examples],
                add_generation_prompt=False
            )
        elif "text" in examples[0]:
            texts = [example["text"] for example in examples]
        else:
            raise ValueError("Examples must contain either 'messages' or 'text' key")
        
        # Process through the processor to get pixel_values and tokenized inputs
        processed = self.processor(
            videos=visual_inputs if has_videos else None,
            images=visual_inputs if has_images and not has_videos else None,
            text=texts,
            padding=True,
            padding_side=self.padding_side,
            return_tensors=self.return_tensors,
            add_special_tokens=False,
        )
        
        # Extract target labels if present
        has_custom_labels = "labels" in examples[0]
        if has_custom_labels:
            # Custom labels provided (e.g., for classification)
            target_labels = torch.stack([example["labels"] for example in examples])
        else:
            # Create labels from input_ids for language modeling
            target_labels = processed["input_ids"].clone()
            # Mask padding tokens
            target_labels[processed["attention_mask"] == 0] = -100
        
        # Build output dictionary
        output = {
            "input_ids": processed["input_ids"],
            "attention_mask": processed["attention_mask"],
            "labels": target_labels,
        }
        
        # Add pixel values with appropriate key
        if "pixel_values" in processed:
            output["pixel_values"] = processed["pixel_values"]
        if "pixel_values_videos" in processed:
            output["pixel_values"] = processed["pixel_values_videos"]
        
        # Add grid tensors if present (for Qwen2.5-VL)
        if "image_grid_thw" in processed:
            output["image_grid_thw"] = processed["image_grid_thw"]
        if "video_grid_thw" in processed:
            output["video_grid_thw"] = processed["video_grid_thw"]
        
        # Add video IDs for tracking (not used by model)
        if "video_id" in examples[0]:
            output["video_id"] = [example["video_id"] for example in examples]
        
        return output


def main():
    parser = argparse.ArgumentParser(description="Finetune VLM for reaction distribution prediction")
    
    # Data arguments
    parser.add_argument("--metadata_dir", type=str, required=True,
                        help="Directory containing train.json, val.json, test.json")
    parser.add_argument("--key_frame_dir", type=str, required=True,
                        help="Directory containing key frames")
    parser.add_argument("--output_type", type=str, default="reaction_distribution",
                        choices=["reaction_distribution", "reaction_dominant"],
                        help="Type of reaction outcome to predict")
    
    # Model arguments
    parser.add_argument("--model_name", type=str, default="llava_next",
                        help="Name of VLM model to use (must be in VLM_ZOO)")
    parser.add_argument("--use_lora", action="store_true", default=True,
                        help="Use LoRA for finetuning")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    
    # Training arguments
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save model checkpoints")
    parser.add_argument("--loss_type", type=str, default="kl_divergence",
                        choices=["kl_divergence", "cross_entropy", "mse", "mae"],
                        help="Loss function to use for training")
    parser.add_argument("--num_epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8,
                        help="Evaluation batch size")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--warmup_steps", type=int, default=100,
                        help="Number of warmup steps")
    parser.add_argument("--logging_steps", type=int, default=10,
                        help="Logging interval")
    parser.add_argument("--eval_steps", type=int, default=100,
                        help="Evaluation interval")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Checkpoint saving interval")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                        help="Number of gradient accumulation steps")
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping")
    
    # Taxonomy augmentation arguments
    parser.add_argument("--apply_augmentation", action="store_true",
                        help="Apply taxonomy augmentation (synonym and reordering) during training")
    parser.add_argument("--synonym_prob", type=float, default=0.3,
                        help="Probability of using synonym during training")
    parser.add_argument("--reorder_prob", type=float, default=0.5,
                        help="Probability of reordering categories during training")
    
    # Debug and other arguments
    parser.add_argument("--ablate_text", action="store_true",
                        help="If set, exclude clip_description from the VLM prompt")
    parser.add_argument("--ablate_visual", action="store_true",
                        help="If set, replace video frames with blank frames so the model is finetuned on text (clip_description) only")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: run with 1 batch, 1 epoch, and print example inputs")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)
    
    # Create run name and update output directory
    run_name = f"{args.model_name}_ep{args.num_epochs}_lr{args.learning_rate}"
    if args.use_lora:
        run_name += f"_lora_r{args.lora_r}_alpha{args.lora_alpha}"
    
    # Create subdirectory for loss type
    loss_type_dir = os.path.join(args.output_dir, args.loss_type)
    os.makedirs(loss_type_dir, exist_ok=True)
    
    # Update output directory to include loss type and run name
    args.output_dir = os.path.join(loss_type_dir, run_name)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save arguments
    with open(os.path.join(args.output_dir, "training_args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)
    
    # Initialize wandb
    wandb.init(
        project=f"reaction-video-finetuning-{args.model_name}",
        config=vars(args),
        name=run_name
    )
    
    # Load model and processor
    logger.info(f"Loading model: {args.model_name}")
    model, processor = get_model_and_processor(args.model_name, use_lora=args.use_lora)
    
    # Load datasets
    logger.info("Loading datasets...")
    train_dataset = VLMReactionDataset(
        os.path.join(args.metadata_dir, "train.json"),
        args.key_frame_dir,
        processor,
        args.output_type,
        apply_augmentation=args.apply_augmentation,
        synonym_prob=args.synonym_prob,
        reorder_prob=args.reorder_prob,
        ablate_text=args.ablate_text,
        ablate_visual=args.ablate_visual
    )

    val_dataset = VLMReactionDataset(
        os.path.join(args.metadata_dir, "val.json"),
        args.key_frame_dir,
        processor,
        args.output_type,
        apply_augmentation=False,  # Never apply augmentation to validation
        ablate_text=args.ablate_text,
        ablate_visual=args.ablate_visual
    )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Debug mode: print example conversation and limit dataset
    if args.debug:
        logger.info("="*80)
        logger.info("DEBUG MODE: Printing example input/output")
        logger.info("="*80)
        
        # Get first sample
        sample = train_dataset.base_dataset[0]
        
        # Override training parameters for debug
        args.num_epochs = 1
        args.logging_steps = 1
        args.eval_steps = 10000  # Don't eval during debug
        args.save_steps = 10000  # Don't save during debug
        
        logger.info(f"Debug: Will stop after 1 training batch")
    
    # Setup training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        eval_strategy="steps",
        save_strategy="steps",
        save_total_limit=3,  # Only keep 3 most recent checkpoints
        load_best_model_at_end=False,  # Disable to avoid issues with checkpoint loading
        max_grad_norm=args.max_grad_norm,
        fp16=True,
        save_safetensors=False,  # Save as pytorch bin files for better compatibility
        dataloader_num_workers=4,
        dataloader_pin_memory=False,
        remove_unused_columns=False,
        report_to=["wandb"],
        logging_dir=os.path.join(args.output_dir, "logs"),
    )
    
    # Create data collator
    data_collator = DataCollatorForVideoLLModeling(
        processor=processor,
        padding_side="left",  # Use left padding for causal LM
    )
    
    # Initialize trainer
    trainer = ReactionDistributionTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        processor=processor,
        debug=args.debug,
        loss_type=args.loss_type,
    )
    
    # Debug mode: print example batch after collate_fn
    if args.debug:
        logger.info("="*80)
        logger.info("DEBUG MODE: Printing example batch after collate_fn")
        logger.info("="*80)
        
        # Get the first batch
        train_dataloader = trainer.get_train_dataloader()
        first_batch = next(iter(train_dataloader))
        
        logger.info(f"\n--- Batch Information ---")
        logger.info(f"Batch size: {first_batch['input_ids'].shape[0]}")
        logger.info(f"Input IDs shape: {first_batch['input_ids'].shape}")
        logger.info(f"Attention mask shape: {first_batch['attention_mask'].shape}")
        if 'pixel_values' in first_batch:
            logger.info(f"Pixel values shape: {first_batch['pixel_values'].shape}")
        if 'image_grid_thw' in first_batch:
            logger.info(f"Image grid thw shape: {first_batch['image_grid_thw'].shape}")
        if 'video_grid_thw' in first_batch:
            logger.info(f"Video grid thw shape: {first_batch['video_grid_thw'].shape}")
        logger.info(f"Labels shape: {first_batch['labels'].shape}")
        
        # Decode first item's input_ids
        logger.info(f"\n--- First Item Decoded Text ---")
        first_input_ids = first_batch['input_ids'][0]
        decoded_text = processor.decode(first_input_ids, skip_special_tokens=False)
        logger.info(f"Decoded input:\n{decoded_text}")

        logger.info(f"\n--- First Item Pixel Values Info ---")
        logger.info(f"Pixel values shape: {first_batch['pixel_values'][0].shape}")
        
        logger.info(f"\n--- First Item Target Distribution ---")
        first_labels = first_batch['labels'][0]
        logger.info(f"Target distribution: {first_labels.cpu().numpy()}")
        
        logger.info("="*80)
    
    # Train the model
    logger.info("Starting training...")
    trainer.train()
    
    # Save the final model
    logger.info("Saving final model...")
    trainer.save_model(os.path.join(args.output_dir, "final_model"))
    processor.save_pretrained(os.path.join(args.output_dir, "final_model"))
    
    # Finish wandb run
    wandb.finish()
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()
