import argparse
from pathlib import Path
import torch
from collections import namedtuple
from transformers import BertConfig, DPRConfig, DPRContextEncoder, DPRQuestionEncoder

CheckpointState = namedtuple(
    "CheckpointState",
    ["model_dict", "optimizer_dict", "scheduler_dict", "offset", "epoch", "encoder_params"]
)

def load_states_from_checkpoint(model_file: str):
    print(f"Reading saved model from {model_file}")
    state_dict = torch.load(model_file, map_location='cpu')
    return CheckpointState(**state_dict)

def convert_dpr_checkpoint(args):
    output_dir = Path(args.output_dir)
    ctx_dir = output_dir / "ctx_encoder"
    question_dir = output_dir / "question_encoder"
    
    ctx_dir.mkdir(parents=True, exist_ok=True)
    question_dir.mkdir(parents=True, exist_ok=True)
    
    base_config = BertConfig.get_config_dict("google-bert/bert-base-uncased")[0]
    dpr_config = DPRConfig(**base_config)
    
    ctx_encoder = DPRContextEncoder(dpr_config)
    question_encoder = DPRQuestionEncoder(dpr_config)
    
    print(f"Loading checkpoint from {args.checkpoint}")
    saved_state = load_states_from_checkpoint(args.checkpoint)
    
    ctx_state_dict = {}
    for key, value in saved_state.model_dict.items():
        if key.startswith("ctx_model."):
            new_key = key[len("ctx_model."):]
            if not new_key.startswith("encode_proj."):
                new_key = "bert_model." + new_key
            ctx_state_dict[new_key] = value
    
    q_state_dict = {}
    for key, value in saved_state.model_dict.items():
        if key.startswith("question_model."):
            new_key = key[len("question_model."):]
            if not new_key.startswith("encode_proj."):
                new_key = "bert_model." + new_key
            q_state_dict[new_key] = value
    
    position_ids = torch.arange(512).expand((1, -1))
    ctx_state_dict["bert_model.embeddings.position_ids"] = position_ids
    q_state_dict["bert_model.embeddings.position_ids"] = position_ids
    
    ctx_encoder.ctx_encoder.load_state_dict(ctx_state_dict, strict=False)
    question_encoder.question_encoder.load_state_dict(q_state_dict, strict=False)
    
    ctx_encoder.save_pretrained(ctx_dir)
    question_encoder.save_pretrained(question_dir)
    
    print(f"Saved context encoder to {ctx_dir}")
    print(f"Saved question encoder to {question_dir}")

def parse_args():
    parser = argparse.ArgumentParser(description='Convert DPR checkpoint to HuggingFace format')
    parser.add_argument('--checkpoint', required=True,
                       help='Path to DPR checkpoint file')
    parser.add_argument('--output_dir', required=True,
                       help='Output directory for converted models')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    convert_dpr_checkpoint(args)