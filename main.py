#!/usr/bin/env python
# coding=utf-8

import json
import os
from dataclasses import field, dataclass
from functools import partial

import numpy as np
import torch
import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorForSeq2Seq, HfArgumentParser

from solve import DecodingArguments, solve
from task import GSMTask


@dataclass
class MainArguments:
    data_file: str = field(
        default="./gsm8k_data/test.jsonl",
        metadata={"help": "Caminho para o arquivo de dados de teste"}
    )
    model_name_or_path: str = field(
        default="mistralai/Mistral-7B-Instruct-v0.1",
        metadata={"help": "Modelo a ser usado"}
    )
    batch_size: int = field(
        default=4,  # Reduzido por segurança de memória com CoT
        metadata={"help": "Tamanho do batch de inferência"}
    )
    output_fname: str = field(
        default="outputs/model_predictions.jsonl",
        metadata={"help": "Nome base para os arquivos de saída"}
    )
    max_samples: int = field(
        default=-1,
        metadata={"help": "Limita o número de amostras para teste rápido. Use -1 para rodar tudo."}
    )


def encode_function(example, tokenizer, task):
    prompt = task.encode_prompt(example)
    tokenized = tokenizer(prompt, return_tensors='pt')
    input_ids = tokenized.input_ids
    attention_mask = torch.ones_like(input_ids)
    return {
        'input_ids': input_ids.flatten(),
        'attention_mask': attention_mask.flatten(),
    }


@torch.no_grad()
def main():
    parser = HfArgumentParser((MainArguments, DecodingArguments))
    main_args, decoding_args = parser.parse_args_into_dataclasses()
    
    # Inicializa a tarefa e o tokenizer
    task = GSMTask(encode_format=decoding_args.encode_format)
    
    # Adicione use_fast=False se tiver problemas com protobuf/versões antigas
    tokenizer = AutoTokenizer.from_pretrained(main_args.model_name_or_path, padding_side='left')
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Carrega o modelo (com suporte a 4-bit se bitsandbytes estiver instalado, ou padrão bfloat16)
    try:
        from transformers import BitsAndBytesConfig
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
        print("Carregando modelo em 4-bit...")
        model = AutoModelForCausalLM.from_pretrained(
            main_args.model_name_or_path, 
            quantization_config=quantization_config,
            device_map='auto'
        )
    except ImportError:
        print("BitsAndBytes não encontrado, carregando em bfloat16...")
        model = AutoModelForCausalLM.from_pretrained(
            main_args.model_name_or_path, 
            torch_dtype=torch.bfloat16, 
            device_map='auto'
        )

    # Carrega dataset
    raw_dataset = load_dataset("json", data_files={'test': main_args.data_file})['test']
    
    # Filtro opcional para testes rápidos
    if main_args.max_samples > 0:
        raw_dataset = raw_dataset.select(range(main_args.max_samples))
        print(f"DEBUG: Dataset limitado a {main_args.max_samples} amostras.")

    encode_function_partial = partial(
        encode_function,
        tokenizer=tokenizer,
        task=task,
    )
    
    lm_dataset = raw_dataset.map(
        encode_function_partial,
        batched=False,
        num_proc=4,
        remove_columns=[name for name in raw_dataset.column_names if name not in ["input_ids", "attention_mask"]],
        desc="Tokenizing data",
    )

    # Dataloader
    dataloader = DataLoader(
        lm_dataset, shuffle=False, batch_size=main_args.batch_size,
        collate_fn=DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="longest"),
    )

    # Listas para armazenar os datasets ricos
    dataset_greedy = []
    dataset_max = []
    dataset_agg = []

    correct_agg = 0
    total_samples = 0

    print("Iniciando geração...")
    pbar = tqdm.tqdm(dataloader, total=len(dataloader))
    
    # --- LOOP DE GERAÇÃO ---
    for batch in pbar:
        # Chama a função solve (que deve estar atualizada no solve.py)
        outputs = solve(model, tokenizer, task, batch, args=decoding_args)
        
        current_batch_size = len(outputs)
        
        for i, output in enumerate(outputs):
            idx_global = total_samples + i
            gt_example = raw_dataset[idx_global]
            question_text = gt_example['question']
            
            # ID Único para a chave (formato aninhado)
            sample_id = f"sample_{idx_global}"
            
            # Verifica corretude (apenas para métricas e metadados)
            is_greedy_correct = task.is_correct(gt_example, output.get('answer_greedy', ''))
            is_max_correct    = task.is_correct(gt_example, output.get('answer_max_path', ''))
            is_agg_correct    = task.is_correct(gt_example, output.get('answer_aggregated', ''))
            
            if is_agg_correct: correct_agg += 1

            # Função para construir o objeto Rico Aninhado
            def create_rich_entry(method_suffix, is_correct):
                # Determina chaves baseadas no método
                if method_suffix == 'greedy':
                    ans = output.get('answer_greedy')
                    txt = output.get('text_greedy')
                    score = output.get('score_greedy')
                    span = output.get('span_greedy')
                elif method_suffix == 'max':
                    ans = output.get('answer_max_path')
                    txt = output.get('text_max_path')
                    score = output.get('score_max_path')
                    span = output.get('span_max_path')
                else: # aggregated
                    ans = output.get('answer_aggregated')
                    txt = output.get('text_aggregated')
                    score = output.get('score_aggregated') # Score do representante
                    span = output.get('span_aggregated')

                # O objeto de valor (Value Object)
                rich_info = {
                    "question": question_text,
                    "generated_answer": ans,
                    "generated_text": txt,
                    "answer_span": span,
                    "score": score,
                    "is_correct": is_correct,
                    "method": method_suffix,
                    # Adiciona campo conversations se quiser manter compatibilidade futura com chat templates
                    "conversations": [
                        {"role": "user", "content": question_text},
                        {"role": "assistant", "content": txt}
                    ],
                    # Informações extras de debug
                    "candidates_count": len(output.get('candidates', [])),
                    "vote_scores": output.get('answer_scores') if method_suffix == 'aggregated' else None
                }
                
                # Retorna formato aninhado: { "sample_0": { ... } }
                return {sample_id: rich_info}

            # Adiciona aos datasets correspondentes
            dataset_greedy.append(create_rich_entry('greedy', is_greedy_correct))
            dataset_max.append(create_rich_entry('max', is_max_correct))
            dataset_agg.append(create_rich_entry('aggregated', is_agg_correct))

        total_samples += current_batch_size
        
        # Atualiza barra de progresso com acurácia atual do método agregado
        if total_samples > 0:
            pbar.set_postfix(acc_agg=f"{correct_agg / total_samples:.2%}")

    # --- SALVAMENTO ---
    base_fname = os.path.splitext(main_args.output_fname)[0]
    
    files_to_save = [
        (f"{base_fname}_greedy_rich.jsonl", dataset_greedy),
        (f"{base_fname}_max_rich.jsonl", dataset_max),
        (f"{base_fname}_aggregated_rich.jsonl", dataset_agg)
    ]

    print("\n" + "="*40)
    print("Processamento concluído.")
    for fname, dataset in files_to_save:
        print(f"Salvando {len(dataset)} entradas ricas em {fname}...")
        
        os.makedirs(os.path.dirname(fname) if os.path.dirname(fname) else '.', exist_ok=True)
        
        with open(fname, "w") as f:
            for entry in dataset:
                f.write(json.dumps(entry) + '\n')
    print("="*40 + "\n")


if __name__ == "__main__":
    main()