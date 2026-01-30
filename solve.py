import collections
from dataclasses import dataclass, field
import torch


@dataclass
class DecodingArguments:
    encode_format: str = field(
        default="instruct",  # choices=["instruct", "qa", ]
    )
    max_new_tokens: int = field(
        default=512,
    )
    decoding: str = field(
        default="greedy",  # choices=["greedy", "cot", ],
    )
    cot_n_branches: int = field(default=10)
    cot_aggregate: str = field(
        default="sum"  # choices=["max", "sum", "self_consistency", ]
    )


def greedy_decoding_solve(model, tokenizer, task, batch, args: DecodingArguments):
    """
    Decodificação simples (Greedy) sem Chain-of-Thought complexo.
    """
    gen_ids = model.generate(
        input_ids=batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda(),
        do_sample=False, max_new_tokens=args.max_new_tokens,
    )
    ret = []
    for i in range(len(gen_ids)):
        text = tokenizer.decode(gen_ids[i, batch.input_ids.shape[-1]:], skip_special_tokens=True)
        answer, answer_span = task.extract_model_answer(text)
        
        # Limpeza básica se encontrar resposta
        clean_text = text
        if answer_span is not None:
            clean_text = text[:answer_span[1]]

        # Retorna estrutura compatível com o main.py
        ret.append({
            'answer_greedy': answer,
            'text_greedy': clean_text,
            'score_greedy': 1.0,  # Greedy padrão não tem score de confiança relativo
            'span_greedy': answer_span,
            
            # Campos vazios para manter compatibilidade caso o main tente acessar
            'candidates': [],
            'answer_scores': None,
            'answer_max_path': None, 
            'answer_aggregated': None
        })
    return ret


def cot_decoding_solve(model, tokenizer, task, batch, args: DecodingArguments):
    """
    Decodificação Chain-of-Thought com Múltiplos Ramos (Branches).
    Calcula Greedy (Branch 0), Max Path (Melhor Score) e Aggregated (Consenso).
    """
    bsz = batch.input_ids.shape[0]
    n_branches = args.cot_n_branches

    # --- 1. GERAÇÃO DOS RAMOS ---
    # Passo 1: Gera o primeiro token para abrir N caminhos
    input_ids = model.generate(
        input_ids=batch.input_ids.cuda(), attention_mask=batch.attention_mask.cuda(),
        do_sample=False, num_beams=n_branches, num_return_sequences=n_branches, max_new_tokens=1,
        min_new_tokens=1, 
    )
    
    # Prepara máscara de atenção expandida
    attention_mask = batch.attention_mask.cuda().repeat_interleave(n_branches, 0)
    attention_mask = torch.cat([
        attention_mask, torch.ones((len(attention_mask), 1), device=attention_mask.device, dtype=attention_mask.dtype),
    ], dim=1)

    # Passo 2: Gera o restante do raciocínio para cada ramo
    outputs = model.generate(
        input_ids=input_ids, attention_mask=attention_mask, do_sample=False, max_new_tokens=args.max_new_tokens - 1,
        output_logits=True, return_dict_in_generate=True,
    )
    
    # Processa IDs e Logits para calcular probabilidades
    gen_ids = outputs['sequences'][:, input_ids.shape[1] - 1:].reshape(bsz, n_branches, -1)
    gen_probs = torch.stack(outputs['logits'], dim=1).softmax(-1)
    n_vocab = gen_probs.shape[-1]
    
    # Adiciona probabilidade uniforme para o primeiro token (já gerado antes)
    gen_probs = torch.cat(
        [torch.full((bsz * n_branches, 1, n_vocab), 1 / n_vocab, dtype=gen_probs.dtype, device=gen_probs.device),
         gen_probs, ], dim=1)
    gen_probs = gen_probs.reshape(bsz, n_branches, -1, n_vocab)

    # --- FUNÇÕES AUXILIARES ---
    def decode_with_offsets(generation_ids):
        tokens = tokenizer.convert_ids_to_tokens(generation_ids)
        text = ''
        offsets = []
        for i in range(len(generation_ids)):
            if tokens[i] == tokenizer.eos_token:
                break
            text = tokenizer.convert_tokens_to_string(tokens[:i + 1])
            offsets.append(len(text))
        offsets += [-1 for _ in range(len(tokens) - len(offsets))]
        return text, offsets

    def match_answer_span(answer_span, offsets):
        answer_s, answer_e = answer_span
        inds = []
        for i, offset in enumerate(offsets):
            if answer_s < offset:
                inds.append(i)
                if answer_e <= offset:
                    break
        return inds

    def get_cot_score(probs):
        # Score = Média da diferença entre a prob do Top-1 e Top-2 (Margem de Confiança)
        probs = probs.topk(k=2, dim=-1, sorted=True).values
        score = (probs[:, 0] - probs[:, 1]).mean()
        return float(score)

    # --- 2. EXTRAÇÃO E CÁLCULO DAS MÉTRICAS ---
    ret = []
    for i in range(bsz):
        candidates = []
        for j in range(n_branches):
            text, offsets = decode_with_offsets(gen_ids[i, j])
            answer, answer_span = task.extract_model_answer(text)
            
            # --- LIMPEZA DE TEXTO ---
            # Corta o texto imediatamente após a resposta encontrada para remover repetições
            clean_text = text
            if answer_span is not None:
                clean_text = text[:answer_span[1]]
            # ------------------------

            if answer_span is None:
                candidates.append({
                    'text': clean_text, 
                    'answer': answer, 
                    'answer_span': answer_span, 
                    'score': 0
                })
            else:
                answer_tokens = match_answer_span(answer_span, offsets)
                answer_probs = gen_probs[i, j][torch.LongTensor(answer_tokens).cuda()]
                cot_score = get_cot_score(answer_probs)
                candidates.append({
                    'text': clean_text, 
                    'answer': answer, 
                    'answer_span': answer_span, 
                    'score': cot_score
                })
        
        # Objeto base de retorno
        result = {'candidates': candidates}

        # === 1. GREEDY (Primeiro ramo do Beam Search) ===
        # Assume-se que o índice 0 é o de maior probabilidade a priori
        result['answer_greedy'] = candidates[0]['answer']
        result['score_greedy']  = candidates[0]['score']
        result['text_greedy']   = candidates[0]['text']
        result['span_greedy']   = candidates[0]['answer_span']

        # === 2. MAX PATH (Caminho Único com Maior Confiança) ===
        best_candidate = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
        result['answer_max_path'] = best_candidate['answer']
        result['score_max_path']  = best_candidate['score']
        result['text_max_path']   = best_candidate['text']
        result['span_max_path']   = best_candidate['answer_span']

        # === 3. AGGREGATED (Consenso / Soma dos Scores) ===
        answer_scores = {}
        for candidate in candidates:
            ans = candidate['answer']
            answer_scores[ans] = answer_scores.get(ans, 0) + candidate['score']
        
        # Resposta vencedora pela soma
        best_sum_answer = sorted(answer_scores.items(), key=lambda x: x[1], reverse=True)[0][0]
        
        # Escolhe o "Texto Representante" para o método Agregado:
        # Pega o caminho com maior score individual DENTRE os que votaram na resposta vencedora
        representative_cand = max(
            [c for c in candidates if c['answer'] == best_sum_answer],
            key=lambda x: x['score'],
            default=candidates[0]
        )

        result['answer_aggregated'] = best_sum_answer
        result['text_aggregated']   = representative_cand['text']
        result['span_aggregated']   = representative_cand['answer_span']
        result['score_aggregated']  = representative_cand['score'] # Score do representante
        result['answer_scores']     = answer_scores # Dicionário com a votação completa

        # Define a resposta "oficial" (para cálculo rápido de acurácia no terminal)
        result['answer'] = best_sum_answer

        ret.append(result)

    return ret


def solve(model, tokenizer, task, batch, args: DecodingArguments):
    if args.decoding == 'greedy':
        return greedy_decoding_solve(model, tokenizer, task, batch, args)
    elif args.decoding == 'cot':
        return cot_decoding_solve(model, tokenizer, task, batch, args)
    else:
        raise ValueError("Invalid decoding " + args.decoding)